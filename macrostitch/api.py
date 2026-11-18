"""Workspace-scoped local REST service and command line entry point."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .pipeline import VERSION, Config, run_stack
from .storage import DEFAULT_CACHE_MAX_BYTES


class FocusRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    y: tuple[float, float]
    roi: tuple[float, float, float, float]
    tone_roi: tuple[float, float, float, float] | None = None
    tone_reference: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_region(self):
        if not 0 <= self.y[0] < self.y[1] <= 1:
            raise ValueError("Region y must be normalized and increasing")
        for roi in (self.roi, self.tone_roi):
            if roi is not None:
                x0, y0, x1, y1 = roi
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError("Region ROI must be normalized x0,y0,x1,y1 with positive area")
        return self


class LocalAlignmentAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: int = Field(ge=0, strict=True)
    frame: int = Field(ge=0, strict=True)


class LocalAlignmentRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    roi: tuple[float, float, float, float]
    fit_roi: tuple[float, float, float, float]
    validation_rois: tuple[tuple[float, float, float, float], ...] = Field(min_length=1, max_length=8)
    anchor: LocalAlignmentAnchor
    max_shift_px: float = Field(default=6, gt=.15, le=6)
    feather_px: int = Field(default=24, ge=1, le=64, strict=True)


class StackOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_fps: float = Field(default=10, gt=0, le=120)
    proxy_width: int = Field(default=960, ge=64, le=4096)
    max_frames: int = Field(default=48, ge=2, le=512)
    coverage: float = Field(default=.985, ge=.5, le=1)
    roi: tuple[float, float, float, float] | None = None
    fusion_mode: Literal["pyramid", "regional", "spatial"] = "pyramid"
    selection_mode: Literal["proxy", "native"] = "proxy"
    spatial_blend_px: float = Field(default=1, ge=0, le=4)
    spatial_smoothness: float = Field(default=.12, ge=0, le=1)
    local_alignment_regions: tuple[LocalAlignmentRegion, ...] | None = Field(default=None, min_length=1, max_length=8)
    required_frames: tuple[LocalAlignmentAnchor, ...] | None = Field(default=None, min_length=1, max_length=512)
    background_mode: Literal["reference", "transparent"] = "reference"
    background_opacity: float | None = Field(default=None, ge=0, le=1)
    background_margin_px: float = Field(default=0, ge=0, le=512)
    object_masks: tuple[dict, ...] | None = Field(default=None, min_length=1, max_length=32)
    canvas_mode: Literal["common", "reference"] = "common"
    focus_regions: tuple[FocusRegion, ...] | None = Field(default=None, min_length=1, max_length=32)
    score_normalization_roi: tuple[float, float, float, float] | None = None
    region_blend_px: float = Field(default=3, ge=0, le=32)
    matte_feather_px: float = Field(default=1.5, ge=.1, le=8)
    matte_erode_px: float = Field(default=1, ge=0, le=8)
    subject_only: bool = True
    subject_polygon: tuple[tuple[float, float], ...] | None = Field(default=None, min_length=3, max_length=128)
    pyramid_levels: int = Field(default=6, ge=1, le=12)
    threads: int = Field(default=4, ge=1, le=64)

    @model_validator(mode="after")
    def validate_roi(self):
        if self.local_alignment_regions is not None:
            if self.fusion_mode != "spatial" or self.selection_mode != "native":
                raise ValueError("Local alignment requires spatial fusion and native selection")
            from .local_alignment import normalize_local_regions
            normalize_local_regions([region.model_dump() for region in self.local_alignment_regions])
        for roi in (self.roi, self.score_normalization_roi):
            if roi is not None:
                x0, y0, x1, y1 = roi
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError("roi must be normalized x0,y0,x1,y1 with positive area")
        if self.fusion_mode == "regional" and self.focus_regions is None:
            raise ValueError("Regional fusion requires focus_regions")
        if self.focus_regions is not None:
            previous_end = 0.0
            for region in self.focus_regions:
                if region.y[0] != previous_end:
                    raise ValueError("Focus regions must cover [0,1] consecutively without gaps or overlap")
                previous_end = region.y[1]
                if region.tone_reference is not None and region.tone_reference >= len(self.focus_regions):
                    raise ValueError("tone_reference must index an existing focus region")
            if previous_end != 1.0:
                raise ValueError("Focus regions must cover [0,1] consecutively without gaps or overlap")
        if self.subject_polygon is not None:
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.subject_polygon):
                raise ValueError("subject_polygon coordinates must be normalized to [0,1]")
        try:
            Config(**self.model_dump()).validate()
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError(str(exc)) from exc
        return self


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[str] = Field(min_length=1, max_length=32)
    output_dir: str
    config: StackOptions = Field(default_factory=StackOptions)


def valid_timestamp(value):
    # Persisted state is local input too; bad dates must not break all job lists.
    return (value is None or (type(value) in (int, float)
                             and 0 <= value <= 253402300799 and math.isfinite(value)))


def create_app(workspace: Path, max_pending: int = 4) -> FastAPI:
    """Serve one running job and at most max_pending-1 queued jobs."""
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or max_pending < 1:
        raise ValueError("workspace must be a directory and max_pending positive")
    state_dir = root / ".macrostitch" / "jobs"
    if not state_dir.resolve().is_relative_to(root):
        raise ValueError("job state directory escapes workspace")
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    jobs: dict[str, dict] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="macrostitch")
    slots = threading.BoundedSemaphore(max_pending)
    closed = False

    def save(job):
        temporary = state_dir / (job["id"] + ".tmp")
        temporary.write_text(json.dumps(job, indent=2, default=str))
        temporary.replace(state_dir / (job["id"] + ".json"))

    for file in state_dir.glob("*.json"):
        try:
            job = json.loads(file.read_text())
            if job["id"] != file.stem or len(file.stem) != 32:
                continue
            # A partial/foreign JSON record must not poison every later submission.
            if (not isinstance(job.get("key"), str)
                    or not isinstance(job.get("output_dir"), str)
                    or not isinstance(job.get("inputs"), list)
                    or not all(isinstance(value, str) for value in job["inputs"])
                    or not isinstance(job.get("config"), dict)
                    or not isinstance(job.get("input_signatures"), list)
                    or job.get("status") not in {"queued", "running", "completed", "failed", "interrupted"}
                    or not isinstance(job.get("artifacts", {}), dict)
                    or not isinstance(job.get("artifact_signatures", {}), dict)
                    or not all(isinstance(name, str) and isinstance(value, str)
                               for name, value in job.get("artifacts", {}).items())
                    or any(not valid_timestamp(job.get(name))
                           for name in ("created_at", "started_at", "finished_at"))):
                continue
            if job["status"] in {"queued", "running"}:
                job.update(status="interrupted", error="Service restarted before completion")
                save(job)
            jobs[job["id"]] = job
        except (ValueError, KeyError, TypeError):
            continue

    def scoped(value: str, *, exists=False) -> Path:
        p = Path(value)
        try:
            p = (p if p.is_absolute() else root / p).resolve(strict=exists)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(400, "Path does not exist or cannot be resolved") from exc
        if not p.is_relative_to(root):
            raise HTTPException(400, "Path must remain inside workspace")
        return p

    def signature(path: Path):
        stat = path.stat()
        return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

    def output_safe(output, inputs):
        if output == root or output.is_relative_to(root / ".macrostitch"):
            raise HTTPException(400, "Choose a dedicated output directory")
        if (output.exists() and not output.is_dir()) or any(p.is_relative_to(output) for p in inputs):
            raise HTTPException(400, "Output must be a directory separate from input files")
        cache = output.parent / ".macrostitch" / "cache"
        if not cache.resolve().is_relative_to(root):
            raise HTTPException(400, "Cache directory escapes workspace")
        # Existing output aliases must never redirect pipeline writes into sources.
        input_identities = {(s.st_dev, s.st_ino) for s in (p.stat() for p in inputs)}
        for directory in (output, cache):
            if directory.exists():
                for path in directory.rglob("*"):
                    try:
                        entry = path.lstat()
                    except FileNotFoundError:
                        # Pipeline cache writes/eviction may finish while this
                        # submission enumerates entries; dequeue rechecks paths.
                        continue
                    if (stat.S_ISLNK(entry.st_mode)
                            or (stat.S_ISREG(entry.st_mode)
                                and (entry.st_dev, entry.st_ino) in input_identities)):
                        raise HTTPException(400, "Output/cache contains an unsafe file alias")

    def artifacts_current(job):
        try:
            return all(signature(scoped(path, exists=True)) == job.get("artifact_signatures", {}).get(name)
                       for name, path in job.get("artifacts", {}).items())
        except (OSError, HTTPException):
            return False

    def execute(job):
        try:
            with lock:
                job.update(status="running", started_at=time.time())
                save(job)

            def progress(event):
                with lock:
                    job["progress"] = event
                    save(job)

            # Recheck paths when dequeued, including symlinks changed while queued.
            inputs = [scoped(source, exists=True) for source in job["inputs"]]
            output = scoped(job["output_dir"])
            if str(output) != job["output_dir"]:
                raise ValueError("Output path changed while job was queued; submit again")
            output_safe(output, inputs)
            if [signature(p) for p in inputs] != job["input_signatures"]:
                raise ValueError("Input changed while job was queued; submit again")
            result = run_stack(job["inputs"], job["output_dir"], Config(**job["config"]), progress)
            artifacts = {}
            output = Path(job["output_dir"]).resolve()
            for name, value in result.get("artifacts", {}).items():
                if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                    continue
                path = Path(value)
                path = (path if path.is_absolute() else output / path).resolve()
                if path.is_relative_to(output) and path.is_relative_to(root) and path.is_file():
                    artifacts[name] = str(path)
            with lock:
                job.update(status="completed", result=result, artifacts=artifacts,
                           artifact_signatures={name: signature(Path(path)) for name, path in artifacts.items()},
                           finished_at=time.time())
                save(job)
        except Exception as exc:
            with lock:
                job.update(status="failed", error=str(exc), finished_at=time.time())
                save(job)
        finally:
            slots.release()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal closed
        yield
        with lock:
            closed = True
        executor.shutdown(wait=True)

    app = FastAPI(title="Macrostitch local API", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def editor():
        from .editor import EDITOR_HTML
        return HTMLResponse(EDITOR_HTML)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/workspace/files")
    def workspace_files(directory: str = "", limit: int = Query(default=200, ge=1, le=1000)):
        """Browse one directory at a time without scanning caches or following links."""
        folder = scoped(directory, exists=True)
        if not folder.is_dir() or any(part.startswith(".") for part in folder.relative_to(root).parts):
            raise HTTPException(400, "Choose a visible workspace directory")
        entries = []
        try:
            for path in folder.iterdir():
                if path.name.startswith(".") or path.is_symlink():
                    continue
                if path.is_dir():
                    kind = "directories"
                elif path.is_file() and path.suffix.lower() in {".mov", ".mp4", ".mkv", ".json"}:
                    kind = "recipes" if path.suffix.lower() == ".json" else "videos"
                else:
                    continue
                entries.append((kind, path))
        except OSError as exc:
            raise HTTPException(400, "Cannot read this workspace directory") from exc
        entries.sort(key=lambda entry: (entry[0] != "directories", entry[1].name.casefold(), entry[1].name))
        result = {"workspace_root": str(root),
                  "directory": folder.relative_to(root).as_posix() if folder != root else "",
                  "parent": (folder.parent.relative_to(root).as_posix() if folder.parent != root else "") if folder != root else None,
                  "directories": [], "videos": [], "recipes": [], "truncated": len(entries) > limit}
        for kind, path in entries[:limit]:
            result[kind].append({"name": path.name, "path": path.relative_to(root).as_posix()})
        return result

    @app.get("/workspace/recipe")
    def workspace_recipe(path: str):
        source = scoped(path, exists=True)
        if (not source.is_file() or source.suffix.lower() != ".json"
                or any(part.startswith(".") for part in source.relative_to(root).parts)):
            raise HTTPException(400, "Choose a recipe JSON file in the workspace")
        try:
            # Bound reading as well as parsing, including files that grow after stat.
            with source.open("rb") as stream:
                content = stream.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("Recipe exceeds the 1 MiB limit")
            config = json.loads(content)
            StackOptions.model_validate(config)
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(400, f"Invalid recipe: {exc}") from exc
        return {"name": source.name, "path": source.relative_to(root).as_posix(), "config": config}

    @app.get("/jobs")
    def recent_jobs(limit: int = Query(default=20, ge=1, le=100),
                    status: Literal["queued", "running", "completed", "failed", "interrupted"] | None = None):
        """Return compact recent-job summaries; full reports stay at /jobs/{id}."""
        fields = ("id", "status", "created_at", "started_at", "finished_at", "output_dir", "inputs", "progress", "error")

        def order(job):
            value = job.get("created_at")
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0

        with lock:
            selected = [job for job in jobs.values() if status is None or job["status"] == status]
            selected.sort(key=order, reverse=True)
            return {"jobs": [{field: job[field] for field in fields if field in job} for job in selected[:limit]],
                    "total": len(selected)}

    @app.post("/jobs", status_code=202)
    def submit(request: JobRequest):
        inputs = [scoped(p, exists=True) for p in request.inputs]
        if any(not p.is_file() or p.suffix.lower() not in {".mov", ".mp4", ".mkv"} for p in inputs):
            raise HTTPException(400, "Inputs must be local MOV, MP4, or MKV files")
        output = scoped(request.output_dir)
        output_safe(output, inputs)
        config = request.config.model_dump()
        input_signatures = [signature(p) for p in inputs]
        identity = {"pipeline_version": VERSION, "inputs": list(zip(map(str, inputs), input_signatures)),
                    "output_dir": str(output), "config": config}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with lock:
            if closed:
                raise HTTPException(503, "Service is shutting down")
            for job in jobs.values():
                if job.get("key") == key and job["status"] in {"queued", "running"}:
                    return dict(job)
                owned = Path(job["output_dir"])
                if job["status"] in {"queued", "running"} and (output.is_relative_to(owned) or owned.is_relative_to(output)):
                    raise HTTPException(409, "Another job owns this output directory")
            for job in jobs.values():
                if job.get("key") == key and job["status"] == "completed" and artifacts_current(job):
                    return dict(job)
            if not slots.acquire(blocking=False):
                raise HTTPException(429, "Job queue is full")
            job = {"id": uuid.uuid4().hex, "key": key, "status": "queued", "created_at": time.time(),
                   "inputs": list(map(str, inputs)), "output_dir": str(output), "config": config,
                   "input_signatures": input_signatures,
                   "progress": None, "artifacts": {}}
            try:
                save(job)
                jobs[job["id"]] = job
                executor.submit(execute, job)
            except Exception as exc:
                # Submission may fail after the job was persisted. Make it retryable
                # instead of returning an orphaned queued job on every future POST.
                job.update(status="failed", error=str(exc), finished_at=time.time())
                try:
                    if job["id"] in jobs:
                        save(job)
                finally:
                    slots.release()
                raise
            return dict(job)

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, "Job not found")
            return dict(jobs[job_id])

    @app.get("/jobs/{job_id}/artifacts/{filename}")
    def artifact(job_id: str, filename: str):
        job = status(job_id)
        value = job.get("artifacts", {}).get(filename)
        if job["status"] != "completed" or value is None:
            raise HTTPException(404, "Artifact not found")
        if not artifacts_current(job):
            raise HTTPException(409, "Job artifacts changed; submit the job again")
        path = scoped(value, exists=True)
        if not path.is_relative_to(Path(job["output_dir"]).resolve()) or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, filename=filename)

    return app


def format_progress(event):
    """Human-readable progress without implying a whole-job percentage."""
    stage = str(event.get("stage", "working")).replace("_", " ").capitalize()
    message = event.get("message", stage)
    if isinstance(event.get("completed"), (int, float)) and isinstance(event.get("total"), (int, float)) and event["total"] > 0:
        message += f" ({event['completed']:,}/{event['total']:,})"
    elif event.get("selected") is not None:
        message = f"Selected {event['selected']} source frames from {event.get('candidates', '?')} candidates"
    elapsed = event.get("elapsed_seconds")
    return f"[{elapsed:.1f}s] {message}" if isinstance(elapsed, (int, float)) else message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start workspace-scoped API on localhost")
    serve.add_argument("--workspace", type=Path, default=Path.cwd())
    serve.add_argument("--port", type=int, default=8000)
    stack = commands.add_parser("stack", help="Run a stack directly")
    stack.add_argument("inputs", nargs="+")
    stack.add_argument("--output", required=True)
    stack.add_argument("--config", type=Path, help="JSON configuration file")
    stack.add_argument("--background-opacity", type=float, help="Background retention from 0 (transparent) to 1 (reference)")
    stack.add_argument("--background-margin-px", type=float, help="Retain surrounding context in reference pixels (0..512)")
    stack.add_argument("--object-masks", type=Path, help="JSON array of named stack/reference/exclude polygon masks")
    stack.add_argument("--keep-frames", action="store_true", help="Retain original/aligned candidate TIFF exports (large)")
    stack.add_argument("--progress", choices=("auto", "plain", "json", "quiet"), default="auto",
                       help="Progress style: auto uses readable terminal output and JSON when redirected; quiet prints only the result")
    stack.add_argument("--cache-max-mib", type=int, default=DEFAULT_CACHE_MAX_BYTES // 1024**2,
                       help="Intermediate cache budget between runs; 0 discards all stage entries (default: 2048)")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run(create_app(args.workspace), host="127.0.0.1", port=args.port)
    else:
        try:
            settings = json.loads(args.config.read_text()) if args.config else {}
            if not isinstance(settings, dict):
                raise ValueError("Recipe must be a JSON object containing configuration settings")
            if args.background_opacity is not None:
                settings["background_opacity"] = args.background_opacity
            if args.background_margin_px is not None:
                settings["background_margin_px"] = args.background_margin_px
            if args.object_masks is not None:
                settings["object_masks"] = json.loads(args.object_masks.read_text())
            options = StackOptions.model_validate(settings)
            if args.cache_max_mib < 0:
                raise ValueError("--cache-max-mib must be zero or greater")
        except ValidationError as exc:
            parser.error("Invalid configuration: " + "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'recipe'}: {error['msg']}" for error in exc.errors()))
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        style = args.progress
        if style == "auto":
            style = "plain" if sys.stdout.isatty() else "json"

        def progress(event):
            if style != "quiet":
                print(format_progress(event) if style == "plain" else json.dumps(event), flush=True)

        try:
            result = run_stack(args.inputs, args.output, Config(**options.model_dump()), progress,
                               keep_frames=args.keep_frames, cache_max_bytes=args.cache_max_mib * 1024**2)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            parser.exit(1, f"Stack failed: {exc}\n")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
