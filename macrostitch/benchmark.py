"""Repeatable isolated benchmark: python -m macrostitch.benchmark --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np

from .pipeline import Config, digest, run_stack


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


# Only sessions with this manifest participate in recovery; older or unrelated
# directories are never inferred to be disposable from their names.
_SESSION_LOCKS = {}


def _remove_generated_media(path):
    """Preserve JSON/log evidence, and never traverse a symlink."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            _remove_generated_media(child)
        if not any(path.iterdir()):
            path.rmdir()
    elif path.exists() and path.suffix.lower() not in (".json", ".log"):
        path.unlink()


class ArtifactSession:
    """Track generated paths and recover interrupted runs without touching inputs."""
    marker = "generated-artifacts.json"

    def __init__(self, root, keep_artifacts=False, protected_paths=()):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.recover(root, protected_paths)
        self.path = Path(tempfile.mkdtemp(prefix="benchmark-", dir=root))
        self.lock = (self.path / ".artifact-lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        _SESSION_LOCKS[str(self.path)] = self.lock.fileno()
        self.manifest = {"schema_version": 1, "keep_artifacts": keep_artifacts, "paths": []}
        self.save()

    def save(self):
        # Recovery cannot observe a partial manifest after interruption.
        temporary = self.path / (self.marker + ".tmp")
        write_json(temporary, self.manifest)
        temporary.replace(self.path / self.marker)

    def register(self, path):
        path = Path(path)
        if path.parent != self.path:
            raise ValueError("Generated benchmark artifacts must be direct session children")
        if path.name not in self.manifest["paths"]:
            if path.exists() or path.is_symlink():
                raise ValueError(f"Generated benchmark path already exists: {path}")
            self.manifest["paths"].append(path.name)
            self.save()

    @classmethod
    def clean(cls, session, manifest):
        for name in manifest["paths"]:
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise ValueError("Invalid generated benchmark artifact path")
            _remove_generated_media(session / name)

    @classmethod
    def recover(cls, root, protected_paths=()):
        protected = [Path(path).resolve() for path in protected_paths]
        for session in root.iterdir():
            if session.is_symlink() or not session.is_dir():
                continue
            if any(path == session or session in path.parents for path in protected):
                continue
            marker, lock = session / cls.marker, session / ".artifact-lock"
            if marker.is_symlink() or not marker.is_file() or lock.is_symlink() or not lock.is_file():
                continue
            try:
                with lock.open("r+") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    manifest = json.loads(marker.read_text())
                    if manifest.get("schema_version") == 1 and not manifest.get("keep_artifacts"):
                        cls.clean(session, manifest)
            except (OSError, ValueError, KeyError, TypeError):
                # A malformed/unreadable manifest grants no deletion authority.
                continue

    def close(self):
        try:
            if not self.manifest["keep_artifacts"]:
                self.clean(self.path, self.manifest)
        finally:
            _SESSION_LOCKS.pop(str(self.path), None)
            self.lock.close()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rss_bytes(value, system=None):
    """getrusage reports bytes on macOS, KiB on Linux."""
    return int(value if (system or platform.system()) == "Darwin" else value * 1024)


def pixel_data(path):
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint16:
        raise ValueError(f"Expected decoded 16-bit image: {path}")
    return pixels


def pixel_fingerprint(path):
    pixels = pixel_data(path)
    # Canonical little-endian values make the hash independent of TIFF encoding.
    return {"shape": list(pixels.shape), "dtype": "uint16", "channel_order": "OpenCV BGR/BGRA",
            "pixel_sha256": hashlib.sha256(pixels.astype("<u2", copy=False).tobytes()).hexdigest()}


def pixel_difference(left, right):
    a, b = pixel_data(left), pixel_data(right)
    if a.shape != b.shape:
        return {"same_shape": False, "mae_16bit": None, "max_difference_16bit": None, "pixels_equal": False}
    total = 0
    maximum = 0
    # Row chunks avoid full-size int64 temporary arrays on large images.
    for start in range(0, len(a), 128):
        delta = np.abs(a[start:start+128].astype(np.int32) - b[start:start+128].astype(np.int32))
        total += int(delta.sum(dtype=np.int64))
        maximum = max(maximum, int(delta.max(initial=0)))
    return {"same_shape": True, "mae_16bit": total / a.size,
            "max_difference_16bit": maximum, "pixels_equal": maximum == 0}


def selection_fingerprint(path):
    selection = json.loads(Path(path).read_text())
    fields = ("source", "frame", "candidate_index", "transform", "full_transform")
    value = {"reference_index": selection["reference_index"],
             "selected": [{k: row[k] for k in fields} for row in selection["selected"]]}
    fusion_path = Path(path).parent / "fusion.json"
    if fusion_path.exists():
        fusion = json.loads(fusion_path.read_text())
        if "regions" in fusion:
            value["regional_sources"] = [{k: row[k] for k in
                ("region", "source_index", "source", "frame", "linear_tone_gain")}
                for row in fusion["regions"]]
        if "contributing_sources" in fusion:
            value["spatial_sources"] = [{k: row[k] for k in
                ("source_index", "source", "frame", "dominant_pixels")}
                for row in fusion["contributing_sources"]]
    return {"sha256": json_hash(value), **value}


def usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own, children


def stage_intervals(events, elapsed):
    """Approximate stage intervals from callback boundaries, not profiler spans."""
    totals = {}
    for i, event in enumerate(events):
        end = events[i + 1]["worker_elapsed_seconds"] if i + 1 < len(events) else elapsed
        stage = event.get("stage", "unknown")
        totals[stage] = totals.get(stage, 0.0) + max(0.0, end - event["worker_elapsed_seconds"])
    return {"method": "Approximate time from each progress event to the next; stage events can describe completed work, so these are not exact stage costs.",
            "seconds_by_stage": totals,
            "before_first_event_seconds": events[0]["worker_elapsed_seconds"] if events else elapsed}


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    telemetry = {"mode": request["mode"], "output": request["output"], "cache": request["cache"]}
    before, before_children = usage()
    started = time.perf_counter()
    events = []
    def progress(event):
        observed = {**event, "worker_elapsed_seconds": time.perf_counter() - started}
        events.append(observed)
        print(json.dumps(observed, sort_keys=True), flush=True)
    try:
        result = run_stack(request["inputs"], request["output"], Config(**request["config"]),
                           progress=progress, cache_dir=request["cache"], cache_max_bytes=None)
        elapsed = time.perf_counter() - started
        after, children = usage()
        telemetry.update(status="ok", pipeline_wall_seconds=elapsed,
            cpu_seconds=(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
            child_cpu_seconds=(children.ru_utime + children.ru_stime - before_children.ru_utime - before_children.ru_stime),
            peak_rss_bytes=rss_bytes(after.ru_maxrss), child_peak_rss_bytes=rss_bytes(children.ru_maxrss),
            cache_hit=bool(result["cache_hit"]), pipeline_version=result["version"])
        expected_hit = request["mode"] == "final_cache"
        if telemetry["cache_hit"] != expected_hit:
            raise RuntimeError(f"Expected cache_hit={expected_hit}; received {telemetry['cache_hit']}")
        output = Path(request["output"])
        telemetry["pixels"] = pixel_fingerprint(output / "stack.tiff")
        telemetry["selection"] = selection_fingerprint(output / "selection.json")
        telemetry["artifact_sha256"] = result.get("artifact_sha256", {})
    except Exception:
        telemetry.update(status="error", error=traceback.format_exc())
    telemetry["progress_events"] = events
    telemetry["stage_intervals"] = stage_intervals(events, telemetry.get("pipeline_wall_seconds", time.perf_counter() - started))
    write_json(request["telemetry"], telemetry)
    return 0 if telemetry["status"] == "ok" else 1


def code_fingerprints():
    return {str(p.relative_to(Path(__file__).parent)): digest(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}


def command_version(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    import scipy
    import importlib.metadata
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": sys.version, "executable": sys.executable,
            "opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "ffmpeg": command_version(["ffmpeg", "-version"]),
            "ffprobe": command_version(["ffprobe", "-version"]),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "thread_environment": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "opencv_build": cv2.getBuildInformation()}
    if platform.system() == "Darwin":
        info["hardware"] = {name: command_version(["sysctl", "-n", name]) for name in
                            ("hw.model", "hw.memsize", "hw.physicalcpu", "machdep.cpu.brand_string")}
    return info


def launch(request, directory):
    directory.mkdir()
    request_path = directory / "request.json"
    request["telemetry"] = str(directory / "telemetry.json")
    write_json(request_path, request)
    started = time.perf_counter()
    with (directory / "worker.log").open("w") as log:
        completed = subprocess.Popen([sys.executable, "-m", "macrostitch.benchmark", "--worker", str(request_path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=tuple([_SESSION_LOCKS[str(directory.parent)]])
                                   if str(directory.parent) in _SESSION_LOCKS else ())
        try:
            completed.wait()
        except BaseException:
            # Stop the entire worker group (including image/video subprocesses)
            # before the caller's finally removes their files.
            try:
                os.killpg(completed.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                completed.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The leader can exit before a descendant; kill any remaining group.
            try:
                os.killpg(completed.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            completed.wait()
            raise
    if not Path(request["telemetry"]).exists():
        raise RuntimeError(f"Worker exited {completed.returncode} without telemetry; see {directory / 'worker.log'}")
    result = json.loads(Path(request["telemetry"]).read_text())
    result.update(process_wall_seconds=time.perf_counter()-started, exit_code=completed.returncode,
                  telemetry_path=request["telemetry"])
    if completed.returncode or result["status"] != "ok":
        raise RuntimeError(f"Benchmark worker failed; see {request['telemetry']}")
    return result


def compare_runs(runs):
    recomputed = [r for r in runs if r["mode"] in ("cold_application", "warm_recompute")]
    comparisons = []
    if recomputed:
        baseline = recomputed[0]
        for row in recomputed[1:]:
            comparisons.append({"baseline": baseline["output"], "output": row["output"], "mode": row["mode"],
                **pixel_difference(Path(baseline["output"]) / "stack.tiff", Path(row["output"]) / "stack.tiff"),
                "pixel_hash_equal": baseline["pixels"] == row["pixels"],
                "selection_and_transforms_equal": baseline["selection"] == row["selection"]})
    cold = sum(r["mode"] == "cold_application" for r in runs)
    return {"comparisons": comparisons, "recomputation_count": len(recomputed),
            "cold_repeatability_tested": cold >= 2,
            "cold_repeatability_limitation": None if cold >= 2 else "At least two cold application-cache runs are required to assess cold repeatability.",
            "all_recomputations_identical": all(c["pixels_equal"] and c["pixel_hash_equal"] and c["selection_and_transforms_equal"] for c in comparisons) if comparisons else None}


def timing_summary(runs):
    summary = {}
    for mode in ("cold_application", "warm_recompute", "final_cache"):
        rows = [r for r in runs if r["mode"] == mode]
        metrics = {}
        for name in ("pipeline_wall_seconds", "process_wall_seconds", "cpu_seconds", "child_cpu_seconds", "peak_rss_bytes", "child_peak_rss_bytes"):
            values = [r[name] for r in rows if name in r]
            if values:
                metrics[name] = {"count": len(values), "min": min(values), "max": max(values),
                                 "median": statistics.median(values), "mean": statistics.mean(values),
                                 "sample_stdev": statistics.stdev(values) if len(values) >= 2 else None}
        summary[mode] = {"count": len(rows), "metrics": metrics}
    return summary


def benchmark(inputs, benchmark_root, config, cold_runs=1, warm_runs=3, cache_runs=3, *, keep_artifacts=False):
    if min(cold_runs, warm_runs, cache_runs) < 0 or cold_runs + warm_runs == 0:
        raise ValueError("Nonnegative run counts and at least one recomputation required")
    config.validate()
    inputs = [str(Path(p).resolve(strict=True)) for p in inputs]
    if not inputs:
        raise ValueError("Input videos required")
    root = Path(benchmark_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = {p: digest(p) for p in inputs}
    code = code_fingerprints()
    environment_info = environment()
    artifacts = ArtifactSession(root, keep_artifacts, protected_paths=inputs)
    session = artifacts.path
    report = {"schema_version": 1, "session": str(session), "status": "running", "environment": environment_info,
              "source_sha256": source_hashes, "config": asdict(config), "config_sha256": json_hash(asdict(config)),
              "code_sha256": code, "code_bundle_sha256": json_hash(code), "runs": [],
              "definitions": {"cold_application": "Fresh empty application cache; OS/filesystem caches are uncontrolled, not cold.",
                "warm_recompute": "Existing intermediate cache, fresh output directory; reusable stages may be restored before final outputs are written.",
                "final_cache": "Same output directory; pipeline verifies existing final artifact checksums.",
                "timing": "Pipeline wall/CPU excludes post-run image hashing; process wall includes interpreter startup and evidence extraction.",
                "memory": "Worker lifetime peak RSS and child-process peak RSS reported separately; neither is simultaneous total process-tree memory.",
                "retention": "Generated images/caches are discarded after comparisons and on error unless keep_artifacts is enabled; JSON/log evidence is retained.",
                "keep_artifacts": keep_artifacts}}
    write_json(session / "benchmark.json", report)
    shared_cache = session / "cache-000"
    def run(mode, output, cache):
        artifacts.register(output)
        artifacts.register(cache)
        request = {"inputs": inputs, "config": asdict(config), "output": str(output), "cache": str(cache), "mode": mode}
        row = launch(request, session / f"run-{len(report['runs']):03d}-{mode}")
        report["runs"].append(row)
        write_json(session / "benchmark.json", report)
    try:
        for i in range(cold_runs):
            run("cold_application", session / f"cold-{i:03d}", session / f"cache-{i:03d}")
        if cold_runs == 0:
            run("warmup", session / "warmup", shared_cache)
        for i in range(warm_runs):
            run("warm_recompute", session / f"warm-{i:03d}", shared_cache)
        final_output = Path(report["runs"][-1]["output"])
        final_cache = Path(report["runs"][-1]["cache"])
        for i in range(cache_runs):
            run("final_cache", final_output, final_cache)
        report["repeatability"] = compare_runs(report["runs"])
        report["timing_summary"] = timing_summary(report["runs"])
        report["provenance_unchanged"] = source_hashes == {p: digest(p) for p in inputs} and code == code_fingerprints()
        if not report["provenance_unchanged"]:
            raise RuntimeError("Source files or code changed during benchmark; timings are not a stable-state benchmark")
        report["status"] = "complete"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        raise
    finally:
        try:
            artifacts.close()
        finally:
            write_json(session / "benchmark.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--config", type=Path, help="JSON object containing Config fields")
    parser.add_argument("--cold-runs", type=int, default=1)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--cache-runs", type=int, default=3)
    parser.add_argument("--keep-artifacts", action="store_true", help="Retain generated images and caches for manual inspection")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if not args.inputs or args.benchmark_root is None:
        parser.error("inputs and --benchmark-root are required")
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    result = benchmark(args.inputs, args.benchmark_root, config, args.cold_runs, args.warm_runs, args.cache_runs, keep_artifacts=args.keep_artifacts)
    print(json.dumps({"report": str(Path(result["session"]) / "benchmark.json"), "status": result["status"],
                      "repeatability": result["repeatability"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
