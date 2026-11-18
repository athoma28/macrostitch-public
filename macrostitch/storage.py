"""Bounded scratch storage and explicit workspace cleanup.

Run ``python -m macrostitch.storage --workspace .`` to preview legacy cleanup;
add ``--apply`` to remove reproducible intermediates. Sources, delivered output
images, recipes, code, JSON evidence and logs are preserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time

DEFAULT_CACHE_MAX_BYTES = 2 * 1024**3
CACHE_STAGES = {"proxies", "analysis", "native-analysis", "frames", "aligned", "fusion"}
EXPORT_NAME = re.compile(r"\d{3}_clip\d+_frame\d{6,}\.tiff\Z")
MEDIA_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".mkv"}


def files_below(root):
    """Walk real directories only; never descend through a symlink."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        return
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def usage(root):
    seen = set()
    logical = allocated = count = 0
    for path in files_below(root):
        stat = path.stat()
        count += 1
        logical += stat.st_size
        identity = stat.st_dev, stat.st_ino
        if identity not in seen:
            allocated += stat.st_blocks * 512
            seen.add(identity)
    return {"files": count, "logical_bytes": logical, "allocated_bytes": allocated}


def validate_limit(max_bytes):
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("cache_max_bytes must be a nonnegative integer or None")


def prune_cache(cache, max_bytes=DEFAULT_CACHE_MAX_BYTES):
    """Evict whole stage entries, oldest first. Caller holds pipeline.lock.

    Large frame/registration entries go first, preserving small analysis records
    longer. Stage namespaces are private pipeline scratch; files outside them
    and symlinks are never deleted. Accounting deliberately
    counts hardlinks at each path, so the budget is conservative.
    """
    validate_limit(max_bytes)
    cache = Path(cache)
    if cache.resolve() != cache.absolute():
        raise ValueError("Cache cleanup must not traverse a symlink")
    if max_bytes is None:
        return {"limit_bytes": None, "removed_entries": []}
    entries = []
    total = 0
    for stage_name in sorted(CACHE_STAGES):
        stage = cache / stage_name
        if stage.is_symlink() or not stage.is_dir():
            continue
        for entry in stage.iterdir():
            if entry.is_symlink():
                continue
            paths = list(files_below(entry)) if entry.is_dir() else [entry]
            stats = [p.stat() for p in paths if p.is_file()]
            size = sum(s.st_size for s in stats)
            total += size
            newest = max((s.st_mtime_ns for s in stats), default=entry.stat().st_mtime_ns)
            priority = 0 if stage_name in {"frames", "aligned"} else 1 if stage_name == "proxies" else 2
            entries.append((priority, newest, str(entry), size))
    before = total
    removed = []
    for _, _, name, size in sorted(entries):
        if total <= max_bytes:
            break
        entry = Path(name)
        if entry.is_symlink():
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        total -= size
        removed.append(str(entry.relative_to(cache)))
    return {"limit_bytes": max_bytes, "before_bytes": before, "after_bytes": total,
            "removed_entries": removed}


@contextmanager
def cache_retention(cache, max_bytes):
    """Enforce the cache budget on entry and on success, failure or cancellation.

    The working set may exceed the budget while a pipeline operation holds its
    lock; eviction only occurs when no stage is reading it.
    """
    prune_cache(cache, max_bytes)
    try:
        yield
    finally:
        prune_cache(cache, max_bytes)


def clear_frame_exports(output):
    """Remove only pipeline-named candidate links, leaving unrelated files alone."""
    for name in ("frames", "aligned"):
        directory = Path(output) / name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.iterdir():
            if EXPORT_NAME.fullmatch(path.name) and not path.is_symlink() and path.is_file():
                path.unlink()
        if not any(directory.iterdir()):
            directory.rmdir()


@contextmanager
def workspace_guard(workspace, *, maintenance=False):
    """Coordinate maintenance with writers using any output/cache in a project."""
    workspace = Path(workspace)
    directory = workspace / ".macrostitch"
    path = directory / "storage.lock"
    if directory.is_symlink() or path.is_symlink():
        raise ValueError(f"Storage coordination must not traverse a symlink: {path}")
    directory.mkdir(exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX | fcntl.LOCK_NB) if maintenance else fcntl.LOCK_SH)
        except BlockingIOError as exc:
            raise RuntimeError(f"Active work holds {path}; retry cleanup after it finishes") from exc
        yield


@contextmanager
def pipeline_workspace_guard(*locations):
    workspaces = set()
    for location in locations:
        for parent in Path(location).parents:
            if (parent / "macrostitch/pipeline.py").is_file() and (parent / "pyproject.toml").is_file():
                workspaces.add(parent)
                break
    with ExitStack() as stack:
        for workspace in sorted(workspaces):
            stack.enter_context(workspace_guard(workspace))
        # Standalone output/cache trees have no project maintenance command.
        yield


@contextmanager
def maintenance_locks(workspace):
    """Refuse maintenance when a pipeline/benchmark has an active writer.

    Keep lock files in place: unlinking a lock would permit two independent locks
    on different inodes at the same path.
    """
    with ExitStack() as stack:
        stack.enter_context(workspace_guard(workspace, maintenance=True))
        cache = workspace / "outputs" / ".macrostitch" / "cache"
        # Establish the same lock as future default-cache writers, even when
        # this is the first operation in a workspace. Existing output locks
        # below also protect writers using custom caches.
        cache.mkdir(parents=True, exist_ok=True)
        canonical = cache / "pipeline.lock"
        handle = stack.enter_context(canonical.open("a"))
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Active work holds {canonical}; retry cleanup after it finishes") from exc
        paths = [p for base in (workspace / "outputs", workspace / "diagnostics", workspace / "benchmarks")
                 for p in files_below(base)
                 if p != canonical and p.name in {"pipeline.lock", ".pipeline.lock", ".artifact-lock"}]
        paths.sort(key=lambda p: (p.name != "pipeline.lock", str(p)))
        for path in paths:
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Active work holds {path}; retry cleanup after it finishes") from exc
        yield


def cleanup_candidates(workspace):
    """Conservative legacy policy, scoped to generated project directories."""
    outputs = workspace / "outputs"
    for path in files_below(outputs):
        relative = path.relative_to(outputs)
        if (len(relative.parts) == 3 and relative.parts[1] in {"frames", "aligned"}
                and EXPORT_NAME.fullmatch(path.name)):
            yield path
    for base in (workspace / "diagnostics", workspace / "benchmarks"):
        for path in files_below(base):
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            parts = path.relative_to(base).parts
            # New benchmark sessions own their cleanup and may be starting
            # after maintenance's lock snapshot. Leave these to their manifest
            # recovery; this command handles historical unmarked sessions.
            if any((parent / "generated-artifacts.json").is_file() for parent in path.parents
                   if parent != workspace and parent.is_relative_to(base)):
                continue
            benchmark = base.name == "benchmarks" or any(p.startswith("benchmark-") for p in parts[:-1])
            scratch = any(p in {"fresh-cache", "test-run", "native-analysis", "proxies"}
                          or re.fullmatch(r"cache-\d{3}", p) for p in parts[:-1])
            duplicate_stack = path.name == "stack.png" and any((path.parent / name).is_file()
                                                               for name in ("report.json", "fusion.json"))
            if benchmark or scratch or duplicate_stack or path.suffix.lower() in {".tif", ".tiff", ".npy", ".npz"}:
                yield path


def cleanup_workspace(workspace, *, apply=False, cache_max_bytes=DEFAULT_CACHE_MAX_BYTES):
    """Preview/apply legacy cleanup and retain an exact removal inventory."""
    validate_limit(cache_max_bytes)
    workspace = Path(workspace).resolve(strict=True)
    if not (workspace / "macrostitch" / "pipeline.py").is_file() or not (workspace / "pyproject.toml").is_file():
        raise ValueError("Choose a Macrostitch project workspace")
    for relative in (".macrostitch", "outputs", "outputs/.macrostitch", "outputs/.macrostitch/cache", "diagnostics", "benchmarks"):
        path = workspace / relative
        if path.resolve() != path:
            raise ValueError(f"Cleanup must not traverse a symlink: {path}")
    report_path = workspace / "diagnostics" / "storage-cleanup.json"
    if report_path.is_symlink():
        raise ValueError(f"Cleanup report must not be a symlink: {report_path}")
    before = usage(workspace)
    with maintenance_locks(workspace):
        candidates = list(cleanup_candidates(workspace))
        inventory = [{"path": str(p.relative_to(workspace)), "bytes": p.stat().st_size} for p in candidates]
        cache = workspace / "outputs" / ".macrostitch" / "cache"
        cache_result = None
        if apply:
            for path in candidates:
                path.unlink()
            if not cache.is_symlink() and cache.is_dir():
                cache_result = prune_cache(cache, cache_max_bytes)
        report = {"schema_version": 1, "applied": apply, "created_at": time.time(),
                  "before": before, "after": usage(workspace) if apply else None,
                  "removed_files" if apply else "candidate_files": inventory,
                  "cache": cache_result, "cache_limit_bytes": cache_max_bytes,
                  "preserved": "Source videos, delivered output images, recipes, code, JSON evidence, logs and small diagnostic review images."}
        if apply:
            destination = workspace / "diagnostics" / "storage-cleanup.json"
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Apply cleanup; default is a preview")
    parser.add_argument("--cache-max-mib", type=int, default=DEFAULT_CACHE_MAX_BYTES // 1024**2)
    args = parser.parse_args(argv)
    report = cleanup_workspace(args.workspace, apply=args.apply, cache_max_bytes=args.cache_max_mib * 1024**2)
    inventory = report.get("removed_files", report.get("candidate_files", []))
    cache = report["cache"]
    cache_summary = {key: value for key, value in cache.items() if key != "removed_entries"} if cache else None
    if cache_summary is not None:
        cache_summary["removed_entries"] = len(cache["removed_entries"])
    print(json.dumps({"applied": report["applied"], "before": report["before"], "after": report["after"],
                      "files": len(inventory), "candidate_path_bytes": sum(p["bytes"] for p in inventory),
                      "cache": cache_summary, "report": str(args.workspace / "diagnostics/storage-cleanup.json") if args.apply else None}, indent=2))


if __name__ == "__main__":
    main()
