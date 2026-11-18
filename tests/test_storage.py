"""Retention must reclaim space without touching sources, evidence or active jobs."""
import fcntl
import json
import os
from pathlib import Path

import pytest

from macrostitch import pipeline
from macrostitch import storage as s


def put(root, name, data=b"generated"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def project(root):
    put(root, "macrostitch/pipeline.py")
    put(root, "pyproject.toml")
    return root


def test_cache_evicts_whole_entries_preferring_large_scratch(tmp_path):
    cache = tmp_path / "cache"
    first = put(cache, "frames/first/frame.tiff", b"f" * 20)
    sidecar = put(cache, "frames/first/frame.tiff.verified.json")
    aligned = put(cache, "aligned/second/aligned.tiff", b"a" * 20)
    proxy = put(cache, "proxies/proxy/p.png", b"p" * 20)
    analysis = put(cache, "analysis/scores/scores.npy", b"s" * 20)
    unknown = put(cache, "outside-private-stages.txt", b"keep")
    os.utime(first, ns=(1, 1)); os.utime(sidecar, ns=(1, 1))
    result = s.prune_cache(cache, 40)
    assert result["after_bytes"] == 40
    assert not first.exists() and not sidecar.exists() and not aligned.exists()
    assert proxy.exists() and analysis.exists() and unknown.read_bytes() == b"keep"
    assert s.prune_cache(cache, 0)["after_bytes"] == 0
    assert unknown.exists()


@pytest.mark.parametrize("limit", [-1, 1.2, True, "4"])
def test_invalid_budget_cannot_delete_files(tmp_path, limit):
    frame = put(tmp_path, "frames/key/frame.tiff")
    with pytest.raises(ValueError):
        s.prune_cache(tmp_path, limit)
    assert frame.exists()


def test_cache_budget_on_failure_and_no_limit(tmp_path):
    with pytest.raises(RuntimeError, match="test failure"):
        with s.cache_retention(tmp_path, 0):
            frame = put(tmp_path, "frames/key/frame.tiff")
            raise RuntimeError("test failure")
    assert not frame.exists()
    with s.cache_retention(tmp_path, None):
        frame = put(tmp_path, "frames/key/frame.tiff")
    assert frame.exists()


def test_cache_never_follows_symlinks(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    external = put(tmp_path, "external/original.tiff", b"original")
    (cache / "frames").symlink_to(external.parent, target_is_directory=True)
    put(cache, "aligned/key/entry.tiff")
    (cache / "aligned/key/alias").symlink_to(external.parent, target_is_directory=True)
    s.prune_cache(cache, 0)
    assert external.read_bytes() == b"original"
    alias = tmp_path / "cache-alias"; alias.symlink_to(cache, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        s.prune_cache(alias, 0)


def test_export_cleanup_preserves_unrelated_files_and_hardlink_source(tmp_path):
    source = put(tmp_path, "source.tiff", b"original")
    frame = tmp_path / "out/frames/000_clip0_frame000012.tiff"
    frame.parent.mkdir(parents=True); os.link(source, frame)
    unrelated = put(tmp_path, "out/frames/manual.tiff", b"manual")
    linked_dir = tmp_path / "out/aligned"; linked_dir.symlink_to(source.parent, target_is_directory=True)
    s.clear_frame_exports(tmp_path / "out")
    assert not frame.exists()
    assert source.read_bytes() == b"original" and unrelated.read_bytes() == b"manual"
    assert linked_dir.is_symlink()


def test_workspace_preview_and_cleanup_preserve_delivery_and_evidence(tmp_path):
    root = project(tmp_path)
    kept = [put(root, name, b"keep") for name in (
        "focus-rack.mov", "recipe.json", "outputs/final-full/stack.tiff",
        "outputs/final-full/report.json", "outputs/final-full/frames/manual.tiff",
        "diagnostics/audit/comparison.json", "diagnostics/audit/run.log",
        "diagnostics/audit/review.png", "diagnostics/audit/repeat.py",
        "diagnostics/audit/trial/report.json",
        "diagnostics/audit/fusion-trial/fusion.json",
        "benchmarks/benchmark-old/benchmark.json", "benchmarks/benchmark-old/run/worker.log")]
    removed = [put(root, name) for name in (
        "outputs/final-full/frames/000_clip0_frame000001.tiff",
        "outputs/final-full/aligned/000_clip0_frame000001.tiff",
        "diagnostics/audit/fresh-cache/proxies/key/p000001.png",
        "diagnostics/audit/fresh-cache/frames/key/f000001.tiff",
        "diagnostics/audit/scan/scores.npy", "diagnostics/audit/trial/stack.tiff", "diagnostics/audit/trial/stack.png",
        "diagnostics/audit/fusion-trial/stack.png",
        "benchmarks/benchmark-old/cache-000/proxies/p.png",
        "benchmarks/benchmark-old/warm-000/stack.png")]
    cache = put(root, "outputs/.macrostitch/cache/frames/key/f.tiff")
    dry_run = s.cleanup_workspace(root)
    assert not dry_run["applied"] and all(p.exists() for p in removed)
    report = s.cleanup_workspace(root, apply=True, cache_max_bytes=0)
    assert all(p.read_bytes() == b"keep" for p in kept)
    assert not any(p.exists() for p in removed) and not cache.exists()
    assert len(report["removed_files"]) == len(removed)
    assert report["after"]["logical_bytes"] < report["before"]["logical_bytes"]
    assert json.loads((root / "diagnostics/storage-cleanup.json").read_text())["applied"]
    assert s.cleanup_workspace(root, apply=True, cache_max_bytes=0)["removed_files"] == []


@pytest.mark.parametrize("alias", ["outputs", "outputs/.macrostitch", "diagnostics", "benchmarks"])
def test_workspace_rejects_symlinked_cleanup_ancestors(tmp_path, alias):
    root = project(tmp_path / "project")
    external = put(tmp_path, "external/.macrostitch/cache/frames/key/original.tiff", b"original")
    link = root / alias; link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(tmp_path / "external", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        s.cleanup_workspace(root, apply=True, cache_max_bytes=0)
    assert external.read_bytes() == b"original"


@pytest.mark.parametrize("lock_name", ["outputs/.macrostitch/cache/pipeline.lock", "outputs/final/.pipeline.lock",
                                       "benchmarks/benchmark-active/.artifact-lock"])
def test_workspace_refuses_active_writers_without_deletion(tmp_path, lock_name):
    root = project(tmp_path)
    artifact = put(root, "diagnostics/audit/stack.tiff")
    lock = put(root, lock_name, b"")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Active work"):
            s.cleanup_workspace(root, apply=True, cache_max_bytes=0)
    assert artifact.exists()


def test_maintenance_establishes_default_lock_before_a_new_writer(tmp_path):
    root = project(tmp_path)
    with s.maintenance_locks(root):
        with (root / "outputs/.macrostitch/cache/pipeline.lock").open("a") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_workspace_guard_protects_custom_cache_and_diagnostic_outputs(tmp_path):
    root = project(tmp_path)
    artifact = put(root, 'diagnostics/new-output/stack.tiff')
    # Shared project guard precedes all per-cache/per-output locks, including
    # writers whose output lock did not yet exist during maintenance discovery.
    with s.pipeline_workspace_guard(artifact.parent):
        with pytest.raises(RuntimeError, match='Active work'):
            s.cleanup_workspace(root, apply=True)
    assert artifact.exists()
    with s.maintenance_locks(root):
        with (root/'.macrostitch/storage.lock').open('a') as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_workspace_guard_also_protects_cache_when_output_is_elsewhere(tmp_path):
    root = project(tmp_path/'project')
    with s.pipeline_workspace_guard(tmp_path/'outside', root/'diagnostics/custom-cache'):
        with pytest.raises(RuntimeError, match='Active work'):
            s.cleanup_workspace(root, apply=True)


def test_pipeline_failure_enforces_cache_budget(tmp_path, monkeypatch):
    source = put(tmp_path, "source.mov")
    cache = tmp_path / "cache"
    monkeypatch.setattr(pipeline, "probe", lambda path: {"sha256": "fixture", "width": 64, "height": 64})
    def fail(*args):
        put(cache, "frames/partial/frame.tiff")
        raise RuntimeError("decode failed")
    monkeypatch.setattr(pipeline, "analyze", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        pipeline.run_stack([source], tmp_path / "output", cache_dir=cache, cache_max_bytes=0)
    assert not (cache / "frames/partial").exists()
    assert (cache / "pipeline.lock").exists()


def test_pipeline_rejects_symlinked_frame_export_directories(tmp_path):
    source = put(tmp_path, "source.mov")
    external = put(tmp_path, "external/000_clip0_frame000000.tiff", b"original")
    output = tmp_path / "output"; output.mkdir()
    (output / "frames").symlink_to(external.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        pipeline.run_stack([source], output, keep_frames=True)
    assert external.read_bytes() == b"original"
