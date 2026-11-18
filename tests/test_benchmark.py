import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from macrostitch import benchmark as b
from macrostitch.pipeline import Config


def image(path, value=0):
    data = np.zeros((8, 9, 3), np.uint16)
    data[0, 0, 0] = value
    assert cv2.imwrite(str(path), data)


def selection(path, root="cache-a", transform=1):
    data = {"reference_index": 0, "selected": [{"source": 0, "frame": 12, "candidate_index": 2,
            "transform": [[transform, 0, 0], [0, 1, 0]], "full_transform": [[1, 0, 0], [0, 1, 0]],
            "proxy": root + "/proxy.png", "aligned": root + "/aligned.tiff"}]}
    path.write_text(json.dumps(data))


def test_memory_units():
    assert b.rss_bytes(123, "Darwin") == 123
    assert b.rss_bytes(123, "Linux") == 123 * 1024


def test_decoded_pixels_and_precise_difference(tmp_path):
    a, c = tmp_path / "a.tiff", tmp_path / "c.png"
    image(a, 64000)
    image(c, 64000)
    assert b.pixel_fingerprint(a) == b.pixel_fingerprint(c)
    image(c, 64007)
    diff = b.pixel_difference(a, c)
    assert diff["max_difference_16bit"] == 7
    assert diff["mae_16bit"] == pytest.approx(7 / (8*9*3))
    assert not diff["pixels_equal"]
    cv2.imwrite(str(c), np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError, match="16-bit"):
        b.pixel_fingerprint(c)


def test_selection_ignores_cache_paths_but_detects_transform(tmp_path):
    a, c = tmp_path / "a.json", tmp_path / "c.json"
    selection(a)
    selection(c, root="fresh-cache")
    assert b.selection_fingerprint(a) == b.selection_fingerprint(c)
    selection(c, transform=1.001)
    assert b.selection_fingerprint(a) != b.selection_fingerprint(c)


def fake_launcher(request, directory):
    directory.mkdir()
    output = Path(request["output"])
    if request["mode"] != "final_cache":
        assert not output.exists(), "Recomputation must use a fresh output"
        output.mkdir()
        image(output / "stack.tiff")
        selection(output / "selection.json", root=request["cache"])
    else:
        assert (output / "stack.tiff").exists()
    return {**request, "status": "ok", "cache_hit": request["mode"] == "final_cache",
            "pixels": b.pixel_fingerprint(output / "stack.tiff"),
            "selection": b.selection_fingerprint(output / "selection.json")}


@pytest.mark.parametrize("cold,warm,expected", [(2, 2, True), (1, 3, False), (0, 2, False)])
def test_isolated_modes_and_repeatability(tmp_path, monkeypatch, cold, warm, expected):
    monkeypatch.setattr(b, "launch", fake_launcher)
    monkeypatch.setattr(b, "environment", lambda: {})
    source = tmp_path / "original.mov"
    source.write_bytes(b"original")
    root = tmp_path / "bench"
    root.mkdir()
    prior = root / "existing-result"
    prior.write_bytes(b"keep")
    result = b.benchmark([source], root, Config(), cold, warm, 3)
    runs = result["runs"]
    assert result["status"] == "complete"
    assert result["repeatability"]["cold_repeatability_tested"] is expected
    assert result["repeatability"]["recomputation_count"] == cold + warm
    assert result["repeatability"]["all_recomputations_identical"] is True
    assert len(result["repeatability"]["comparisons"]) == cold + warm - 1
    warm_rows = [r for r in runs if r["mode"] == "warm_recompute"]
    assert len({r["cache"] for r in warm_rows}) == 1
    cold_rows = [r for r in runs if r["mode"] == "cold_application"]
    assert len({r["cache"] for r in cold_rows}) == cold
    assert all(r["output"] == runs[-4]["output"] for r in runs[-3:])
    assert prior.read_bytes() == b"keep" and source.read_bytes() == b"original"
    assert all(not (Path(r["output"]) / "stack.tiff").exists() for r in runs)
    assert all((Path(r["output"]) / "selection.json").exists() for r in runs)
    assert (Path(result["session"]) / "benchmark.json").exists()


def test_cache_reads_do_not_prove_repeatability(tmp_path):
    row = fake_launcher({"output": str(tmp_path / "out"), "cache": "c", "mode": "cold_application"}, tmp_path / "run")
    report = b.compare_runs([row, {**row, "mode": "final_cache"}, {**row, "mode": "final_cache"}])
    assert report["all_recomputations_identical"] is None
    assert not report["cold_repeatability_tested"]


def test_worker_checks_cache_mode_and_saves_error(tmp_path, monkeypatch):
    request = {"mode": "warm_recompute", "output": str(tmp_path / "out"), "cache": str(tmp_path / "cache"),
               "inputs": [], "config": {}, "telemetry": str(tmp_path / "telemetry.json")}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(b, "run_stack", lambda *a, **kw: {"cache_hit": True, "version": "test"})
    assert b.worker(path) == 1
    result = json.loads(Path(request["telemetry"]).read_text())
    assert result["status"] == "error"
    assert "Expected cache_hit=False" in result["error"]
    assert result["peak_rss_bytes"] > 0


def test_worker_success_telemetry(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    image(output / "stack.tiff")
    selection(output / "selection.json")
    request = {"mode": "cold_application", "output": str(output), "cache": str(tmp_path / "cache"),
               "inputs": [], "config": {}, "telemetry": str(tmp_path / "telemetry.json")}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(b, "run_stack", lambda *a, **kw: {"cache_hit": False, "version": "test"})
    assert b.worker(path) == 0
    result = json.loads(Path(request["telemetry"]).read_text())
    assert result["pipeline_wall_seconds"] >= 0 and result["cpu_seconds"] >= 0
    assert result["child_cpu_seconds"] >= 0
    assert result["pixels"]["dtype"] == "uint16"


def test_worker_logs_progress_and_approximate_intervals(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out"
    output.mkdir()
    image(output / "stack.tiff")
    selection(output / "selection.json")
    request = {"mode": "cold_application", "output": str(output), "cache": str(tmp_path / "cache"),
               "inputs": [], "config": {}, "telemetry": str(tmp_path / "telemetry.json")}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    def run_stack(inputs, output, config, *, progress, cache_dir, cache_max_bytes):
        assert cache_dir == request["cache"]
        assert cache_max_bytes is None
        progress({"stage": "probe", "message": "Checking inputs"})
        progress({"stage": "fuse", "completed": 1})
        return {"cache_hit": False, "version": "test"}
    monkeypatch.setattr(b, "run_stack", run_stack)
    assert b.worker(path) == 0
    logged = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["stage"] for e in logged] == ["probe", "fuse"]
    result = json.loads(Path(request["telemetry"]).read_text())
    assert result["progress_events"] == logged
    intervals = result["stage_intervals"]
    assert "Approximate" in intervals["method"]
    assert all(v >= 0 for v in intervals["seconds_by_stage"].values())
    assert sum(intervals["seconds_by_stage"].values()) + intervals["before_first_event_seconds"] == pytest.approx(result["pipeline_wall_seconds"])


def test_rgba_alpha_is_part_of_pixel_evidence(tmp_path):
    a, c = tmp_path / "a.tiff", tmp_path / "c.tiff"
    data = np.zeros((4, 5, 4), np.uint16)
    data[..., :3] = 12000
    data[..., 3] = 65535
    assert cv2.imwrite(str(a), data)
    data[0, 0, 3] -= 9
    assert cv2.imwrite(str(c), data)
    assert b.pixel_fingerprint(a)["shape"] == [4, 5, 4]
    assert b.pixel_fingerprint(a) != b.pixel_fingerprint(c)
    assert not b.pixel_difference(a, c)["pixels_equal"]


def test_regional_winners_are_part_of_selection_evidence(tmp_path):
    path = tmp_path / "selection.json"
    selection(path)
    fusion = {"regions": [{"region": 0, "source_index": 0, "source": 0, "frame": 12, "linear_tone_gain": 1.0}]}
    b.write_json(tmp_path / "fusion.json", fusion)
    before = b.selection_fingerprint(path)
    fusion["regions"][0]["source_index"] = 1
    b.write_json(tmp_path / "fusion.json", fusion)
    assert b.selection_fingerprint(path) != before


def test_spatial_contributors_ignore_paths_and_track_dominant_pixels(tmp_path):
    path=tmp_path/'selection.json';selection(path)
    fusion={'contributing_sources':[{'source_index':0,'source':0,'frame':12,'dominant_pixels':20,'path':'old'}]}
    b.write_json(tmp_path/'fusion.json',fusion)
    before=b.selection_fingerprint(path)
    fusion['contributing_sources'][0]['path']='new'
    b.write_json(tmp_path/'fusion.json',fusion)
    assert b.selection_fingerprint(path)==before
    fusion['contributing_sources'][0]['dominant_pixels']=21
    b.write_json(tmp_path/'fusion.json',fusion)
    assert b.selection_fingerprint(path)!=before


def test_failed_benchmark_discards_partial_media_preserves_evidence_and_external_symlink(tmp_path, monkeypatch):
    source = tmp_path / 'original.mov'; source.write_bytes(b'input')
    external = tmp_path / 'external'; external.mkdir()
    (external / 'original.tiff').write_bytes(b'untouched')
    def fail(request, directory):
        directory.mkdir()
        (directory / 'worker.log').write_text('failed worker')
        output = Path(request['output']); output.mkdir()
        (output / 'partial.tiff').write_bytes(b'partial')
        (output / 'report.json').write_text('{}')
        (output / 'external').symlink_to(external, target_is_directory=True)
        cache = Path(request['cache']); cache.mkdir()
        (cache / 'intermediate.npy').write_bytes(b'cache')
        raise RuntimeError('test failure')
    monkeypatch.setattr(b, 'launch', fail)
    monkeypatch.setattr(b, 'environment', lambda: {})
    with pytest.raises(RuntimeError, match='test failure'):
        b.benchmark([source], tmp_path / 'bench', Config(), 1, 0, 0)
    session, = (tmp_path / 'bench').iterdir()
    assert not (session / 'cold-000' / 'partial.tiff').exists()
    assert not (session / 'cold-000' / 'external').exists()
    assert not (session / 'cache-000').exists()
    assert (session / 'cold-000' / 'report.json').exists()
    assert (session / 'run-000-cold_application' / 'worker.log').read_text() == 'failed worker'
    assert json.loads((session / 'benchmark.json').read_text())['status'] == 'error'
    assert (external / 'original.tiff').read_bytes() == b'untouched'


def test_keep_artifacts_is_explicit_opt_in(tmp_path, monkeypatch):
    source = tmp_path / 'source.mov'; source.write_bytes(b'input')
    monkeypatch.setattr(b, 'launch', fake_launcher)
    monkeypatch.setattr(b, 'environment', lambda: {})
    result = b.benchmark([source], tmp_path / 'bench', Config(), 1, 0, 0, keep_artifacts=True)
    assert (Path(result['runs'][0]['output']) / 'stack.tiff').exists()
    b.ArtifactSession.recover(tmp_path / 'bench')
    assert (Path(result['runs'][0]['output']) / 'stack.tiff').exists()


def test_recovery_cleans_abandoned_owned_media_and_skips_active_sessions(tmp_path):
    active = b.ArtifactSession(tmp_path)
    output = active.path / 'cold-000'; active.register(output); output.mkdir()
    (output / 'stack.tiff').write_bytes(b'active')
    b.ArtifactSession.recover(tmp_path)
    assert (output / 'stack.tiff').exists()
    # Simulate process death after the OS releases its lock, before finally.
    b._SESSION_LOCKS.pop(str(active.path))
    active.lock.close()
    unrelated = tmp_path / 'benchmark-unmanaged'; unrelated.mkdir()
    (unrelated / 'original.tiff').write_bytes(b'keep')
    b.ArtifactSession.recover(tmp_path)
    assert not output.exists()
    assert (unrelated / 'original.tiff').read_bytes() == b'keep'


def test_recovery_preserves_managed_session_reused_as_external_input(tmp_path):
    previous = b.ArtifactSession(tmp_path)
    output = previous.path / 'warm-000'; previous.register(output); output.mkdir()
    source = output / 'source.mov'; source.write_bytes(b'original')
    b._SESSION_LOCKS.pop(str(previous.path)); previous.lock.close()
    new = b.ArtifactSession(tmp_path, protected_paths=[source])
    try:
        assert source.read_bytes() == b'original'
    finally:
        new.close()


def test_launch_cancellation_stops_worker_group_before_session_cleanup(tmp_path, monkeypatch):
    session = b.ArtifactSession(tmp_path)
    calls = []
    class Process:
        pid = 12345
        def __init__(self, *args, **kwargs):
            assert kwargs['start_new_session'] is True
            assert kwargs['pass_fds'] == (session.lock.fileno(),)
            self.wait_count = 0
        def wait(self, timeout=None):
            self.wait_count += 1
            calls.append(('wait', timeout))
            if self.wait_count == 1:
                raise KeyboardInterrupt()
            return 0
    monkeypatch.setattr(b.subprocess, 'Popen', Process)
    monkeypatch.setattr(b.os, 'killpg', lambda pid, sig: calls.append(('kill', pid, sig)))
    try:
        with pytest.raises(KeyboardInterrupt):
            b.launch({'mode': 'cold_application'}, session.path / 'run-000')
        assert calls == [('wait', None), ('kill', 12345, b.signal.SIGTERM),
                         ('wait', 5), ('kill', 12345, b.signal.SIGKILL), ('wait', None)]
    finally:
        session.close()
