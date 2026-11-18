"""Retain reviewed focal planes without changing automatic quality thresholds."""
from dataclasses import asdict
import json
import sys

import numpy as np
import pytest

from macrostitch import api, native_scan, pipeline


@pytest.mark.parametrize("anchors", [
    [], [{"source": 0, "frame": -1}], [{"source": True, "frame": 1}],
    [{"source": 0, "frame": 1.5}], [{"source": 0, "frame": 1, "other": 2}],
    [{"source": 0, "frame": 1}, {"source": 0, "frame": 1}],
])
def test_required_frames_reject_malformed_requests_in_python_and_api(anchors):
    with pytest.raises(ValueError):
        pipeline.Config(selection_mode="native", required_frames=anchors).validate()
    with pytest.raises(ValueError):
        api.StackOptions(selection_mode="native", required_frames=anchors)


def test_required_frames_need_native_selection_and_sufficient_budget():
    anchors = [{"source": 0, "frame": i} for i in range(3)]
    with pytest.raises(ValueError, match="native selection"):
        pipeline.Config(required_frames=anchors).validate()
    with pytest.raises(ValueError, match="frame budget"):
        pipeline.Config(selection_mode="native", max_frames=2, required_frames=anchors).validate()


def test_required_frames_are_canonical_across_recipe_order_and_api_roundtrip():
    anchors = [{"source": 0, "frame": 160}, {"source": 0, "frame": 0}]
    left = pipeline.Config(selection_mode="native", required_frames=anchors)
    right = pipeline.Config(selection_mode="native", required_frames=list(reversed(anchors)))
    left.validate(); right.validate()
    assert asdict(left) == asdict(right)
    options = api.StackOptions(selection_mode="native", required_frames=anchors)
    restored = pipeline.Config(**options.model_dump()); restored.validate()
    assert asdict(restored) == asdict(left)


@pytest.fixture
def selection_scene(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"; source.write_bytes(b"fixture")
    metadata = {"path": str(source), "sha256": pipeline.digest(source),
                "width": 384, "height": 320, "frames": 4, "fps": 30}
    records = [{"source": 0, "frame": i, "accepted": i != 1} for i in range(4)]
    scores = np.ones((4, 20, 20), np.float32)
    scores[:2] = 0
    scores[2, 2:6, 2:6] = 1000
    scores[3, 12:16, 12:16] = 1000
    native_dir = tmp_path / "native"; native_dir.mkdir()
    (native_dir / "complete.json").write_text("{}")
    monkeypatch.setattr(pipeline, "probe", lambda *a: metadata)
    monkeypatch.setattr(pipeline, "analyze", lambda *a: (records, scores, 0, tmp_path))
    monkeypatch.setattr(native_scan, "native_candidates",
                        lambda *a: (records, scores, 0, np.ones((20, 20)), native_dir))
    return source


def test_required_focal_plane_survives_proxy_rejection_without_wasting_budget(
        tmp_path, monkeypatch, selection_scene):
    captured = []
    class StopAfterSelection(Exception):
        pass
    def capture(sources, selected, *args):
        captured.extend(selected)
        raise StopAfterSelection
    monkeypatch.setattr(pipeline, "extract", capture)
    config = pipeline.Config(selection_mode="native", fusion_mode="spatial", max_frames=4,
                             required_frames=[{"source": 0, "frame": 1}])
    with pytest.raises(StopAfterSelection):
        pipeline.run_stack([selection_scene], tmp_path / "output", config)
    assert {r["frame"] for r in captured} == {0, 1, 2, 3}
    required = next(r for r in captured if r["frame"] == 1)
    assert required["required_frame"] is True and required["accepted"] is False
    assert all(not r["required_frame"] for r in captured if r["frame"] != 1)


@pytest.mark.parametrize("anchor", [{"source": 1, "frame": 0}, {"source": 0, "frame": 4}])
def test_nonexistent_required_plane_fails_before_extraction(
        tmp_path, monkeypatch, selection_scene, anchor):
    monkeypatch.setattr(pipeline, "extract", lambda *a: pytest.fail("Invalid anchor must not decode"))
    with pytest.raises(ValueError, match="Required frame not found"):
        pipeline.run_stack([selection_scene], tmp_path / "output",
                           pipeline.Config(selection_mode="native", required_frames=[anchor]))


def test_cli_forwards_known_sharp_planes(tmp_path, monkeypatch, capsys):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"selection_mode": "native", "required_frames": [
        {"source": 0, "frame": 160}, {"source": 0, "frame": 0}]}))
    captured = []
    def run(inputs, output, config, progress, **kwargs):
        config.validate(); captured.append(config)
        return {"status": "ok"}
    monkeypatch.setattr(api, "run_stack", run)
    monkeypatch.setattr(sys, "argv", ["macrostitch", "stack", "clip.mov", "--output", "out",
                                      "--config", str(recipe), "--progress", "quiet"])
    api.main()
    assert captured[0].required_frames == ({"source": 0, "frame": 0}, {"source": 0, "frame": 160})
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
