"""Workspace discovery and recoverable job history for the local editor."""
import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from macrostitch import api
from test_api import payload, wait_done


def test_workspace_browse_is_shallow_sorted_and_scoped(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "nested.MOV").write_bytes(b"video")
    (tmp_path / "z.MP4").write_bytes(b"video")
    (tmp_path / "a.mov").write_bytes(b"video")
    (tmp_path / "recipe.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / ".hidden.mov").write_bytes(b"video")
    (tmp_path / "alias.mov").symlink_to(tmp_path / "a.mov")
    with TestClient(api.create_app(tmp_path)) as client:
        listing = client.get("/workspace/files").json()
        assert listing["workspace_root"] == str(tmp_path)
        assert listing["directory"] == "" and listing["parent"] is None
        assert listing["directories"] == [{"name": "videos", "path": "videos"}]
        assert listing["videos"] == [{"name": "a.mov", "path": "a.mov"}, {"name": "z.MP4", "path": "z.MP4"}]
        assert listing["recipes"] == [{"name": "recipe.json", "path": "recipe.json"}]
        assert listing["truncated"] is False
        nested = client.get("/workspace/files", params={"directory": "videos"}).json()
        assert nested["parent"] == ""
        assert nested["videos"][0]["path"] == "videos/nested.MOV"
        limited = client.get("/workspace/files?limit=1").json()
        assert limited["truncated"] is True and len(limited["directories"]) == 1
        for directory in ("..", "missing", "a.mov", ".macrostitch"):
            assert client.get("/workspace/files", params={"directory": directory}).status_code == 400
        for limit in (0, 1001):
            assert client.get("/workspace/files", params={"limit": limit}).status_code == 422


def test_workspace_recipe_preserves_supplied_settings(tmp_path):
    settings = {"threads": 6, "background_opacity": .2, "canvas_mode": "reference"}
    (tmp_path / "recipe.json").write_text(json.dumps(settings))
    with TestClient(api.create_app(tmp_path)) as client:
        response = client.get("/workspace/recipe", params={"path": "recipe.json"})
        assert response.status_code == 200
        assert response.json() == {"name": "recipe.json", "path": "recipe.json", "config": settings}


@pytest.mark.parametrize("content", ["{", "[]", '{"artifacts":{}}', '{"threads":0}', '{"threads":NaN}', ' ' * (1024 * 1024 + 1), '[' * 20000 + ']' * 20000])
def test_workspace_rejects_invalid_recipe(tmp_path, content):
    (tmp_path / "recipe.json").write_text(content)
    with TestClient(api.create_app(tmp_path)) as client:
        response = client.get("/workspace/recipe", params={"path": "recipe.json"})
        assert response.status_code == 400
        assert "Invalid recipe" in response.json()["detail"]


def test_workspace_discovery_cannot_escape_through_symlinks(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside-workspace")
    (outside / "recipe.json").write_text("{}")
    (tmp_path / "outside").symlink_to(outside, target_is_directory=True)
    (tmp_path / ".private.json").write_text("{}")
    with TestClient(api.create_app(tmp_path)) as client:
        assert client.get("/workspace/files").json()["directories"] == []
        assert client.get("/workspace/files?directory=outside").status_code == 400
        assert client.get("/workspace/recipe?path=outside/recipe.json").status_code == 400
        assert client.get("/workspace/recipe?path=.private.json").status_code == 400


def test_recent_jobs_are_compact_sorted_filtered_and_persisted(tmp_path, monkeypatch):
    (tmp_path / "clip.mov").write_bytes(b"video")
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    with TestClient(api.create_app(tmp_path)) as client:
        assert client.get("/jobs").json() == {"jobs": [], "total": 0}
        first = client.post("/jobs", json=payload("first")).json()
        wait_done(client, first["id"])
        second = client.post("/jobs", json=payload("second")).json()
        wait_done(client, second["id"])
        listing = client.get("/jobs?limit=1&status=completed").json()
        assert listing["total"] == 2
        assert len(listing["jobs"]) == 1
        assert listing["jobs"][0]["id"] == second["id"]
        assert "config" not in listing["jobs"][0] and "result" not in listing["jobs"][0]
        assert client.get("/jobs?status=failed").json() == {"jobs": [], "total": 0}
        assert client.get("/jobs?limit=0").status_code == 422
        assert client.get("/jobs?status=bogus").status_code == 422
    with TestClient(api.create_app(tmp_path)) as client:
        assert [job["id"] for job in client.get("/jobs").json()["jobs"]] == [second["id"], first["id"]]


def test_recent_jobs_includes_queued_and_running_without_waiting(tmp_path, monkeypatch):
    (tmp_path / "clip.mov").write_bytes(b"video")
    started, release = threading.Event(), threading.Event()

    def worker(*args):
        started.set()
        assert release.wait(5)
        return {}

    monkeypatch.setattr(api, "run_stack", worker)
    with TestClient(api.create_app(tmp_path)) as client:
        try:
            first = client.post("/jobs", json=payload("first")).json()
            assert started.wait(2)
            second = client.post("/jobs", json=payload("second")).json()
            states = {job["id"]: job["status"] for job in client.get("/jobs").json()["jobs"]}
            assert states == {first["id"]: "running", second["id"]: "queued"}
        finally:
            release.set()


@pytest.mark.parametrize("field", ["created_at", "started_at", "finished_at"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10 ** 1000, -1, "yesterday", True])
def test_invalid_persisted_dates_do_not_break_job_history(tmp_path, monkeypatch, field, value):
    (tmp_path / "clip.mov").write_bytes(b"video")
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    with TestClient(api.create_app(tmp_path)) as client:
        job = client.post("/jobs", json=payload()).json()
        wait_done(client, job["id"])
    record = tmp_path / ".macrostitch/jobs" / (job["id"] + ".json")
    saved = json.loads(record.read_text())
    saved[field] = value
    record.write_text(json.dumps(saved))
    with TestClient(api.create_app(tmp_path)) as client:
        assert client.get("/jobs").json() == {"jobs": [], "total": 0}
        assert client.get("/jobs/" + job["id"]).status_code == 404


def test_disappearing_scratch_file_does_not_fail_submission(tmp_path, monkeypatch):
    (tmp_path / "clip.mov").write_bytes(b"video")
    scratch = tmp_path / ".macrostitch/cache/frames/scratch.tmp"
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"temporary")
    original = Path.lstat

    def lstat(path):
        if path == scratch:
            scratch.unlink(missing_ok=True)
        return original(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    with TestClient(api.create_app(tmp_path)) as client:
        response = client.post("/jobs", json=payload())
        assert response.status_code == 202
        assert wait_done(client, response.json()["id"])["status"] == "completed"
