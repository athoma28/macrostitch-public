import json
import sys

import pytest

from macrostitch import api


@pytest.mark.parametrize("recipe,expected", [
    ("[]", "JSON object"), ("{", "Expecting"),
    ('{"threads":0}', "threads:"), ('{"unknown":true}', "unknown:"),
])
def test_bad_cli_recipe_is_actionable_without_traceback(tmp_path, monkeypatch, capsys, recipe, expected):
    source = tmp_path / "recipe.json"
    source.write_text(recipe)
    monkeypatch.setattr(sys, "argv", ["macrostitch", "stack", "clip.mov", "--output", "out", "--config", str(source)])
    monkeypatch.setattr(api, "run_stack", lambda *args, **kwargs: pytest.fail("Invalid recipe must not start work"))
    with pytest.raises(SystemExit) as failure:
        api.main()
    assert failure.value.code == 2
    message = capsys.readouterr().err
    assert expected in message and "Traceback" not in message


@pytest.mark.parametrize("style", ["plain", "json", "quiet"])
def test_cli_progress_styles_keep_final_result_machine_readable(monkeypatch, capsys, style):
    event = {"stage": "native_scan", "message": "Examining source frames", "completed": 100, "total": 200, "elapsed_seconds": 12.5}

    def worker(inputs, output, config, progress, **kwargs):
        progress(event)
        return {"artifacts": {"stack.png": "out/stack.png"}}

    monkeypatch.setattr(api, "run_stack", worker)
    monkeypatch.setattr(sys, "argv", ["macrostitch", "stack", "clip.mov", "--output", "out", "--progress", style])
    api.main()
    text = capsys.readouterr().out
    if style == "plain":
        assert text.startswith("[12.5s] Examining source frames (100/200)\n")
        text = text.split("\n", 1)[1]
    elif style == "json":
        first, text = text.split("\n", 1)
        assert json.loads(first) == event
    assert json.loads(text) == {"artifacts": {"stack.png": "out/stack.png"}}


def test_cli_pipeline_failure_has_nonzero_exit_and_short_error(monkeypatch, capsys):
    def worker(*args, **kwargs):
        raise ValueError("All clips must have equal dimensions")
    monkeypatch.setattr(api, "run_stack", worker)
    monkeypatch.setattr(sys, "argv", ["macrostitch", "stack", "clip.mov", "--output", "out"])
    with pytest.raises(SystemExit) as failure:
        api.main()
    assert failure.value.code == 1
    assert capsys.readouterr().err == "Stack failed: All clips must have equal dimensions\n"


def test_negative_cache_limit_rejected_before_work(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["macrostitch", "stack", "clip.mov", "--output", "out", "--cache-max-mib", "-1"])
    monkeypatch.setattr(api, "run_stack", lambda *args, **kwargs: pytest.fail("Invalid budget must not start work"))
    with pytest.raises(SystemExit) as failure:
        api.main()
    assert failure.value.code == 2
    assert "--cache-max-mib must be zero or greater" in capsys.readouterr().err
