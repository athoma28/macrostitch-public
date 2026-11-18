import json
import os
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
import pytest

from macrostitch import api


def wait_done(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.01)
    pytest.fail("job did not finish")


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "clip.mov").write_bytes(b"dummy")
    return tmp_path


def payload(output="output"):
    return {"inputs": ["clip.mov"], "output_dir": output}


def test_job_progress_artifacts_and_idempotency(workspace, monkeypatch):
    calls = []

    def worker(inputs, output_dir, config, progress):
        calls.append(inputs)
        progress({"stage": "stacking"})
        output = Path(output_dir)
        output.mkdir()
        (output / "stack.png").write_bytes(b"image")
        return {"artifacts": {"stack.png": "stack.png", "forbidden.mov": inputs[0]}}

    monkeypatch.setattr(api, "run_stack", worker)
    with TestClient(api.create_app(workspace)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/jobs", json=payload())
        assert response.status_code == 202
        job = wait_done(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["progress"] == {"stage": "stacking"}
        assert client.get(f"/jobs/{job['id']}/artifacts/stack.png").content == b"image"
        assert client.get(f"/jobs/{job['id']}/artifacts/forbidden.mov").status_code == 404
        assert client.post("/jobs", json=payload()).json()["id"] == job["id"]
        assert len(calls) == 1
        saved = json.loads((workspace / ".macrostitch/jobs" / (job["id"] + ".json")).read_text())
        assert saved["status"] == "completed"


def test_queue_bound_and_output_conflict(workspace, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def worker(*args):
        started.set()
        assert release.wait(3)
        return {}

    monkeypatch.setattr(api, "run_stack", worker)
    with TestClient(api.create_app(workspace, max_pending=1)) as client:
        try:
            first = client.post("/jobs", json=payload()).json()
            assert started.wait(1)
            assert client.post("/jobs", json=payload()).json()["id"] == first["id"]
            assert client.post("/jobs", json=payload("other")).status_code == 429
            changed = payload()
            changed["config"] = {"sample_fps": 5}
            assert client.post("/jobs", json=changed).status_code == 409
            assert client.post("/jobs", json=payload("output/nested")).status_code == 409
        finally:
            release.set()


@pytest.mark.parametrize("change", [
    {"output_dir": "."}, {"output_dir": ".."}, {"output_dir": ".macrostitch/jobs"},
    {"output_dir": "clip.mov"}, {"inputs": ["../escape.mov"]},
    {"inputs": ["https://example.com/video.mov"]},
])
def test_invalid_paths(workspace, change):
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload() | change).status_code == 400


@pytest.mark.parametrize("config", [
    {"sample_fps": 0}, {"max_frames": 10000}, {"roi": [0, 0, 2, 1]},
    {"roi": [1, 0, 0, 1]}, {"threads": 0}, {"unknown": True}, {"coverage": .1},
    {"subject_only": "invalid"},
    {"subject_polygon": []}, {"subject_polygon": [[0, 0], [1, 1]]},
    {"subject_polygon": [[0, 0], [1, 0], [1, 1.1]]},
    {"subject_polygon": [[-.1, 0], [1, 0], [1, 1]]},
    {"subject_polygon": [[0, 0]] * 129},
    {"subject_polygon": [[0, 0, 0], [1, 0], [1, 1]]},
    {"subject_polygon": [["NaN", 0], [1, 0], [1, 1]]},
])
def test_invalid_config(workspace, config):
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload() | {"config": config}).status_code == 422


def test_symlink_escape(workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "clip.mov").write_bytes(b"outside")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload() | {"inputs": ["escape/clip.mov"]}).status_code == 400
        assert client.post("/jobs", json=payload("escape/output")).status_code == 400


def test_failed_and_interrupted_jobs_can_retry(workspace, monkeypatch):
    def failing(*args):
        raise RuntimeError("bad video")

    monkeypatch.setattr(api, "run_stack", failing)
    with TestClient(api.create_app(workspace)) as client:
        first = client.post("/jobs", json=payload()).json()
        assert wait_done(client, first["id"])["status"] == "failed"
        second = client.post("/jobs", json=payload()).json()
        assert second["id"] != first["id"]
        wait_done(client, second["id"])
    file = workspace / ".macrostitch/jobs" / (second["id"] + ".json")
    saved = json.loads(file.read_text())
    saved["status"] = "running"
    file.write_text(json.dumps(saved))
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    with TestClient(api.create_app(workspace)) as client:
        assert client.get(f"/jobs/{second['id']}").json()["status"] == "interrupted"
        third = client.post("/jobs", json=payload()).json()
        assert third["id"] not in {first["id"], second["id"]}
        assert wait_done(client, third["id"])["status"] == "completed"


def test_changed_source_with_restored_mtime_is_new_job(workspace, monkeypatch):
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    with TestClient(api.create_app(workspace)) as client:
        first = client.post("/jobs", json=payload()).json()
        wait_done(client, first["id"])
        source = workspace / "clip.mov"
        stat = source.stat()
        source.write_bytes(b"other")
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = client.post("/jobs", json=payload()).json()
        assert second["id"] != first["id"]


def test_pipeline_version_change_invalidates_completed_job(workspace, monkeypatch):
    monkeypatch.setattr(api, "run_stack", lambda *args: {})
    monkeypatch.setattr(api, "VERSION", "test-v1")
    with TestClient(api.create_app(workspace)) as client:
        first = client.post("/jobs", json=payload()).json()
        assert wait_done(client, first["id"])["status"] == "completed"
        assert client.post("/jobs", json=payload()).json()["id"] == first["id"]
    monkeypatch.setattr(api, "VERSION", "test-v2")
    with TestClient(api.create_app(workspace)) as client:
        second = client.post("/jobs", json=payload()).json()
        assert second["id"] != first["id"]
        assert wait_done(client, second["id"])["status"] == "completed"


def test_modified_artifacts_require_resubmission(workspace, monkeypatch):
    def worker(inputs, output, *args):
        directory = Path(output)
        directory.mkdir(exist_ok=True)
        (directory / "stack.png").write_bytes(b"original")
        return {"artifacts": {"stack.png": "stack.png"}}

    monkeypatch.setattr(api, "run_stack", worker)
    with TestClient(api.create_app(workspace)) as client:
        first = client.post("/jobs", json=payload()).json()
        wait_done(client, first["id"])
        (workspace / "output/stack.png").write_bytes(b"different")
        assert client.get(f"/jobs/{first['id']}/artifacts/stack.png").status_code == 409
        second = client.post("/jobs", json=payload()).json()
        assert second["id"] != first["id"]
        assert wait_done(client, second["id"])["status"] == "completed"


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_output_alias_cannot_overwrite_source(workspace, kind):
    output = workspace / "output"
    output.mkdir()
    if kind == "symlink":
        (output / "stack.png").symlink_to(workspace / "clip.mov")
    else:
        os.link(workspace / "clip.mov", output / "stack.png")
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload()).status_code == 400
    assert (workspace / "clip.mov").read_bytes() == b"dummy"


def test_nested_cache_escape(workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("cache-outside")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / ".macrostitch").symlink_to(outside, target_is_directory=True)
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload("nested/output")).status_code == 400


@pytest.mark.parametrize("options, expected", [({}, True), ({"subject_only": False}, False)])
def test_subject_only_default_and_propagation(workspace, monkeypatch, options, expected):
    received = []

    def worker(inputs, output, config, progress):
        received.append(config.subject_only)
        return {}

    monkeypatch.setattr(api, "run_stack", worker)
    with TestClient(api.create_app(workspace)) as client:
        response = client.post("/jobs", json=payload() | {"config": options})
        assert response.status_code == 202
        job = wait_done(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["config"]["subject_only"] is expected
        assert received == [expected]


@pytest.mark.parametrize("polygon", [None, [[0, 0], [1, 0], [.5, 1]]])
def test_subject_polygon_default_and_propagation(workspace, monkeypatch, polygon):
    received = []

    def worker(inputs, output, config, progress):
        received.append(config.subject_polygon)
        return {}

    monkeypatch.setattr(api, "run_stack", worker)
    options = {} if polygon is None else {"subject_polygon": polygon}
    with TestClient(api.create_app(workspace)) as client:
        response = client.post("/jobs", json=payload() | {"config": options})
        assert response.status_code == 202
        job = wait_done(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["config"]["subject_polygon"] == polygon
        expected = None if polygon is None else tuple(tuple(point) for point in polygon)
        assert received == [expected]


def region(y=(0, 1), **kw):
    return {"y": list(y), "roi": [0, 0, 1, 1], **kw}


@pytest.mark.parametrize("config", [
    {"fusion_mode": "unknown"}, {"background_mode": "unknown"},
    {"fusion_mode": "regional"}, {"background_mode": "transparent"},
    {"focus_regions": []}, {"focus_regions": [region()] * 33},
    {"focus_regions": [region((.1, 1))]}, {"focus_regions": [region((0, .9))]},
    {"focus_regions": [region((0, .5)), region((.6, 1))]},
    {"focus_regions": [region((0, .6)), region((.5, 1))]},
    {"focus_regions": [region((1, 0))]}, {"focus_regions": [region((-1, 1))]},
    {"focus_regions": [region(tone_reference=1)]}, {"focus_regions": [region(tone_reference=-1)]},
    {"focus_regions": [region(tone_reference=.5)]}, {"focus_regions": [region(tone_reference=True)]},
    {"focus_regions": [region(tone_roi=[0, 0, 0, 1])]},
    {"focus_regions": [region(roi=[0, 0, 2, 1])]},
    {"focus_regions": [region(unexpected=True)]},
    {"focus_regions": [{"roi": [0, 0, 1, 1]}]},
    {"score_normalization_roi": [0, 1, 1, 0]},
    {"region_blend_px": -.1}, {"region_blend_px": 32.1},
    {"matte_feather_px": .09}, {"matte_feather_px": 8.1},
    {"matte_erode_px": -.1}, {"matte_erode_px": 8.1},
    {"region_blend_px": "NaN"}, {"matte_feather_px": "Infinity"},
])
def test_invalid_regional_options(workspace, config):
    with TestClient(api.create_app(workspace)) as client:
        assert client.post("/jobs", json=payload() | {"config": config}).status_code == 422


def test_regional_schema_dumps_plain_dicts():
    options = api.StackOptions(fusion_mode="regional", background_mode="transparent",
        subject_polygon=[[0, 0], [1, 0], [.5, 1]],
        focus_regions=[region((0, .5)), region((.5, 1), tone_reference=0, tone_roi=[0, 0, .5, .5])],
        score_normalization_roi=[0, 0, 1, 1], region_blend_px=32, matte_feather_px=.1, matte_erode_px=0)
    data = options.model_dump()
    assert isinstance(data["focus_regions"][0], dict)
    assert data["focus_regions"][1]["tone_reference"] == 0
    assert data["fusion_mode"] == "regional" and data["background_mode"] == "transparent"


def test_regional_options_propagate_to_pipeline(workspace, monkeypatch):
    received = []
    def worker(inputs, output, config, progress):
        received.append(config)
        return {}
    monkeypatch.setattr(api, "run_stack", worker)
    options = {"fusion_mode": "regional", "background_mode": "transparent",
               "subject_polygon": [[0, 0], [1, 0], [.5, 1]],
               "focus_regions": [region((0, .5)), region((.5, 1), tone_reference=0)],
               "score_normalization_roi": [.1, .2, .8, .9], "region_blend_px": 4,
               "matte_feather_px": 2, "matte_erode_px": .5}
    with TestClient(api.create_app(workspace)) as client:
        response = client.post("/jobs", json=payload() | {"config": options})
        assert response.status_code == 202
        job = wait_done(client, response.json()["id"])
        assert job["status"] == "completed", job.get("error")
    config = received[0]
    assert config.fusion_mode == "regional" and config.background_mode == "transparent"
    assert isinstance(config.focus_regions[0], dict)
    assert config.focus_regions[1]["tone_reference"] == 0
    assert tuple(config.score_normalization_roi) == (.1, .2, .8, .9)
    assert (config.region_blend_px, config.matte_feather_px, config.matte_erode_px) == (4, 2, .5)


@pytest.mark.parametrize('mode',['common','reference'])
def test_canvas_mode_propagates(workspace,monkeypatch,mode):
    received=[]
    def worker(inputs,output,config,progress):
        received.append(config.canvas_mode)
        return {}
    monkeypatch.setattr(api,'run_stack',worker)
    with TestClient(api.create_app(workspace)) as client:
        response=client.post('/jobs',json=payload()|{'config':{'canvas_mode':mode}})
        assert response.status_code==202
        assert wait_done(client,response.json()['id'])['status']=='completed'
    assert received==[mode]


def test_canvas_schema_default_and_invalid(workspace):
    assert api.StackOptions().canvas_mode=='common'
    with TestClient(api.create_app(workspace)) as client:
        assert client.post('/jobs',json=payload()|{'config':{'canvas_mode':'crop'}}).status_code==422


@pytest.mark.parametrize('options',[
    {'selection_mode':'unknown'}, {'spatial_blend_px':-.01}, {'spatial_blend_px':4.01},
    {'spatial_smoothness':-.01}, {'spatial_smoothness':1.01},
    {'spatial_blend_px':'NaN'}, {'spatial_smoothness':'Infinity'},
])
def test_invalid_spatial_options(workspace,options):
    with TestClient(api.create_app(workspace)) as client:
        assert client.post('/jobs',json=payload()|{'config':options}).status_code==422


@pytest.mark.parametrize('blend,smoothness',[(0,0),(4,1),(1,.12)])
def test_spatial_options_propagate(workspace,monkeypatch,blend,smoothness):
    received=[]
    def worker(inputs,output,config,progress):
        received.append(config)
        return {}
    monkeypatch.setattr(api,'run_stack',worker)
    options={'fusion_mode':'spatial','selection_mode':'native','canvas_mode':'reference',
             'background_mode':'reference','spatial_blend_px':blend,'spatial_smoothness':smoothness}
    with TestClient(api.create_app(workspace)) as client:
        response=client.post('/jobs',json=payload()|{'config':options})
        assert response.status_code==202
        job=wait_done(client,response.json()['id'])
        assert job['status']=='completed',job.get('error')
    config=received[0]
    assert config.fusion_mode=='spatial' and config.selection_mode=='native'
    assert config.spatial_blend_px==blend and config.spatial_smoothness==smoothness
    assert config.canvas_mode=='reference' and config.background_mode=='reference'


def test_spatial_config_defaults_canonicalization_and_validation():
    assert api.StackOptions().selection_mode=='proxy'
    config=api.Config(spatial_blend_px=1,spatial_smoothness='0.12')
    config.validate()
    assert type(config.spatial_blend_px) is float
    assert config.spatial_smoothness==.12
    for options in ({'selection_mode':'bad'},{'spatial_blend_px':float('nan')},
                    {'spatial_smoothness':float('inf')},{'spatial_blend_px':4.01},
                    {'spatial_smoothness':-1}):
        with pytest.raises(ValueError):
            api.Config(**options).validate()


def test_executor_rejection_does_not_leave_queued_job(workspace,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    real_submit=ThreadPoolExecutor.submit
    attempts=[]
    def reject_once(executor,*args,**kwargs):
        attempts.append(True)
        if len(attempts)==1:raise RuntimeError('temporary executor rejection')
        return real_submit(executor,*args,**kwargs)
    monkeypatch.setattr(ThreadPoolExecutor,'submit',reject_once)
    monkeypatch.setattr(api,'run_stack',lambda *args:{})
    with TestClient(api.create_app(workspace),raise_server_exceptions=False) as client:
        assert client.post('/jobs',json=payload()).status_code==500
        second=client.post('/jobs',json=payload())
        assert second.status_code==202
        assert wait_done(client,second.json()['id'])['status']=='completed'
    saved=[json.loads(p.read_text()) for p in (workspace/'.macrostitch/jobs').glob('*.json')]
    assert sorted(j['status'] for j in saved)==['completed','failed']


def test_incomplete_saved_job_does_not_break_submissions(workspace,monkeypatch):
    state=workspace/'.macrostitch/jobs';state.mkdir(parents=True)
    identifier='a'*32
    (state/f'{identifier}.json').write_text(json.dumps({'id':identifier,'status':'completed'}))
    monkeypatch.setattr(api,'run_stack',lambda *args:{})
    with TestClient(api.create_app(workspace),raise_server_exceptions=False) as client:
        response=client.post('/jobs',json=payload())
        assert response.status_code==202
        assert wait_done(client,response.json()['id'])['status']=='completed'
        assert client.get(f'/jobs/{identifier}').status_code==404


def test_queued_output_cannot_be_redirected_to_another_workspace_directory(workspace,monkeypatch):
    started,release=threading.Event(),threading.Event()
    writes=[]
    def worker(inputs,output,config,progress):
        if Path(output).name=='blocker':
            started.set();assert release.wait(3)
        else:
            writes.append(Path(output).resolve())
        return {}
    monkeypatch.setattr(api,'run_stack',worker)
    protected=workspace/'existing-result';protected.mkdir()
    (protected/'original').write_bytes(b'keep')
    with TestClient(api.create_app(workspace)) as client:
        try:
            first=client.post('/jobs',json=payload('blocker')).json()
            assert started.wait(1)
            queued=client.post('/jobs',json=payload('queued-output')).json()
            (workspace/'queued-output').symlink_to(protected,target_is_directory=True)
        finally:release.set()
        assert wait_done(client,first['id'])['status']=='completed'
        done=wait_done(client,queued['id'])
        assert done['status']=='failed'
        assert 'Output path changed' in done['error']
    assert writes==[] and (protected/'original').read_bytes()==b'keep'


@pytest.mark.parametrize('invalid',[[],None,'invalid',7])
def test_malformed_saved_artifact_signatures_ignored(workspace,monkeypatch,invalid):
    def worker(inputs,output,*args):
        output=Path(output);output.mkdir(exist_ok=True)
        (output/'stack.png').write_bytes(b'result')
        return {'artifacts':{'stack.png':'stack.png'}}
    monkeypatch.setattr(api,'run_stack',worker)
    with TestClient(api.create_app(workspace)) as client:
        first=client.post('/jobs',json=payload()).json()
        assert wait_done(client,first['id'])['status']=='completed'
    path=workspace/'.macrostitch/jobs'/f"{first['id']}.json"
    saved=json.loads(path.read_text());saved['artifact_signatures']=invalid
    path.write_text(json.dumps(saved))
    with TestClient(api.create_app(workspace)) as client:
        assert client.get(f"/jobs/{first['id']}").status_code==404
        retry=client.post('/jobs',json=payload())
        assert retry.status_code==202
        assert wait_done(client,retry.json()['id'])['status']=='completed'


def test_artifact_equal_length_edit_with_restored_mtime_invalidates(workspace,monkeypatch):
    def worker(inputs,output,*args):
        output=Path(output);output.mkdir(exist_ok=True)
        (output/'stack.png').write_bytes(b'original')
        return {'artifacts':{'stack.png':'stack.png'}}
    monkeypatch.setattr(api,'run_stack',worker)
    with TestClient(api.create_app(workspace)) as client:
        first=client.post('/jobs',json=payload()).json()
        assert wait_done(client,first['id'])['status']=='completed'
        artifact=workspace/'output/stack.png';before=artifact.stat()
        artifact.write_bytes(b'modified')
        os.utime(artifact,ns=(before.st_atime_ns,before.st_mtime_ns))
        after=artifact.stat()
        assert (after.st_size,after.st_mtime_ns)==(before.st_size,before.st_mtime_ns)
        assert client.get(f"/jobs/{first['id']}/artifacts/stack.png").status_code==409
        retry=client.post('/jobs',json=payload())
        assert retry.status_code==202 and retry.json()['id']!=first['id']
        assert wait_done(client,retry.json()['id'])['status']=='completed'


def local_region(**changes):
    return {'roi':[.1,.1,.9,.9],'fit_roi':[.2,.2,.5,.4],
            'validation_rois':[[.2,.6,.5,.8]],'anchor':{'source':0,'frame':12},**changes}


@pytest.mark.parametrize('changes',[
    {'roi':[0,0,2,1]}, {'roi':[.8,.1,.1,.9]},
    {'fit_roi':[0,.2,.5,.4]}, {'fit_roi':[.2,.2,.2,.4]},
    {'validation_rois':[[.2,.3,.5,.7]]}, {'validation_rois':[[0,.6,.5,.8]]},
    {'validation_rois':[]}, {'validation_rois':[[.2,.6,.5,.8]]*9},
    {'anchor':{'source':-1,'frame':12}}, {'anchor':{'source':0,'frame':-1}},
    {'anchor':{'source':True,'frame':12}}, {'anchor':{'source':0,'frame':1.5}},
    {'anchor':{'source':0,'frame':12,'unknown':1}},
    {'max_shift_px':.15}, {'max_shift_px':6.01}, {'max_shift_px':'NaN'},
    {'feather_px':0}, {'feather_px':65}, {'feather_px':1.5}, {'unknown':True},
])
def test_invalid_local_alignment_regions(workspace,changes):
    options={'fusion_mode':'spatial','selection_mode':'native','local_alignment_regions':[local_region(**changes)]}
    with TestClient(api.create_app(workspace)) as client:
        assert client.post('/jobs',json=payload()|{'config':options}).status_code==422


@pytest.mark.parametrize('options',[
    {'local_alignment_regions':[local_region()]},
    {'fusion_mode':'spatial','local_alignment_regions':[local_region()]},
    {'selection_mode':'native','local_alignment_regions':[local_region()]},
    {'fusion_mode':'spatial','selection_mode':'native','local_alignment_regions':[]},
    {'fusion_mode':'spatial','selection_mode':'native','local_alignment_regions':[local_region()]*9},
    {'fusion_mode':'spatial','selection_mode':'native','local_alignment_regions':[local_region(),local_region()]},
])
def test_local_alignment_cross_field_validation(workspace,options):
    with TestClient(api.create_app(workspace)) as client:
        assert client.post('/jobs',json=payload()|{'config':options}).status_code==422


def test_local_alignment_config_serialization_and_propagation(workspace,monkeypatch):
    options=api.StackOptions(fusion_mode='spatial',selection_mode='native',local_alignment_regions=[local_region()])
    plain=options.model_dump()
    region=plain['local_alignment_regions'][0]
    assert isinstance(region,dict) and isinstance(region['anchor'],dict)
    assert region['anchor']=={'source':0,'frame':12}
    assert region['max_shift_px']==6 and region['feather_px']==24
    received=[]
    def worker(inputs,output,config,progress):
        received.append(config.local_alignment_regions)
        return {}
    monkeypatch.setattr(api,'run_stack',worker)
    with TestClient(api.create_app(workspace)) as client:
        response=client.post('/jobs',json=payload()|{'config':plain})
        assert response.status_code==202
        job=wait_done(client,response.json()['id'])
        assert job['status']=='completed',job.get('error')
    assert received[0][0]['anchor']=={'source':0,'frame':12}
