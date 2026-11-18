import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import cv2
import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'tools/benchmark_native.py'
spec=importlib.util.spec_from_file_location('benchmark_native',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.fixture
def fixture(tmp_path):
    source=tmp_path/'source.mov';source.write_bytes(b'source')
    cache=tmp_path/'existing-cache';cache.mkdir();(cache/'original').write_bytes(b'keep')
    reference=tmp_path/'reference';reference.mkdir()
    config=module.Config();config.validate()
    config_path=tmp_path/'config.json';module.b.write_json(config_path,asdict(config))
    cv2.imwrite(str(reference/'stack.tiff'),np.full((5,7,3),12000,np.uint16))
    for name in ('source_index.png','confidence.png'):
        cv2.imwrite(str(reference/name),np.ones((5,7),np.uint16))
    module.b.write_json(reference/'selection.json',{'reference_index':0,'selected':[
        {'source':0,'frame':2,'candidate_index':1,'transform':[[1,0,0],[0,1,0]],'full_transform':[[1,0,0],[0,1,0]]}]})
    report={'config':asdict(config),'sources':[{'path':str(source),'sha256':module.digest(source)}],
            'artifact_sha256':{p.name:module.digest(p) for p in reference.iterdir()}}
    module.b.write_json(reference/'report.json',report)
    return config_path,reference,tmp_path/'bench',cache


@pytest.mark.parametrize("keep_artifacts", [False, True])
def test_warm_only_runner_retains_existing_files_and_checks_maps(fixture,monkeypatch,keep_artifacts):
    config_path,reference,output,cache=fixture
    calls=[]
    def launch(request,directory):
        calls.append(request.copy());directory.mkdir()
        target=Path(request['output'])
        if request['mode']=='warm_recompute':
            assert not target.exists()
            shutil.copytree(reference,target)
        else:assert target.exists()
        return {**request,'status':'ok','cache_hit':request['mode']=='final_cache'}
    monkeypatch.setattr(module.b,'launch',launch)
    monkeypatch.setattr(module.b,'environment',lambda:{})
    report=module.run(config_path,reference,output,cache,keep_artifacts=keep_artifacts)
    assert [r['mode'] for r in calls]==['warm_recompute','warm_recompute','final_cache']
    assert all(r['cache']==str(cache) for r in calls)
    assert calls[1]['output']==calls[2]['output']
    assert report['all_runs_match_reference'] and report['provenance_unchanged']
    assert not report['repeatability']['cold_repeatability_tested']
    assert (cache/'original').read_bytes()==b'keep'
    if not keep_artifacts:
        assert all(not (Path(row['output'])/'stack.tiff').exists() for row in calls)
        assert all((Path(row['output'])/'report.json').exists() for row in calls)
        return
    altered=Path(calls[0]['output']);cv2.imwrite(str(altered/'source_index.png'),np.zeros((5,7),np.uint16))
    compared=module.compare(module.evidence(reference),module.evidence(altered))
    assert compared['pixels_equal'] and not compared['maps_equal']
    module.b.write_json(altered/'local-alignment.json',{'correction':'changed'})
    compared=module.compare(module.evidence(reference),module.evidence(altered))
    assert not compared['local_alignment_equal']


def test_empty_cache_and_config_mismatch_rejected(fixture):
    config,reference,output,cache=fixture
    (cache/'original').unlink()
    with pytest.raises(ValueError,match='populated cache'):
        module.run(config,reference,output,cache)
    (cache/'original').write_bytes(b'keep')
    data=json.loads(config.read_text());data['max_frames']=3
    module.b.write_json(config,data)
    with pytest.raises(ValueError,match='config does not match'):
        module.run(config,reference,output,cache)


def test_default_native_cleanup_after_failed_worker_preserves_external_cache(fixture, monkeypatch):
    config, reference, output, cache = fixture
    def fail(request, directory):
        directory.mkdir()
        (directory / 'worker.log').write_text('failure')
        target = Path(request['output']); target.mkdir()
        (target / 'partial.tiff').write_bytes(b'partial')
        (target / 'report.json').write_text('{}')
        raise RuntimeError('worker failed')
    monkeypatch.setattr(module.b, 'launch', fail)
    monkeypatch.setattr(module.b, 'environment', lambda: {})
    with pytest.raises(RuntimeError, match='worker failed'):
        module.run(config, reference, output, cache)
    session, = output.iterdir()
    assert not (session / 'warm-000' / 'partial.tiff').exists()
    assert (session / 'warm-000' / 'report.json').exists()
    assert (cache / 'original').read_bytes() == b'keep'
    assert (reference / 'stack.tiff').exists()
    assert json.loads((session / 'benchmark.json').read_text())['status'] == 'error'


def test_native_failure_bounds_existing_cache_under_pipeline_lock(fixture, monkeypatch):
    config, reference, output, cache = fixture
    frame = cache / 'frames' / 'generated'; frame.mkdir(parents=True)
    (frame / 'frame.tiff').write_bytes(b'generated cache')
    monkeypatch.setattr(module, 'DEFAULT_CACHE_MAX_BYTES', 0)
    monkeypatch.setattr(module.b, 'environment', lambda: {})
    def fail(*args):
        raise RuntimeError('worker failure')
    monkeypatch.setattr(module.b, 'launch', fail)
    with pytest.raises(RuntimeError, match='worker failure'):
        module.run(config, reference, output, cache)
    assert not frame.exists()
    assert (cache / 'original').read_bytes() == b'keep'
    session, = output.iterdir()
    report = json.loads((session / 'benchmark.json').read_text())
    assert report['cache_retention']['after_bytes'] == 0
    assert report['cache_retention']['removed_entries'] == ['frames/generated']
