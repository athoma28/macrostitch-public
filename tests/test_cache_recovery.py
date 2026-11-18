"""Corrupt-cache recovery regressions using only small images and mocked decoding."""
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from macrostitch import pipeline as p


def emit(**event):
    pass


@pytest.fixture
def mock_stack(tmp_path,monkeypatch):
    source=tmp_path/'source.mov';source.write_bytes(b'fixture')
    image=tmp_path/'frame.tiff';cv2.imwrite(str(image),np.ones((32,32,3),np.uint16))
    record={'source':0,'frame':0,'accepted':True,'extracted':str(image),'aligned':str(image)}
    info={'path':str(source),'sha256':p.digest(source),'width':32,'height':32,'frames':1}
    monkeypatch.setattr(p,'probe',lambda path:dict(info))
    monkeypatch.setattr(p,'analyze',lambda *a:([dict(record)],np.ones((1,2,2),np.float32),0,tmp_path))
    monkeypatch.setattr(p,'select_coverage',lambda *a:([0],{}))
    monkeypatch.setattr(p,'extract',lambda *a:None)
    monkeypatch.setattr(p,'align_selected',lambda *a:([image],np.ones((32,32),np.uint8)))
    calls=[]
    def fuse(paths,reference,common,config,output,callback):
        calls.append(True)
        for name in ('stack.tiff','stack.png','preview.jpg','comparison.jpg','source_index.png','confidence.png'):
            pixels=(np.full((32,32,3),80,np.uint8) if name.endswith('.jpg') else
                    np.full((32,32),12000,np.uint16) if name in ('source_index.png','confidence.png') else
                    np.full((32,32,3),12000,np.uint16))
            assert cv2.imwrite(str(output/name),pixels)
        return {'width':32,'height':32}
    monkeypatch.setattr(p,'fuse',fuse)
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**kw:'ffmpeg fixture')
    output=tmp_path/'result';config=p.Config()
    first=p.run_stack([source],output,config)
    assert not first['cache_hit']
    return source,output,config,calls


@pytest.mark.parametrize('damage',['invalid_json','missing_artifacts','empty_artifacts','partial_checksums','json_list'])
def test_final_cache_recovers_invalid_or_incomplete_manifest(mock_stack,damage):
    source,output,config,calls=mock_stack
    path=output/'report.json';report=json.loads(path.read_text())
    if damage=='invalid_json':path.write_text('{"incomplete":')
    elif damage=='json_list':path.write_text('[]')
    else:
        if damage=='missing_artifacts':report.pop('artifacts')
        elif damage=='empty_artifacts':report['artifacts']={}
        else:
            report['artifact_sha256']={'preview.jpg':report['artifact_sha256']['preview.jpg']}
            (output/'stack.tiff').unlink()
        path.write_text(json.dumps(report))
    result=p.run_stack([source],output,config)
    assert result['cache_hit'] is False
    assert len(calls)==2
    assert (output/'stack.tiff').exists()
    assert 'stack.tiff' in result['artifact_sha256']


def test_frame_exports_are_opt_in_and_removed_on_cached_default_run(mock_stack):
    source,output,config,calls=mock_stack
    assert not (output/'frames').exists() and not (output/'aligned').exists()
    first=json.loads((output/'report.json').read_text())
    exported=p.run_stack([source],output,config,keep_frames=True)
    assert not exported['cache_hit'] and len(calls)==2
    for directory in ('frames','aligned'):
        files=list((output/directory).glob('*.tiff'))
        assert len(files)==1 and files[0].stat().st_nlink>=2
    cached=p.run_stack([source],output,config,keep_frames=True)
    assert cached['cache_hit'] and len(calls)==2
    compact=p.run_stack([source],output,config)
    assert compact['cache_hit'] and len(calls)==2
    assert first['artifact_sha256']['stack.tiff']==compact['artifact_sha256']['stack.tiff']
    assert exported['artifact_sha256']==compact['artifact_sha256']
    assert not (output/'frames').exists() and not (output/'aligned').exists()


@pytest.fixture
def analysis_fixture(tmp_path,monkeypatch):
    proxies=tmp_path/'proxies';proxies.mkdir()
    image=np.random.default_rng(3).integers(0,255,(64,64,3),dtype=np.uint8)
    cv2.imwrite(str(proxies/'p000000.png'),image)
    meta={'timestamp_basis':'fixture','timestamps':[0.],'count':1,'step':1}
    monkeypatch.setattr(p,'proxies',lambda *a:(proxies,meta))
    source={'sha256':'fixture','frames':1,'fps':1}
    cache=tmp_path/'cache';config=p.Config(proxy_width=64)
    records,scores,reference,directory=p.analyze([source],cache,config,emit)
    return source,cache,config,directory,records,scores


@pytest.mark.parametrize('damage',['truncated_scores','invalid_metadata','wrong_score_shape','nonfinite_scores','missing_record_fields'])
def test_analysis_recovers_invalid_cache(analysis_fixture,damage):
    source,cache,config,directory,records,expected=analysis_fixture
    if damage=='truncated_scores':(directory/'scores.npy').write_bytes(b'\x93NUMPY')
    elif damage=='invalid_metadata':(directory/'analysis.json').write_text('{')
    elif damage=='wrong_score_shape':np.save(directory/'scores.npy',np.ones((20,1,1),np.float32))
    elif damage=='nonfinite_scores':np.save(directory/'scores.npy',np.full_like(expected,np.nan))
    else:(directory/'analysis.json').write_text(json.dumps({'reference':0,'records':[{'source':0}]}))
    rebuilt,actual,reference,_=p.analyze([source],cache,config,emit)
    assert actual.shape==expected.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual,expected)
    assert len(rebuilt)==1 and 'transform' in rebuilt[0] and 'proxy' in rebuilt[0]
    assert reference==0


def test_aligned_cache_rebuilds_truncated_tiff(tmp_path,monkeypatch):
    image=np.random.default_rng(9).integers(0,65535,(64,64,3),dtype=np.uint16)
    extracted=tmp_path/'extracted.tiff';proxy=tmp_path/'proxy.png'
    cv2.imwrite(str(extracted),image);cv2.imwrite(str(proxy),(image/256).astype(np.uint8))
    record={'source':0,'frame':0,'proxy':str(proxy),'extracted':str(extracted),
            'transform':[[1,0,0],[0,1,0]]}
    monkeypatch.setattr(p,'register',lambda *a,**k:(np.eye(2,3,dtype=np.float32),{'correlation':1.}))
    paths,_=p.align_selected([record],0,tmp_path/'cache',p.Config(),emit)
    paths[0].write_bytes(b'II*\x00truncated')
    paths,_=p.align_selected([record],0,tmp_path/'cache',p.Config(),emit)
    restored=cv2.imread(str(paths[0]),cv2.IMREAD_UNCHANGED)
    assert restored is not None
    np.testing.assert_array_equal(restored,image)


def test_extraction_does_not_promote_stale_pending_frame(tmp_path,monkeypatch):
    source={'sha256':'fixture','path':'fixture.mov','width':16,'height':16}
    records=[{'source':0,'frame':10},{'source':0,'frame':20}]
    cache=tmp_path/'cache';pending=cache/'frames'/'fixture'/'pending';pending.mkdir(parents=True)
    stale=np.full((16,16,3),60000,np.uint16)
    cv2.imwrite(str(pending/'f000001.tiff'),stale)
    def short_decode(command,log):
        # A successful short stream produces one selected frame, not requested two.
        directory=Path(command[-1]).parent
        cv2.imwrite(str(directory/'f000000.tiff'),np.full_like(stale,10000))
    monkeypatch.setattr(p,'command',short_decode)
    with pytest.raises(RuntimeError):
        p.extract([source],records,cache,p.Config(),emit)
    assert not (cache/'frames'/'fixture'/'f000020.tiff').exists()


def test_extraction_rebuilds_corrupt_cached_frame(tmp_path,monkeypatch):
    source={'sha256':'fixture','path':'fixture.mov','width':16,'height':16}
    records=[{'source':0,'frame':10}]
    cache=tmp_path/'cache';directory=cache/'frames'/'fixture';directory.mkdir(parents=True)
    (directory/'f000010.tiff').write_bytes(b'partial')
    def decode(command,log):
        target=Path(command[-1]).parent/'f000000.tiff'
        cv2.imwrite(str(target),np.full((16,16,3),12000,np.uint16))
    monkeypatch.setattr(p,'command',decode)
    p.extract([source],records,cache,p.Config(),emit)
    restored=cv2.imread(records[0]['extracted'],cv2.IMREAD_UNCHANGED)
    assert restored is not None and restored.dtype==np.uint16
    assert restored.shape==(16,16,3)


@pytest.mark.parametrize('damage',['finite_scores','valid_transform','malformed_integrity'])
def test_analysis_content_identity_detects_plausible_mutation(analysis_fixture,damage):
    source,cache,config,directory,records,expected=analysis_fixture
    if damage=='finite_scores':
        np.save(directory/'scores.npy',expected+np.float32(.125))
    elif damage=='valid_transform':
        path=directory/'analysis.json';data=json.loads(path.read_text())
        data['records'][0]['transform'][0][2]+=2
        path.write_text(json.dumps(data))
    else:(directory/'analysis.verified.json').write_text('{')
    rebuilt,actual,reference,_=p.analyze([source],cache,config,emit)
    np.testing.assert_array_equal(actual,expected)
    assert rebuilt[0]['transform']==records[0]['transform']
    integrity=json.loads((directory/'analysis.verified.json').read_text())
    assert integrity['scores_sha256']==p.digest(directory/'scores.npy')
    assert integrity['metadata_sha256']==p.digest(directory/'analysis.json')


def test_legacy_analysis_is_adopted_without_registration(analysis_fixture,monkeypatch):
    source,cache,config,directory,records,expected=analysis_fixture
    (directory/'analysis.verified.json').unlink()
    metadata=json.loads((directory/'analysis.json').read_text())
    metadata.pop('integrity_schema')
    (directory/'analysis.json').write_text(json.dumps(metadata))
    def no_recompute(*args,**kwargs):pytest.fail('Validated legacy analysis should be adopted without recomputation')
    monkeypatch.setattr(p,'focus_map',no_recompute)
    monkeypatch.setattr(p,'register',no_recompute)
    rebuilt,actual,reference,_=p.analyze([source],cache,config,emit)
    np.testing.assert_array_equal(actual,expected)
    assert rebuilt==records and reference==0
    assert (directory/'analysis.verified.json').exists()
    # A second call uses the new identity without paying registration cost either.
    _,again,_,_=p.analyze([source],cache,config,emit)
    np.testing.assert_array_equal(again,expected)


def test_marked_analysis_missing_stamp_cannot_adopt_changed_scores(analysis_fixture):
    source,cache,config,directory,records,expected=analysis_fixture
    assert json.loads((directory/'analysis.json').read_text())['integrity_schema']==1
    (directory/'analysis.verified.json').unlink()
    np.save(directory/'scores.npy',expected+np.float32(.125))
    rebuilt,actual,reference,_=p.analyze([source],cache,config,emit)
    np.testing.assert_array_equal(actual,expected)
    assert rebuilt==records and reference==0
    assert (directory/'analysis.verified.json').exists()
