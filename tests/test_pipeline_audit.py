"""Independent regressions for concrete dependency/incomplete-cache failures."""
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from macrostitch import pipeline as p, spatial
from macrostitch.cache_io import atomic_image_write


def test_checksum_manifest_list_is_a_cache_miss_not_an_attribute_error(tmp_path):
    config=p.Config();config.validate()
    source={'sha256':'fixture'}
    names=p.artifact_names(config)
    previous={'config':asdict(config),'version':p.VERSION,'sources':[source],
              'artifacts':{name:str(tmp_path/name) for name in names+['report.json']},
              # JSON list has the right members but cannot establish checksums.
              'artifact_sha256':names,'result':{},'coverage':{},'selected_frames':1,
              'candidates':1,'source_frames':1,'elapsed_seconds':0.}
    identifier=p.key([p.VERSION,['fixture'],asdict(config)])
    assert p.final_cache_valid(previous,tmp_path,identifier,config) is False


def test_missing_alignment_dependency_cannot_bless_previous_pixels(tmp_path,monkeypatch):
    old=np.full((64,64,3),12000,np.uint16)
    new=np.full_like(old,42000)
    extracted=tmp_path/'frame.tiff';proxy=tmp_path/'proxy.png'
    atomic_image_write(extracted,old);cv2.imwrite(str(proxy),(old/256).astype(np.uint8))
    record={'source':0,'frame':0,'proxy':str(proxy),'extracted':str(extracted),
            'transform':[[1,0,0],[0,1,0]]}
    calls=[]
    def register(*args,**kwargs):
        calls.append(True)
        return np.eye(2,3,dtype=np.float32),{'correlation':1.}
    monkeypatch.setattr(p,'register',register)
    paths,_=p.align_selected([record],0,tmp_path/'cache',p.Config(),lambda **event:None)
    # A source was correctly regenerated/replaced while a damaged or legacy
    # completion record no longer establishes its previous dependency identity.
    atomic_image_write(extracted,new)
    manifest=paths[0].parent/'transform.json'
    meta=json.loads(manifest.read_text())
    meta.pop('source_sha256');meta.pop('reference_sha256')
    manifest.write_text(json.dumps(meta))
    paths,_=p.align_selected([record],0,tmp_path/'cache',p.Config(),lambda **event:None)
    assert len(calls)==2
    np.testing.assert_array_equal(cv2.imread(str(paths[0]),cv2.IMREAD_UNCHANGED),new)


def test_spatial_crop_invalidates_when_same_aligned_path_gets_new_pixels(tmp_path):
    path=tmp_path/'aligned.tiff';region=(slice(5,20),slice(8,25))
    first=np.full((32,32,3),12000,np.uint16)
    atomic_image_write(path,first)
    np.testing.assert_array_equal(spatial.crop_pixels(path,region),first[region])
    second=np.full_like(first,42000);atomic_image_write(path,second)
    np.testing.assert_array_equal(spatial.crop_pixels(path,region),second[region])


def test_spatial_crop_recovers_interrupted_npy_write(tmp_path):
    path=tmp_path/'aligned.tiff';region=(slice(5,20),slice(8,25))
    pixels=np.full((32,32,3),12000,np.uint16);atomic_image_write(path,pixels)
    spatial.crop_pixels(path,region)
    for cached in tmp_path.glob('spatial-crop-*.npy'):cached.write_bytes(b'\x93NUMPY')
    np.testing.assert_array_equal(spatial.crop_pixels(path,region),pixels[region])


def test_extraction_validates_entire_attempt_before_publishing_any_frame(tmp_path,monkeypatch):
    source={'sha256':'fixture','path':'fixture.mov','width':16,'height':16}
    records=[{'source':0,'frame':10},{'source':0,'frame':20}]
    cache=tmp_path/'cache'
    def malformed_decode(command,log):
        directory=Path(command[-1]).parent
        cv2.imwrite(str(directory/'f000000.tiff'),np.full((16,16,3),12000,np.uint16))
        (directory/'f000001.tiff').write_bytes(b'broken')
    monkeypatch.setattr(p,'command',malformed_decode)
    with pytest.raises(RuntimeError,match='Invalid extracted frame'):
        p.extract([source],records,cache,p.Config(),lambda **event:None)
    assert not (cache/'frames'/'fixture'/'f000010.tiff').exists()
    assert not (cache/'frames'/'fixture'/'f000020.tiff').exists()
    assert not list((cache/'frames'/'fixture').glob('pending-*'))
