import json
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from macrostitch import cache_io as c


def pixels(value=12000):return np.full((12,16,3),value,np.uint16)


def test_atomic_write_and_fast_reuse(tmp_path,monkeypatch):
    path=tmp_path/'image.tiff';record=c.atomic_image_write(path,pixels(),[cv2.IMWRITE_TIFF_COMPRESSION,8])
    assert record['shape']==[12,16,3] and record['dtype']=='uint16'
    assert set(record['stat'])=={'size','mtime_ns','ctime_ns','dev','ino'}
    monkeypatch.setattr(c.cv2,'imread',lambda *a,**kw:pytest.fail('Unchanged cache should not decode'))
    assert c.image_record(path,(12,16))==record
    assert c.image_record(path,(13,16,3)) is None


def test_legacy_image_validates_and_stamps_once(tmp_path):
    path=tmp_path/'legacy.tiff';assert cv2.imwrite(str(path),pixels())
    record=c.image_record(path,(12,16,3))
    assert record is not None
    assert json.loads(Path(str(path)+'.verified.json').read_text())==record


@pytest.mark.parametrize('damage',['truncated','wrong_shape','wrong_dtype','changed_same_shape'])
def test_corrupt_or_changed_image_rejected_without_adoption(tmp_path,damage):
    path=tmp_path/'image.tiff';before=c.atomic_image_write(path,pixels())
    if damage=='truncated':path.write_bytes(b'II*\x00broken')
    elif damage=='wrong_shape':cv2.imwrite(str(path),np.ones((5,5,3),np.uint16))
    elif damage=='wrong_dtype':cv2.imwrite(str(path),np.ones((12,16,3),np.uint8))
    else:cv2.imwrite(str(path),pixels(50000))
    os.utime(path,ns=(path.stat().st_atime_ns,before['stat']['mtime_ns']))
    assert c.image_record(path,(12,16,3)) is None
    assert json.loads(Path(str(path)+'.verified.json').read_text())==before


def test_hardlink_ctime_revalidates_same_content_identity(tmp_path,monkeypatch):
    path=tmp_path/'image.tiff';before=c.atomic_image_write(path,pixels())
    os.link(path,tmp_path/'export.tiff')
    monkeypatch.setattr(c.cv2,'imread',lambda *a,**kw:pytest.fail('Unchanged content must not decode after hardlink ctime change'))
    after=c.image_record(path,(12,16,3))
    assert after['sha256']==before['sha256']
    assert after['stat']['ctime_ns']!=before['stat']['ctime_ns']


@pytest.mark.parametrize('failure',['false','exception'])
def test_failed_encoding_preserves_previous_image_and_stamp(tmp_path,monkeypatch,failure):
    path=tmp_path/'image.tiff';before=c.atomic_image_write(path,pixels());content=path.read_bytes()
    def fail(temporary,*args):
        assert Path(temporary).suffix==path.suffix and Path(temporary).parent==path.parent
        Path(temporary).write_bytes(b'partial')
        if failure=='exception':raise cv2.error('fixture failure')
        return False
    monkeypatch.setattr(c.cv2,'imwrite',fail)
    with pytest.raises((RuntimeError,cv2.error)):c.atomic_image_write(path,pixels(50000))
    assert path.read_bytes()==content
    assert c.image_record(path,(12,16,3))==before
    assert sorted(p.name for p in tmp_path.iterdir())==['image.tiff','image.tiff.verified.json']


def test_atomic_rewrite_replaces_corrupted_content_identity(tmp_path):
    path=tmp_path/'image.tiff';before=c.atomic_image_write(path,pixels())
    path.write_bytes(b'damaged')
    after=c.atomic_image_write(path,pixels(50000))
    assert after['sha256']!=before['sha256']
    assert c.image_record(path,(12,16,3))==after


def test_missing_and_wrong_legacy_images_return_none(tmp_path):
    assert c.image_record(tmp_path/'absent.tiff',(12,16,3)) is None
    path=tmp_path/'wrong.png';cv2.imwrite(str(path),np.ones((12,16,4),np.uint16))
    assert c.image_record(path,(12,16,3)) is None


def test_encoder_cannot_silently_downconvert_cache_to_jpeg(tmp_path):
    with pytest.raises(ValueError,match='TIFF or PNG'):
        c.atomic_image_write(tmp_path/'image.jpg',pixels())
    assert not list(tmp_path.iterdir())


def test_uint8_proxy_validation_requires_explicit_dtype(tmp_path,monkeypatch):
    path=tmp_path/'proxy.png';proxy=np.full((12,16,3),90,np.uint8)
    assert cv2.imwrite(str(path),proxy)
    assert c.image_record(path,(12,16,3)) is None
    record=c.image_record(path,(12,16,3),dtype=np.uint8)
    assert record['dtype']=='uint8'
    assert c.image_record(path,(12,16,3)) is None
    monkeypatch.setattr(c.cv2,'imread',lambda *a,**kw:pytest.fail('Verified proxy should reuse record'))
    assert c.image_record(path,(12,16,3),dtype=np.uint8)==record
    assert c.stamp_image(path,proxy,dtype=np.uint8)['sha256']==record['sha256']
    with pytest.raises(ValueError,match='uint16'):
        c.atomic_image_write(tmp_path/'copy.png',proxy)
