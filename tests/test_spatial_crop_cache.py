import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from macrostitch.cache_io import atomic_image_write
from macrostitch.spatial import crop_pixels

REGION=(slice(2,10),slice(3,12))


def source(tmp_path,value=12000):
    path=tmp_path/'source.tiff';pixels=np.full((16,20,3),value,np.uint16)
    atomic_image_write(path,pixels)
    return path,pixels


def test_authorized_source_replacement_invalidates_crop(tmp_path):
    path,pixels=source(tmp_path)
    np.testing.assert_array_equal(crop_pixels(path,REGION),pixels[REGION])
    pixels[:]=50000;atomic_image_write(path,pixels)
    np.testing.assert_array_equal(crop_pixels(path,REGION),pixels[REGION])
    assert len(list(tmp_path.glob('spatial-crop-*.npy')))==2


@pytest.mark.parametrize('damage',['truncated','wrong_shape','wrong_dtype','changed_values','invalid_metadata'])
def test_crop_cache_repairs_corruption(tmp_path,damage):
    path,pixels=source(tmp_path);crop_pixels(path,REGION)
    cache=next(tmp_path.glob('spatial-crop-*.npy'))
    if damage=='truncated':cache.write_bytes(b'\x93NUMPY')
    elif damage=='wrong_shape':np.save(cache,np.zeros((2,2,3),np.uint16))
    elif damage=='wrong_dtype':np.save(cache,np.zeros((8,9,3),np.uint8))
    elif damage=='changed_values':np.save(cache,np.zeros((8,9,3),np.uint16))
    else:Path(str(cache)+'.verified.json').write_text('[]')
    np.testing.assert_array_equal(crop_pixels(path,REGION),pixels[REGION])


def test_hardlink_ctime_does_not_change_crop_identity(tmp_path):
    path,pixels=source(tmp_path);crop_pixels(path,REGION)
    cache=next(tmp_path.glob('spatial-crop-*.npy'));stamp=cache.stat().st_mtime_ns
    os.link(path,tmp_path/'export.tiff')
    np.testing.assert_array_equal(crop_pixels(path,REGION),pixels[REGION])
    assert list(tmp_path.glob('spatial-crop-*.npy'))==[cache]
    assert cache.stat().st_mtime_ns==stamp


def test_legacy_crop_is_not_adopted_by_shape(tmp_path):
    path,pixels=source(tmp_path)
    np.save(tmp_path/'spatial-crop-source.tiff-3-2-12-10.npy',np.zeros((8,9,3),np.uint16))
    np.testing.assert_array_equal(crop_pixels(path,REGION),pixels[REGION])


def test_failed_atomic_array_write_retains_prior_cache(tmp_path,monkeypatch):
    path,pixels=source(tmp_path);crop_pixels(path,REGION)
    cached=next(tmp_path.glob('spatial-crop-*.npy'));before=cached.read_bytes()
    Path(str(cached)+'.verified.json').write_text('{}')
    def fail(path,*args,**kwargs):
        Path(path).write_bytes(b'partial');raise OSError('fixture write failure')
    monkeypatch.setattr(np,'save',fail)
    with pytest.raises(OSError,match='fixture write failure'):crop_pixels(path,REGION)
    assert cached.read_bytes()==before
    assert not list(tmp_path.glob('.spatial-crop-*'))


def test_unexpected_source_mutation_rejected(tmp_path):
    path,_=source(tmp_path);crop_pixels(path,REGION)
    cv2.imwrite(str(path),np.full((16,20,3),50000,np.uint16))
    with pytest.raises(ValueError,match='integrity'):crop_pixels(path,REGION)
