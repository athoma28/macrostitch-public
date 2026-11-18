"""Atomic writes and stat-accelerated integrity records for immutable BGR caches.

An unchanged full filesystem signature permits fast reuse. Changed signatures
(including ctime changes caused by exported hardlinks) require SHA256 verification.
An unchanged checksum retains the already verified shape/dtype without decoding. Cache consumers should key dependencies on content SHA256, not stat.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


def _sidecar(path):
    return Path(str(path)+'.verified.json')


def _signature(path):
    value=Path(path).stat()
    return {'size':value.st_size,'mtime_ns':value.st_mtime_ns,'ctime_ns':value.st_ctime_ns,
            'dev':value.st_dev,'ino':value.st_ino}


def _hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def _shape(shape):
    result=tuple(shape)
    if len(result)==2:result=(*result,3)
    if len(result)!=3 or result[2]!=3 or any(type(v) is not int or v<=0 for v in result):
        raise ValueError('Expected positive height, width and three BGR channels')
    return list(result)


def _dtype(dtype):
    result=np.dtype(dtype)
    if result not in (np.dtype(np.uint16),np.dtype(np.uint8)):
        raise ValueError('Cache image dtype must be uint16 or uint8')
    return result


def _valid_image(image,shape,dtype=np.uint16):
    return isinstance(image,np.ndarray) and image.dtype==dtype and list(image.shape)==shape


def _read_record(path):
    try:
        data=json.loads(_sidecar(path).read_text())
        if (not isinstance(data,dict) or data.get('schema_version')!=1 or data.get('dtype') not in {'uint16','uint8'}
                or not isinstance(data.get('sha256'),str) or len(data['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in data['sha256'])
                or not isinstance(data.get('stat'),dict)
                or set(data['stat'])!={'size','mtime_ns','ctime_ns','dev','ino'}
                or any(type(v) is not int for v in data['stat'].values())
                or data.get('shape')!=_shape(data.get('shape',()))):
            return None
        return data
    except (OSError,ValueError,TypeError):
        return None


def _save_record(path,record):
    sidecar=_sidecar(path)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+sidecar.name+'-',suffix='.tmp',dir=sidecar.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(record,stream,sort_keys=True,allow_nan=False)
            stream.write('\n')
        os.replace(temporary,sidecar)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _record(shape,signature,checksum,dtype=np.uint16):
    return {'schema_version':1,'shape':shape,'dtype':np.dtype(dtype).name,'channel_order':'BGR',
            'stat':signature,'sha256':checksum}


def stamp_image(path,image,dtype=np.uint16):
    """Stamp an image just written by the caller, replacing prior cache identity.

    The caller must pass the exact BGR image used for the successful write.
    uint8 is allowed only through the explicit dtype argument for proxy caches.
    External/legacy files must instead go through image_record's decode validation.
    """
    path=Path(path);dtype=_dtype(dtype)
    shape=_shape(image.shape)
    if not _valid_image(image,shape,dtype):raise ValueError(f'Cache image must be {dtype.name} BGR')
    before=_signature(path)
    checksum=_hash(path)
    if before!=_signature(path):raise RuntimeError('Image changed while recording cache identity')
    record=_record(shape,before,checksum,dtype)
    _save_record(path,record)
    return record


def image_record(path,shape,dtype=np.uint16):
    """Return a verified BGR record, or None for corrupt/changed content.

    uint16 is the default; dtype=np.uint8 supports explicitly requested proxies.

    Legacy files without valid records are decoded and stamped once. A valid old
    checksum is never silently replaced after an unexpected content mutation.
    Accept either (height,width) or (height,width,3) as the expected shape.
    """
    path=Path(path);shape=_shape(shape);dtype=_dtype(dtype)
    previous=_read_record(path)
    try:
        before=_signature(path)
        if previous is not None:
            if previous['shape']!=shape or previous['dtype']!=dtype.name:return None
            if previous['stat']==before:return previous
            checksum=_hash(path)
            if before!=_signature(path) or previous['sha256']!=checksum:return None
            # Identical encoded bytes preserve the previously decoded shape/dtype.
            # Hardlink exports change ctime without changing those bytes.
            record=_record(shape,before,checksum,dtype)
            _save_record(path,record)
            return record
        image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if not _valid_image(image,shape,dtype):return None
        checksum=_hash(path)
        if before!=_signature(path):return None
        record=_record(shape,before,checksum,dtype)
        _save_record(path,record)
        return record
    except (OSError,cv2.error):
        return None


def atomic_image_write(path,image,params=()):
    """Encode beside the destination, atomically replace it, then stamp identity."""
    path=Path(path)
    if path.suffix.lower() not in {'.tif','.tiff','.png'}:
        raise ValueError('16-bit cache images require TIFF or PNG encoding')
    shape=_shape(image.shape)
    if not _valid_image(image,shape):raise ValueError('Cache image must be uint16 BGR')
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.stem+'-',suffix=path.suffix,dir=path.parent)
    os.close(descriptor)
    try:
        if not cv2.imwrite(temporary,image,list(params)):
            raise RuntimeError(f'Cannot save cache image: {path}')
        os.replace(temporary,path)
        return stamp_image(path,image)
    finally:
        Path(temporary).unlink(missing_ok=True)
