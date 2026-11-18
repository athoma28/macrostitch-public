"""Verified, bounded snapshots before the destructive presentation compositor.

Only controlled spatial renders participate. Selection still runs normally;
snapshots replace extraction, registration and fusion, never coverage evidence.
The caller holds the pipeline cache lock and applies ordinary cache retention.
"""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

SCHEMA = 1
ALIGNMENT_FIELDS = ('extracted', 'aligned', 'full_transform', 'full_alignment')
BASE_FILES = ('stack.png', 'source_index.png', 'confidence.png', 'fusion.json')


def identity(config, sources, selected, reference, version):
    from .compositing import controls_active
    if config.fusion_mode != 'spatial' or not controls_active(config):
        return None
    settings = asdict(config)
    settings.pop('background_opacity')
    settings.pop('background_margin_px')
    settings['object_masks'] = [row for row in settings['object_masks'] or ()
                                if row['action'] == 'stack']
    # All fresh record fields (including geometry and candidate identity) matter.
    # Only fields added by extraction/registration are restored from the snapshot.
    rows = [{k: v for k, v in row.items() if k not in ALIGNMENT_FIELDS} for row in selected]
    payload = [SCHEMA, version, cv2.__version__, np.__version__, sources,
               settings, rows, reference]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError('Nonfinite snapshot metadata')
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    value = json.loads(path.read_text(), parse_constant=invalid, parse_float=finite)
    if not isinstance(value, dict):
        raise ValueError('Snapshot metadata must be an object')
    return value


def _files(config):
    return BASE_FILES + (('local-alignment.json',) if config.local_alignment_regions else ())


def _directory(cache, identifier):
    path = Path(cache) / 'fusion' / identifier
    # Do not follow aliases when loading, replacing or creating private entries.
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        return None
    try:
        return path if path.resolve() == path.absolute() else None
    except (OSError, RuntimeError):
        return None


def _extent(value, width, height):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError('Invalid snapshot crop')
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Snapshot crop outside reference')
    return y1 - y0, x1 - x0


def _image(path, shape, dtype):
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.shape != shape or pixels.dtype != dtype:
        raise ValueError('Invalid snapshot image')
    return pixels


def restore(cache, identifier, output, config, selected, reference, shape):
    """Return (reference paths, common, result), or a clean cache miss.

    No output or selected record is changed until every cached dependency has
    passed its checksum and structural validation. Output files are copies:
    composition must never mutate a shared snapshot inode.
    """
    if identifier is None:
        return None
    directory = _directory(cache, identifier)
    if directory is None:
        return None
    names = _files(config)
    expected = set(names) | {'reference.tiff', 'common.npy', 'state.json'}
    try:
        if (directory / 'complete.json').is_symlink():
            return None
        manifest = _json(directory / 'complete.json')
        if manifest.get('schema') != SCHEMA or manifest.get('identity') != identifier:
            return None
        checksums = manifest['sha256']
        if not isinstance(checksums, dict) or set(checksums) != expected:
            return None
        for name, checksum in checksums.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or _digest(path) != checksum:
                return None
        state = _json(directory / 'state.json')
        h, w = shape
        if (type(state['reference']) is not int or state['reference'] != reference
                or state['shape'] != [h, w]):
            return None
        result = state['result']
        if not isinstance(result, dict) or result.get('fusion_mode') != 'spatial':
            return None
        ch, cw = _extent(result['crop_xyxy'], w, h)
        _extent(result['fusion_roi_xyxy'], w, h)
        if (type(result['width']) is not int or result['width'] != cw
                or type(result['height']) is not int or result['height'] != ch
                or not isinstance(result['diagnostics'], dict)
                or not isinstance(result['color'], str)
                or type(result['contributing_frames']) is not int
                or not 1 <= result['contributing_frames'] <= len(selected)
                or type(result['locally_corrected_frames']) is not int
                or not 0 <= result['locally_corrected_frames'] <= len(selected)):
            return None
        alignment = state['alignment']
        if not isinstance(alignment, list) or len(alignment) != len(selected):
            return None
        for row in alignment:
            if not isinstance(row, dict) or set(row) != set(ALIGNMENT_FIELDS):
                return None
            transform = np.asarray(row['full_transform'], np.float64)
            if (transform.shape != (2, 3) or not np.isfinite(transform).all()
                    or abs(np.linalg.det(transform[:, :2])) <= 1e-8
                    or not isinstance(row['full_alignment'], dict)
                    or not math.isfinite(row['full_alignment']['correlation'])
                    or any(not isinstance(row[k], str) or not row[k] for k in ('extracted', 'aligned'))):
                return None
        common = np.load(directory / 'common.npy', allow_pickle=False)
        if common.dtype != np.uint8 or common.shape != (h, w) or not common.any() or (common > 1).any():
            return None
        _image(directory / 'reference.tiff', (h, w, 3), np.uint16)
        _image(directory / 'stack.png', (ch, cw, 3), np.uint16)
        labels = _image(directory / 'source_index.png', (ch, cw), np.uint16)
        if (labels >= len(selected)).any():
            return None
        _image(directory / 'confidence.png', (ch, cw), np.uint16)
        for name in names:
            if name.endswith('.json'):
                _json(directory / name)
    except (OSError, ValueError, TypeError, KeyError, EOFError, cv2.error, OverflowError, RecursionError):
        return None
    for name in names:
        shutil.copyfile(directory / name, Path(output) / name)
    for record, fields in zip(selected, alignment):
        record.update(fields)
    paths = [Path(row['aligned']) for row in selected]
    # Only the reference is read after this boundary. Other paths remain
    # historical provenance and need not survive ordinary frame eviction.
    paths[reference] = directory / 'reference.tiff'
    os.utime(directory / 'complete.json', None)
    return paths, common, result


def save(cache, identifier, output, config, selected, reference, paths, common, result):
    """Atomically publish an immutable snapshot before output composition."""
    if identifier is None:
        return
    directory = _directory(cache, identifier)
    if directory is None:
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    names = _files(config)
    with tempfile.TemporaryDirectory(prefix='pending-', dir=directory.parent) as temporary:
        pending = Path(temporary) / 'entry'
        pending.mkdir()
        for name in names:
            shutil.copyfile(Path(output) / name, pending / name)
        shutil.copyfile(paths[reference], pending / 'reference.tiff')
        np.save(pending / 'common.npy', common, allow_pickle=False)
        state = {'reference': reference, 'shape': list(common.shape), 'result': result,
                 'alignment': [{k: row[k] for k in ALIGNMENT_FIELDS} for row in selected]}
        (pending / 'state.json').write_text(json.dumps(state, sort_keys=True, allow_nan=False))
        snapshot_files = names + ('reference.tiff', 'common.npy', 'state.json')
        manifest = {'schema': SCHEMA, 'identity': identifier,
                    'sha256': {name: _digest(pending / name) for name in snapshot_files}}
        (pending / 'complete.json').write_text(json.dumps(manifest, sort_keys=True, allow_nan=False))
        if directory.exists():
            if directory.is_dir():
                shutil.rmtree(directory)
            else:
                directory.unlink()
        os.replace(pending, directory)
