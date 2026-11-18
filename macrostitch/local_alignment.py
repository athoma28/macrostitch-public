"""Opt-in, held-out-validated local translations; no fusion or source selection.

All boxes are integer half-open reference coordinates. ``sample_shift_xy`` maps
anchor/reference pixels to samples in the globally aligned moving image (ECC's
inverse-map convention). It is NOT a forward source-to-reference translation.
Cached crops must share one reference grid. Production rendering samples the
original extracted image, never the already resampled alignment cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TranslationConfig:
    max_shift_px: float = 6.
    min_shift_px: float = .15
    smoothing_sigma: float = 1.2
    min_structure_ratio: float = .12
    min_correlation: float = .85
    min_improvement: float = .005
    feather_px: int = 24
    max_iterations: int = 100


def _box(value):
    if len(value) != 4 or any(int(x) != x for x in value):
        raise ValueError('ROI must contain four integer coordinates')
    x0, y0, x1, y1 = map(int, value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('ROI must have positive area')
    return x0, y0, x1, y1


def _inside(inner, outer):
    return (outer[0] <= inner[0] < inner[2] <= outer[2]
            and outer[1] <= inner[1] < inner[3] <= outer[3])


def _overlaps(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def normalize_local_regions(regions):
    """Validate JSON-compatible recipes and return a stable normalized form.

    Coordinates here are normalized reference-canvas coordinates. Pixel-sized
    measurement guards and the feather inset require the actual image dimensions
    and are checked later by ``estimate_local_translations``.
    """
    if regions is None:
        return None
    if not isinstance(regions, (list, tuple)) or not 1 <= len(regions) <= 8:
        raise ValueError('local_alignment_regions requires 1..8 region dictionaries')
    required = {'roi', 'fit_roi', 'validation_rois', 'anchor'}
    allowed = required | {'max_shift_px', 'feather_px'}

    def normalized_box(value):
        if (not isinstance(value, (list, tuple)) or len(value) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or (isinstance(v, float) and not np.isfinite(v)) for v in value)):
            raise ValueError('Local alignment boxes require four finite numeric coordinates')
        if not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1):
            raise ValueError('Local alignment boxes must be ordered within [0, 1]')
        return tuple(float(v) for v in value)

    normalized = []
    for region in regions:
        if not isinstance(region, dict) or not required <= region.keys() or region.keys() - allowed:
            raise ValueError('Local alignment region has missing or unknown fields')
        roi = normalized_box(region['roi'])
        fit = normalized_box(region['fit_roi'])
        heldout = region['validation_rois']
        if not isinstance(heldout, (list, tuple)) or not heldout:
            raise ValueError('At least one held-out validation ROI is required')
        heldout = tuple(normalized_box(box) for box in heldout)
        if not all(_inside(box, roi) for box in (fit, *heldout)):
            raise ValueError('Fit and validation ROIs must be within the local alignment ROI')
        if any(_overlaps(fit, box) for box in heldout):
            raise ValueError('Validation ROIs must be held out from fitting')
        if any(_overlaps(roi, other['roi']) for other in normalized):
            raise ValueError('Local alignment region ROIs must not overlap')
        anchor = region['anchor']
        if (not isinstance(anchor, dict) or set(anchor) != {'source', 'frame'}
                or any(type(anchor[name]) is not int or anchor[name] < 0 for name in ('source', 'frame'))):
            raise ValueError('Anchor requires exact nonnegative integer source and frame')
        shift = region.get('max_shift_px', 6.)
        if (isinstance(shift, bool) or not isinstance(shift, (int, float))
                or not .15 < shift <= 6 or not np.isfinite(shift)):
            raise ValueError('max_shift_px must be finite and greater than .15, at most 6')
        feather = region.get('feather_px', 24)
        if type(feather) is not int or not 1 <= feather <= 64:
            raise ValueError('feather_px must be an integer within 1..64')
        normalized.append({'roi': roi, 'fit_roi': fit, 'validation_rois': heldout,
                           'anchor': dict(anchor), 'max_shift_px': float(shift), 'feather_px': feather})
    return tuple(normalized)


def _slice(box, origin):
    x0, y0, x1, y1 = box
    return slice(y0-origin[1], y1-origin[1]), slice(x0-origin[0], x1-origin[0])


def _luma(image, encoding):
    x = np.asarray(image)
    if x.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError('Unsupported cached crop dtype')
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Cached crops must contain finite values in [0, 1]')
    if encoding == 'bt709':
        x = np.where(x < .081, x/4.5, ((x+.099)/1.099)**(1/.45))
    elif encoding != 'linear':
        raise ValueError('encoding must be bt709 or linear')
    if x.ndim == 3 and x.shape[2] == 3:
        return (x[..., 0]*.0722 + x[..., 1]*.7152 + x[..., 2]*.2126).astype(np.float32)
    if x.ndim != 2:
        raise ValueError('Crops must be grayscale or BGR')
    return x


def _correlation(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 1e-12 else 0.


def _structure(a):
    # Contrast-normalized band energy is only a rejection gate, not sharpness QA.
    low = cv2.GaussianBlur(a, (0, 0), .8)
    high = cv2.GaussianBlur(a, (0, 0), 2.4)
    return float(np.mean((low-high)**2) / max(float(np.var(a)), 1e-12))


def _array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def estimate_local_translations(
    records: Sequence[dict], cached_crops: Mapping[int, np.ndarray], *,
    roi, fit_roi, validation_rois, anchor, crop_origin=(0, 0),
    candidate_indices=None, labels=None, encoding='bt709', source_hashes=None,
    config: TranslationConfig | None = None,
):
    """Return JSON-safe ``corrections`` and per-candidate ``diagnostics``.

    ``anchor`` is a record index or an explicit ``{'source': ..., 'frame': ...}``.
    The caller chooses a demonstrably sharp anchor; this function does not infer
    physical depth. Fit and held-out ROIs must be disjoint and inside the fully
    corrected core (support ROI inset by feather_px). All held-out regions must
    improve independently. Optional labels share the crop grid and restrict
    consideration to contributors inside the support ROI. ``source_hashes`` maps
    record source identifiers to original-video SHA256 values; cached pixels and
    original full transforms are always fingerprinted even when it is omitted.
    """
    cfg = config or TranslationConfig()
    if not (0 < cfg.max_shift_px <= 6 and 0 <= cfg.min_shift_px < cfg.max_shift_px):
        raise ValueError('Translation bound must be at most six native pixels')
    if (not all(np.isfinite(v) for v in asdict(cfg).values()) or cfg.feather_px < 1
            or cfg.smoothing_sigma <= 0 or cfg.max_iterations < 1):
        raise ValueError('Feather, smoothing and iteration count must be positive')
    if not (0 < cfg.min_structure_ratio <= 1 and 0 < cfg.min_correlation <= 1
            and 0 < cfg.min_improvement < 1):
        raise ValueError('Structure and correlation thresholds must be positive fractions')
    roi, fit_roi = _box(roi), _box(fit_roi)
    validation_rois = [_box(b) for b in validation_rois]
    core = (roi[0]+cfg.feather_px, roi[1]+cfg.feather_px,
            roi[2]-cfg.feather_px, roi[3]-cfg.feather_px)
    if not validation_rois or not all(_inside(b, core) for b in [fit_roi, *validation_rois]):
        raise ValueError('Fit and validation ROIs must be inside the correction core')
    if any(_overlaps(fit_roi, b) for b in validation_rois):
        raise ValueError('Validation ROIs must be held out from fitting')
    if isinstance(anchor, Mapping):
        matches = [i for i, r in enumerate(records)
                   if r.get('source') == anchor.get('source') and r.get('frame') == anchor.get('frame')]
        if len(matches) != 1:
            raise ValueError('Anchor source/frame must identify exactly one record')
        anchor = matches[0]
    anchor = int(anchor)
    indices = list(cached_crops if candidate_indices is None else candidate_indices)
    if (not 0 <= anchor < len(records) or anchor not in cached_crops
            or any(i not in cached_crops or not 0 <= i < len(records) for i in indices)):
        raise ValueError('Missing cached crop or invalid record index')
    if len(set(indices)) != len(indices):
        raise ValueError('Candidate indices must be unique')
    shape = cached_crops[anchor].shape[:2]
    if any(cached_crops[i].shape[:2] != shape for i in indices):
        raise ValueError('Cached crops must share a reference grid')
    extent = (crop_origin[0], crop_origin[1], crop_origin[0]+shape[1], crop_origin[1]+shape[0])
    if not _inside(roi, extent) or (labels is not None and labels.shape != shape):
        raise ValueError('ROI or label grid does not match cached crop extent')
    guard = int(np.ceil(cfg.max_shift_px))+2
    if any(min(b[2]-b[0], b[3]-b[1]) <= 2*guard+4 for b in [fit_roi, *validation_rois]):
        raise ValueError('Measurement ROIs are too small for the shift bound')
    template = _luma(cached_crops[anchor], encoding)
    a = cv2.GaussianBlur(template, (0, 0), cfg.smoothing_sigma)
    fit_slice = _slice(fit_roi, crop_origin)
    af = a[fit_slice]
    structure = _structure(template[fit_slice])
    provenance = {'algorithm': 'heldout-local-translation-v1', 'config': asdict(cfg),
                  'roi': list(roi), 'fit_roi': list(fit_roi), 'validation_rois': validation_rois,
                  'crop_origin': list(crop_origin), 'encoding': encoding, 'anchor_index': anchor,
                  'candidate_indices': indices,
                  'labels_sha256': None if labels is None else _array_hash(labels),
                  'inputs': {}}
    for i in sorted(set(indices + [anchor])):
        r = records[i]
        full = np.asarray(r.get('full_transform'), np.float64)
        if (full.shape != (2, 3) or not np.isfinite(full).all()
                or abs(np.linalg.det(full[:, :2])) < 1e-10):
            raise ValueError('full_transform must be a finite invertible 2x3 source-to-reference matrix')
        provenance['inputs'][str(i)] = {
            'source': r.get('source'), 'frame': r.get('frame'),
            'source_sha256': (source_hashes or {}).get(r.get('source')),
            'cached_crop_sha256': _array_hash(cached_crops[i]),
            'full_transform': full.tolist(),
        }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    corrections, diagnostics = [], []
    for i in indices:
        d = {'source_index': int(i), 'source': records[i].get('source'),
             'frame': records[i].get('frame'), 'accepted': False}
        diagnostics.append(d)
        if i == anchor:
            d['reason'] = 'anchor'; continue
        if labels is not None and not np.any(labels[_slice(roi, crop_origin)] == i):
            d['reason'] = 'no_contributing_footprint'; continue
        moving = _luma(cached_crops[i], encoding)
        ratio = _structure(moving[fit_slice]) / max(structure, 1e-12)
        d['structure_ratio'] = ratio
        if structure < 1e-5 or ratio < cfg.min_structure_ratio:
            d['reason'] = 'insufficient_structure'; continue
        b = cv2.GaussianBlur(moving, (0, 0), cfg.smoothing_sigma)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(af, b[fit_slice], warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                            cfg.max_iterations, 1e-6), None, 3)
        except cv2.error:
            d['reason'] = 'registration_failed'; continue
        shift = warp[:, 2]
        d.update(ecc_correlation=float(cc), sample_shift_xy=shift.tolist())
        distance = float(np.linalg.norm(shift))
        if not np.isfinite(distance) or distance > cfg.max_shift_px:
            d['reason'] = 'shift_exceeds_bound'; continue
        if distance < cfg.min_shift_px:
            d['reason'] = 'no_material_shift'; continue
        aligned = cv2.warpAffine(b, warp, (shape[1], shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT_101)
        measurements = []
        for box in [fit_roi, *validation_rois]:
            inset = (box[0]+guard, box[1]+guard, box[2]-guard, box[3]-guard)
            sl = _slice(inset, crop_origin)
            before, after = _correlation(a[sl], b[sl]), _correlation(a[sl], aligned[sl])
            measurements.append({'roi': list(box), 'before': before, 'after': after,
                                 'improvement': after-before})
        d['measurements'] = measurements
        passes = [m['after'] >= cfg.min_correlation and m['improvement'] >= cfg.min_improvement
                  for m in measurements]
        if not passes[0]:
            d['reason'] = 'fit_not_improved'; continue
        if not all(passes[1:]):
            d['reason'] = 'heldout_not_improved'; continue
        full = np.asarray(records[i]['full_transform'], np.float64)
        corrected = full.copy(); corrected[:, 2] -= shift
        correction = {**d, 'accepted': True, 'reason': 'accepted', 'roi': list(roi),
                      'feather_px': cfg.feather_px, 'original_full_transform': full.tolist(),
                      'core_full_transform': corrected.tolist(), 'provenance_sha256': fingerprint}
        d.update(accepted=True, reason='accepted')
        corrections.append(correction)
    return {'corrections': corrections, 'diagnostics': diagnostics,
            'provenance': provenance, 'provenance_sha256': fingerprint}


def resample_local_correction(original, correction, *, interpolation=cv2.INTER_LANCZOS4):
    """Return only the support-ROI tile, sampled once from original source pixels.

    At the outer pixel boundary the local displacement is exactly zero. The
    caller replaces only this tile in its global alignment; every pixel outside
    the ROI therefore remains untouched. Encoded values/dtype are preserved.
    """
    if not correction.get('accepted'):
        raise ValueError('Only accepted corrections may be rendered')
    x0, y0, x1, y1 = _box(correction['roi'])
    shift = np.asarray(correction['sample_shift_xy'], np.float64)
    if shift.shape != (2,) or not np.isfinite(shift).all() or np.linalg.norm(shift) > 6:
        raise ValueError('Invalid bounded local shift')
    feather = correction['feather_px']
    if feather <= 0:
        raise ValueError('A positive boundary feather is required')
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    distance = np.minimum.reduce([xx-x0, x1-1-xx, yy-y0, y1-1-yy])
    t = np.clip(distance / feather, 0, 1)
    weight = t*t*(3-2*t)
    forward = np.eye(3); forward[:2] = correction['original_full_transform']
    inverse = np.linalg.inv(forward)
    rx, ry = xx+weight*shift[0], yy+weight*shift[1]
    sx = inverse[0, 0]*rx + inverse[0, 1]*ry + inverse[0, 2]
    sy = inverse[1, 0]*rx + inverse[1, 1]*ry + inverse[1, 2]
    return cv2.remap(original, sx.astype(np.float32), sy.astype(np.float32),
                     interpolation, borderMode=cv2.BORDER_REFLECT_101)
