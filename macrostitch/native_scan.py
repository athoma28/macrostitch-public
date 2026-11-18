"""Stream every native video frame into local focus evidence, never image stacks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .imaging import scale_transform
from .cache_io import _hash, _signature

NATIVE_VERSION = "1.0.0"
CELL_PIXELS = 16
CONTEXT_PIXELS = 32
INTEGRITY_FILES = ('raw_scores.npy', 'scores.npy', 'active_mask.npy', 'records.json')


def _save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _interpolate(records, count):
    """Return proxy-coordinate forward matrices and their exact sample brackets."""
    if count < 1 or any(not isinstance(row.get("frame"), (int, np.integer))
                        or isinstance(row["frame"], (bool, np.bool_))
                        or not 0 <= row["frame"] < count for row in records):
        raise ValueError("Proxy frame indices must be integers within the decoded clip")
    records = sorted(records, key=lambda row: row["frame"])
    keys = np.asarray([row["frame"] for row in records], np.int64)
    if not len(keys) or np.any(np.diff(keys) <= 0):
        raise ValueError("Native scan requires unique ordered proxy frames per clip")
    matrices = np.asarray([row["transform"] for row in records], np.float64)
    if matrices.shape != (len(keys), 2, 3) or not np.isfinite(matrices).all():
        raise ValueError("Invalid proxy transforms")
    frames = np.arange(count)
    right = np.searchsorted(keys, frames, side="left")
    lo = np.maximum(right - 1, 0)
    hi = np.minimum(right, len(keys) - 1)
    amount = np.divide(frames - keys[lo], keys[hi] - keys[lo],
                       out=np.zeros(count, np.float64), where=keys[hi] != keys[lo])
    interpolated = matrices[lo] * (1 - amount[:, None, None]) + matrices[hi] * amount[:, None, None]
    nearest = np.where(np.abs(frames - keys[lo]) <= np.abs(keys[hi] - frames), lo, hi)
    return records, interpolated, lo, hi, nearest


def _source_crop(transforms, target, shape):
    """Bound inverse footprints for EVERY interpolated frame, with filter context."""
    h, w = shape[:2]
    x0, y0, x1, y1 = target
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], np.float64)
    bounds = []
    for matrix in transforms:
        if abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            raise ValueError("Singular interpolated proxy transform")
        bounds.append(corners @ cv2.invertAffineTransform(matrix).T)
    points = np.concatenate(bounds)
    low = np.floor((points.min(axis=0) - CONTEXT_PIXELS) / 2).astype(int) * 2
    high = np.ceil((points.max(axis=0) + CONTEXT_PIXELS) / 2).astype(int) * 2
    low = np.maximum(low, 0)
    high = np.minimum(high, [w - w % 2, h - h % 2])
    if np.any(high <= low):
        raise ValueError("Subject ROI has no source support")
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def _crop_transform(full, source_origin, target_origin):
    local = full.copy()
    local[:, 2] += full[:, :2] @ np.asarray(source_origin) - np.asarray(target_origin)
    return local


def _timestamps(source, records):
    # Existing pipeline metadata contains all packet PTS sorted into display order.
    # Its stated provenance is retained instead of calling frame/fps exact time.
    metadata_path = Path(records[0]["proxy"]).parent / "complete.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            times = np.asarray(metadata.get("timestamps", []), dtype=np.float64)
            if (source.get("frames") and times.shape == (source["frames"],)
                    and metadata.get("timestamp_basis") == "sorted packet PTS"
                    and np.isfinite(times).all() and np.all(np.diff(times) >= 0)):
                return times.tolist(), "sorted packet PTS (existing proxy metadata)"
        except (OSError, ValueError, TypeError, AttributeError):
            # A damaged optional proxy cache must not substitute invented times
            # or prevent probing the original clip's decoded timestamps.
            pass
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", source["path"]], text=True)
    frames = json.loads(result).get("frames", [])
    try:
        times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Native scan requires a timestamp for every decoded frame") from exc
    if not times or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Invalid native frame timestamps")
    if source.get("frames") and len(times) != source["frames"]:
        raise ValueError("Source frame count differs from decoded timestamp count")
    return times, "decoded-frame best_effort_timestamp_time"


def _valid_scores(array):
    # Bounded temporary memory even for thousands of native focus maps.
    for start in range(0, len(array), 64):
        block = array[start:start + 64]
        if not np.isfinite(block).all() or np.any(block < 0):
            return False
    return True


def _seal_integrity(directory, identity, metadata, *, legacy=False):
    files = {}
    for name in INTEGRITY_FILES:
        path = directory / name
        before = _signature(path)
        checksum = _hash(path)
        if before != _signature(path):
            raise OSError('Native cache changed while recording integrity')
        files[name] = {'sha256': checksum, 'stat': before}
    record = {'schema_version': 1, 'identity': identity, 'files': files,
              'origin': 'legacy structural validation' if legacy else 'fresh native decode',
              'legacy_adoption_limitation': ('Checksums establish integrity only from this adoption onward; '
                  'valid-looking pre-adoption corruption cannot be detected.' if legacy else None)}
    _save_json(directory / 'native-integrity.json', record)
    metadata['integrity_schema'] = 1
    _save_json(directory / 'complete.json', metadata)


def _integrity_valid(directory, identity, metadata):
    """Adopt structurally valid legacy data once; never re-adopt recorded data."""
    sidecar = directory / 'native-integrity.json'
    if not sidecar.exists():
        if 'integrity_schema' in metadata:
            return False
        _seal_integrity(directory, identity, metadata, legacy=True)
        return True
    record = json.loads(sidecar.read_text())
    if (metadata.get('integrity_schema', 1) != 1 or record['schema_version'] != 1
            or record['identity'] != identity or set(record['files']) != set(INTEGRITY_FILES)):
        return False
    changed = False
    for name in INTEGRITY_FILES:
        previous = record['files'][name]
        checksum = previous['sha256']
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in '0123456789abcdef' for c in checksum)):
            return False
        path = directory / name
        before = _signature(path)
        if previous['stat'] != before:
            if _hash(path) != checksum or before != _signature(path):
                return False
            previous['stat'] = before
            changed = True
    if changed:
        _save_json(sidecar, record)
    if 'integrity_schema' not in metadata:
        metadata['integrity_schema'] = 1
        _save_json(directory / 'complete.json', metadata)
    return True


def _cached(directory, identity, reference_record):
    try:
        metadata = json.loads((directory / "complete.json").read_text())
        if metadata["identity"] != identity or not metadata["all_frames_examined"]:
            return None
        records = json.loads((directory / "records.json").read_text())
        shape = tuple(metadata["score_shape"])
        counts = metadata["frame_counts"]
        if (len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape)
                or not counts or any(type(value) is not int or value <= 0 for value in counts)):
            return None
        raw = np.load(directory / "raw_scores.npy", mmap_mode="r")
        scores = np.load(directory / "scores.npy", mmap_mode="r")
        active = np.load(directory / "active_mask.npy")
        if (raw.shape != shape or scores.shape != shape or raw.dtype != np.float32
                or scores.dtype != np.float32 or active.shape != shape[1:]
                or active.dtype != np.uint8 or not active.any() or not np.isin(active, [0, 255]).all()
                or len(records) != shape[0] or sum(counts) != len(records)):
            return None
        reference = metadata["reference_index"]
        if (type(reference) is not int or not 0 <= reference < len(records)
                or (records[reference]["source"], records[reference]["frame"]) !=
                (reference_record["source"], reference_record["frame"])):
            return None
        offset = 0
        for clip, count in enumerate(counts):
            if any(row["source"] != clip or row["frame"] != frame
                   for frame, row in enumerate(records[offset:offset + count])):
                return None
            clip_records = records[offset:offset + count]
            times = np.asarray([row["timestamp"] for row in clip_records], dtype=np.float64)
            transforms = np.asarray([row["transform"] for row in clip_records], dtype=np.float64)
            if (times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) < 0)
                    or transforms.shape != (count, 2, 3) or not np.isfinite(transforms).all()):
                return None
            offset += count
        if not _valid_scores(raw) or not _valid_scores(scores):
            return None
        if not _integrity_valid(directory, identity, metadata):
            return None
        return records, scores, reference, active, directory
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def native_candidates(sources, proxy_records, proxy_reference, cache, config, emit):
    """Return all-frame records and temporally smoothed LOCAL native score grids.

    Geometry is interpolated proxy geometry, not a full-resolution registration.
    The score grid's reference coordinates are in complete.json/native_crop_xyxy.
    The caller must hold the pipeline cache lock, as for the existing stages.
    """
    if not sources or not proxy_records:
        raise ValueError("Native scan requires sources and proxy registration")
    h, w = int(sources[0]["height"]), int(sources[0]["width"])
    if any((source["height"], source["width"]) != (h, w) for source in sources):
        raise ValueError("Native scan source dimensions must match")
    reference_record = proxy_records[proxy_reference]
    identity_data = {"version": NATIVE_VERSION, "sources": [
        {key: source.get(key) for key in ("sha256", "width", "height", "frames", "fps")}
        for source in sources], "proxies": proxy_records,
        "reference": [reference_record["source"], reference_record["frame"]],
        "roi": config.roi, "polygon": config.subject_polygon}
    from .compositing import stack_polygons, foreground_matte
    added_objects = stack_polygons(config)
    if added_objects:
        # Presentation-only opacity/margins/protection retain the native scan cache.
        identity_data['stack_polygons'] = added_objects
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    directory = Path(cache) / "native-analysis" / identity[:24]
    directory.mkdir(parents=True, exist_ok=True)
    cached = _cached(directory, identity, reference_record)
    if cached is not None:
        emit(stage="native_scan", message="Verified cached all-frame native focus maps", completed=len(cached[0]), total=len(cached[0]))
        return cached
    (directory / "complete.json").unlink(missing_ok=True)

    polygon = config.subject_polygon
    if added_objects:
        active_full = np.where(foreground_matte((h,w),config,feather=.1,erode=0)>.5,255,0).astype(np.uint8)
        ay,ax = np.where(active_full)
        if not len(ax): raise ValueError('Stack objects have no visible focus pixels')
        low,high = np.array([ax.min(),ay.min()]),np.array([ax.max()+1,ay.max()+1])
    elif polygon:
        points = np.asarray(polygon, np.float64) * [w, h]
        low, high = points.min(axis=0), points.max(axis=0)
    else:
        roi = config.roi or (.1, .1, .9, .85)
        low, high = np.asarray(roi[:2]) * [w, h], np.asarray(roi[2:]) * [w, h]
    tx0, ty0 = np.maximum(np.floor(low).astype(int) - CONTEXT_PIXELS, 0)
    tx1, ty1 = np.minimum(np.ceil(high).astype(int) + CONTEXT_PIXELS, [w, h])
    target = [int(tx0), int(ty0), int(tx1), int(ty1)]
    tw, th = int(tx1 - tx0), int(ty1 - ty0)
    if min(tw, th) < 1:
        raise ValueError("Native scan ROI is empty")
    grid = (math.ceil(tw / CELL_PIXELS), math.ceil(th / CELL_PIXELS))
    active = np.zeros((th, tw), np.uint8)
    if added_objects:
        active = active_full[ty0:ty1,tx0:tx1]
        del active_full
    elif polygon:
        local_points = np.rint((points - [tx0, ty0]) * 256).astype(np.int32)
        cv2.fillPoly(active, [local_points], 255, shift=8)
    else:
        x0, y0 = np.rint(low - [tx0, ty0]).astype(int)
        x1, y1 = np.rint(high - [tx0, ty0]).astype(int)
        active[y0:y1, x0:x1] = 255
    active = np.where(cv2.resize(active, grid, interpolation=cv2.INTER_AREA) >= 128, 255, 0).astype(np.uint8)
    if not active.any():
        raise ValueError("Native scan ROI has no active focus cells")

    plans = []
    for clip, source in enumerate(sources):
        rows = [row for row in proxy_records if row["source"] == clip]
        if not rows:
            raise ValueError("Missing clip proxy registration")
        times, basis = _timestamps(source, rows)
        rows, matrices, lo, hi, nearest = _interpolate(rows, len(times))
        proxy_image = cv2.imread(rows[0]["proxy"], cv2.IMREAD_GRAYSCALE)
        if proxy_image is None:
            raise ValueError("Cannot read native-scan proxy geometry")
        full = np.asarray([scale_transform(matrix, proxy_image.shape, (h, w)) for matrix in matrices])
        crop = _source_crop(full, target, (h, w))
        plans.append((rows, matrices, full, lo, hi, nearest, times, basis, crop))
    count = sum(len(plan[6]) for plan in plans)
    shape = (count, grid[1], grid[0])
    raw = np.lib.format.open_memmap(directory / "raw_scores.npy", mode="w+", dtype=np.float32, shape=shape)
    smooth = np.lib.format.open_memmap(directory / "scores.npy", mode="w+", dtype=np.float32, shape=shape)
    np.save(directory / "active_mask.npy", active)
    lut = np.arange(65536, dtype=np.float32) / 65535
    lut = np.where(lut < .081, lut / 4.5, ((lut + .099) / 1.099) ** (1 / .45))
    records, offset = [], 0
    for clip, (source, plan) in enumerate(zip(sources, plans)):
        rows, matrices, full, lo, hi, nearest, times, basis, crop = plan
        sx0, sy0, sx1, sy1 = crop
        sw, sh = sx1 - sx0, sy1 - sy0
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-threads", str(config.threads),
                   "-i", source["path"], "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
                   f"crop={sw}:{sh}:{sx0}:{sy0}", "-fps_mode", "passthrough", "-pix_fmt", "gray16le",
                   "-f", "rawvideo", "pipe:1"]
        valid_source = np.ones((sh, sw), np.uint8)
        frame = 0
        emit(stage="native_scan", source=clip, completed=offset, total=count,
             message="Scanning every frame at native resolution")
        with (directory / f"decode-{clip}.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, bufsize=sw * sh * 4)
            try:
                while True:
                    data = process.stdout.read(sw * sh * 2)
                    if not data:
                        break
                    if len(data) != sw * sh * 2:
                        raise RuntimeError(f"Incomplete native frame in source {clip}, frame {frame}")
                    if frame >= len(times):
                        raise RuntimeError("Native decoder returned more frames than timestamp metadata")
                    image = np.frombuffer(data, dtype="<u2").reshape(sh, sw)
                    warp = _crop_transform(full[frame], (sx0, sy0), (tx0, ty0))
                    linear = cv2.warpAffine(lut[image], warp, (tw, th), flags=cv2.INTER_LANCZOS4)
                    valid = cv2.warpAffine(valid_source, warp, (tw, th), flags=cv2.INTER_NEAREST)
                    valid = cv2.erode(valid, np.ones((49, 49), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    energy = (cv2.GaussianBlur(linear, (0, 0), .8) - cv2.GaussianBlur(linear, (0, 0), 2.4)) ** 2
                    energy = cv2.GaussianBlur(energy, (0, 0), 3) * valid
                    raw[offset + frame] = cv2.resize(energy, grid, interpolation=cv2.INTER_AREA)
                    near = rows[nearest[frame]]
                    if not near.get("accepted", True):
                        raw[offset + frame] = 0
                    records.append({"source": clip, "frame": frame, "timestamp": times[frame],
                        "timestamp_basis": basis, "proxy": near["proxy"], "transform": matrices[frame].tolist(),
                        "mean_linear_gray": float(linear[valid > 0].mean()) if valid.any() else 0.,
                        "alignment": near.get("alignment", {}), "accepted": near.get("accepted", True),
                        "geometry": "proxy matrix coefficients interpolated; not native-refined",
                        "proxy_frames_interpolated": [rows[lo[frame]]["frame"], rows[hi[frame]]["frame"]]})
                    frame += 1
                    if frame % 60 == 0:
                        emit(stage="native_scan", source=clip, completed=offset + frame, total=count,
                             message="Measuring native spatial detail in every frame")
                process.stdout.close()
                returncode = process.wait()
                if returncode:
                    raise RuntimeError(f"Native ffmpeg failed with exit {returncode}; see {directory / f'decode-{clip}.log'}")
                if frame != len(times):
                    raise RuntimeError(f"Native frame count mismatch: decoded {frame}, expected {len(times)}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
        # Temporal filtering stabilizes evidence only. Output pixels are never averaged.
        gaussian_filter1d(raw[offset:offset + frame], 2, axis=0, mode="nearest", output=smooth[offset:offset + frame])
        for local in range(frame):
            if not records[offset + local]["accepted"]:
                smooth[offset + local] = 0
        offset += frame
    raw.flush()
    smooth.flush()
    # Closing writable maps can update their final filesystem signatures. Seal
    # only after that, and expose read-only scores to downstream consumers.
    del raw, smooth
    smooth = np.load(directory / 'scores.npy', mmap_mode='r')
    reference = next((i for i, row in enumerate(records) if
                      (row["source"], row["frame"]) == (reference_record["source"], reference_record["frame"])), None)
    if reference is None:
        raise ValueError("Exact reference frame absent from native scan")
    _save_json(directory / "records.json", records)
    metadata = {"version": NATIVE_VERSION, "identity": identity, "all_frames_examined": True,
        "frame_counts": [len(plan[6]) for plan in plans], "score_shape": list(shape),
        "native_crop_xyxy": target, "grid_shape_yx": list(shape[1:]),
        "cell_size_reference_xy": [tw / grid[0], th / grid[1]], "source_crops_xyxy": [plan[8] for plan in plans],
        "reference_index": reference, "score_formula": "Decoded native gray16le, inverse BT.709 transfer (luma approximation), square Gaussian(.8)-Gaussian(2.4), Gaussian aggregate3, INTER_AREA to ~16px cells; no exposure normalization",
        "temporal_smoothing": "Gaussian sigma2 frames separately per clip; scores only, not image pixels",
        "geometry": "All six proxy forward matrix coefficients interpolated by frame; endpoint-clamped; nearest proxy acceptance and alignment inherited; rejected scores zero",
        "limitations": ["Interpolated proxy geometry is not native refinement.",
                        "Gray transfer inversion approximates linear luma; color channels are not scored independently.",
                        "Focus energy can rank noise, reflections, blur halos, or background; it is not calibrated optical sharpness."]}
    _seal_integrity(directory, identity, metadata)
    emit(stage="native_scan", completed=count, total=count, message="Saved complete native focus maps for every frame")
    return records, smooth, reference, active, directory
