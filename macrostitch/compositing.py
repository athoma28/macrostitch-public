"""Explicit object selection and source-based, straight-alpha compositing.

Coordinates always describe the uncropped reference canvas. The legacy renderers
are unchanged unless controls are supplied. Object polygons are user annotations,
not inferred segmentation; exclusions reveal transparency, never invented fill.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import cv2
import numpy as np
import tifffile

from .imaging import roi_mask, to_linear_bt709, to_srgb16


def controls_active(config):
    return (config.background_opacity is not None or config.background_margin_px != 0
            or bool(config.object_masks))


def resolved_opacity(config):
    if config.background_opacity is not None:
        return config.background_opacity
    return 0. if config.background_mode == 'transparent' else 1.


def _polygon(value):
    try:
        points = tuple(tuple(float(v) for v in p) for p in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Object polygon requires normalized x,y vertices') from exc
    if not 3 <= len(points) <= 128 or any(
            len(p) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
            for p in points):
        raise ValueError('Object polygon requires 3..128 finite normalized x,y vertices')
    if len(set(points)) != len(points):
        raise ValueError('Object polygon vertices must be distinct; do not repeat the closing vertex')
    edges = list(zip(points, points[1:] + points[:1]))
    area = abs(sum(a[0]*b[1] - a[1]*b[0] for a,b in edges)) / 2
    if area < 1e-10:
        raise ValueError('Object polygon must have positive area')

    def orientation(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def touches(a,b,p):
        return (abs(orientation(a,b,p)) < 1e-12 and
                min(a[0],b[0])-1e-12 <= p[0] <= max(a[0],b[0])+1e-12 and
                min(a[1],b[1])-1e-12 <= p[1] <= max(a[1],b[1])+1e-12)

    for i,(a,b) in enumerate(edges):
        for j in range(i+1,len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            c,d = edges[j]
            crossing = (orientation(a,b,c)*orientation(a,b,d) < 0 and
                        orientation(c,d,a)*orientation(c,d,b) < 0)
            if crossing or any((touches(a,b,c),touches(a,b,d),touches(c,d,a),touches(c,d,b))):
                raise ValueError('Object polygon must be simple, without crossing or touching edges')
    return points


def normalize_controls(config):
    for name, upper in (('background_opacity',1),('background_margin_px',512)):
        value = getattr(config,name)
        if name == 'background_opacity' and value is None:
            continue
        try:
            if isinstance(value,bool):
                raise ValueError()
            value = float(value)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'Invalid {name}') from exc
        if not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'Invalid {name}')
        setattr(config,name,value)
    if config.object_masks is None:
        return
    if not isinstance(config.object_masks,(list,tuple)) or not 1 <= len(config.object_masks) <= 32:
        raise ValueError('object_masks requires 1..32 named polygons')
    names = set()
    rows = []
    for row in config.object_masks:
        if not isinstance(row,dict) or set(row) != {'name','action','polygon'}:
            raise ValueError('Object mask requires only name, action, polygon')
        name = row['name']
        if not isinstance(name,str) or not 1 <= len(name.strip()) <= 80 or name.strip() in names:
            raise ValueError('Object mask names must be unique, with 1..80 characters')
        name = name.strip()
        names.add(name)
        if row['action'] not in ('stack','reference','exclude'):
            raise ValueError('Object mask action must be stack, reference, or exclude')
        rows.append(dict(name=name,action=row['action'],polygon=_polygon(row['polygon'])))
    config.object_masks = tuple(rows)


def stack_polygons(config):
    return [row['polygon'] for row in config.object_masks or () if row['action'] == 'stack']


def foreground_matte(shape, config, *, feather=None, erode=None):
    """Union of explicit subjects; new stack objects replace only the inferred ROI."""
    from .regional import matte
    polygons = stack_polygons(config)
    if config.subject_polygon:
        polygons = [config.subject_polygon] + polygons
    if not polygons:
        if not config.subject_only:
            return np.ones(shape[:2],np.float32)
        # Match the existing roi_mask default, but allow finite feather/context.
        x0,y0,x1,y1 = config.roi or (.1,.1,.9,.85)
        polygons = [((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
    alpha = np.zeros(shape[:2],np.float32)
    for polygon in polygons:
        np.maximum(alpha,matte(shape,polygon,
                              config.matte_feather_px if feather is None else feather,
                              config.matte_erode_px if erode is None else erode),out=alpha)
    return alpha


def selection_mask(shape, config):
    """Retain legacy proxy evidence unless extra stack objects were requested."""
    if stack_polygons(config):
        return np.where(foreground_matte(shape,config,feather=.1,erode=0) > .5,255,0).astype(np.uint8)
    return roi_mask(shape,config.roi)


def _action_matte(shape, config, action):
    from .regional import matte
    alpha = np.zeros(shape[:2],np.float32)
    for row in config.object_masks or ():
        if row['action'] == action:
            np.maximum(alpha,matte(shape,row['polygon'],config.matte_feather_px,0),out=alpha)
    return alpha


def _decode_srgb(image):
    encoded = image.astype(np.float32)/65535
    return np.where(encoded <= .04045,encoded/12.92,((encoded+.055)/1.055)**2.4)


@lru_cache(maxsize=1)
def _srgb_uint16_lut():
    table = _decode_srgb(np.arange(65536, dtype=np.uint16))
    table.flags.writeable = False
    return table


def _srgb_linear(image):
    if image.dtype == np.uint16:
        result = _srgb_uint16_lut()[image]
        return np.asarray(result) if image.ndim == 0 else result
    return _decode_srgb(image)


def compose_output(output, paths, reference, common, config, result):
    """Enforce final pixel/visibility overrides after multiscale reconstruction."""
    if not controls_active(config):
        return
    from .regional import checked_write, rgba_preview
    output = Path(output)
    x0,y0,x1,y1 = result['crop_xyxy']
    crop = (slice(y0,y1),slice(x0,x1))
    raw_reference = cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED)
    reference_linear = to_linear_bt709(raw_reference)
    encoded_reference = to_srgb16(reference_linear)
    checked_write(output/'reference.png',encoded_reference,[cv2.IMWRITE_PNG_COMPRESSION,3])
    reference_linear = reference_linear[crop]
    encoded_reference = encoded_reference[crop]
    fused = cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)[...,:3]
    if fused.shape != encoded_reference.shape:
        raise ValueError('Fused output extent does not match reference canvas crop')
    foreground = foreground_matte(common.shape,config)[crop]
    protected = _action_matte(common.shape,config,'reference')[crop]
    excluded = _action_matte(common.shape,config,'exclude')[crop]
    opacity = resolved_opacity(config)
    if config.background_margin_px:
        context = foreground_matte(common.shape,config,
                    erode=config.matte_erode_px-config.background_margin_px)[crop]
        background = opacity + (1-opacity)*context
    else:
        context = foreground
        background = opacity
    # Common validity gates fused detail only. The original reference remains
    # available for context and explicit protection even at invalid warp borders.
    fused_linear = _srgb_linear(fused)
    fused_linear[common[crop] == 0] = reference_linear[common[crop] == 0]
    alpha = foreground + (1-foreground)*background
    premultiplied = (fused_linear*foreground[...,None] +
                     reference_linear*((1-foreground)*background)[...,None])
    alpha = protected + (1-protected)*alpha
    premultiplied = reference_linear*protected[...,None] + premultiplied*(1-protected[...,None])
    rgb = to_srgb16(premultiplied/np.maximum(alpha[...,None],1e-12))
    # Preserve bit-exact reference and source colors in unmixed interiors.
    reference_pixels = (foreground == 0) | (protected == 1) | (common[crop] == 0)
    rgb[reference_pixels] = encoded_reference[reference_pixels]
    source_pixels = (foreground == 1) & (protected == 0) & (common[crop] > 0)
    rgb[source_pixels] = fused[source_pixels]
    alpha *= 1-excluded
    encoded_alpha = np.rint(np.clip(alpha,0,1)*65535).astype(np.uint16)
    invisible = encoded_alpha == 0
    rgb[invisible] = 0
    rgba = np.dstack([rgb,encoded_alpha])
    transparent = opacity < 1 or any(row['action'] == 'exclude' for row in config.object_masks or ())
    if transparent:
        tifffile.imwrite(output/'stack.tiff',rgba[...,[2,1,0,3]],photometric='rgb',
                         extrasamples='unassalpha',compression='zlib',metadata=None)
        checked_write(output/'stack.png',rgba,[cv2.IMWRITE_PNG_COMPRESSION,3])
    else:
        checked_write(output/'stack.tiff',rgb,[cv2.IMWRITE_TIFF_COMPRESSION,8])
        checked_write(output/'stack.png',rgb,[cv2.IMWRITE_PNG_COMPRESSION,3])
    checked_write(output/'matte.png',encoded_alpha)
    # Inspection crop follows selected objects/context, not faint full-canvas alpha.
    ys,xs = np.where(((context > 0) | (protected > 0)) & ~invisible)
    if len(xs):
        bx0=max(0,int(xs.min())-24);bx1=min(rgb.shape[1],int(xs.max())+25)
        by0=max(0,int(ys.min())-24);by1=min(rgb.shape[0],int(ys.max())+25)
    else:
        bx0,by0,bx1,by1 = 0,0,rgb.shape[1],rgb.shape[0]
    checked_write(output/'subject.png',rgba[by0:by1,bx0:bx1],[cv2.IMWRITE_PNG_COMPRESSION,3])
    preview = rgba_preview(rgba)
    scale = min(1600,preview.shape[1])/preview.shape[1]
    preview = cv2.resize(preview,(round(preview.shape[1]*scale),max(1,round(preview.shape[0]*scale))),interpolation=cv2.INTER_AREA)
    checked_write(output/'preview.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,97])
    labels = cv2.imread(str(output/'source_index.png'),cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(str(output/'confidence.png'),cv2.IMREAD_UNCHANGED)
    # Labels identify dominant finest-scale color, even for partial visibility.
    fused_weight = foreground*(1-protected)
    reference_weight = (1-foreground)*background*(1-protected) + protected
    reference_dominant = (reference_weight >= fused_weight) | (common[crop] == 0)
    labels[reference_dominant] = reference
    labels[invisible] = 65535
    confidence = np.rint(confidence.astype(np.float32)*foreground*(1-protected)*(1-excluded)).astype(np.uint16)
    confidence[reference_dominant | invisible] = 0
    checked_write(output/'source_index.png',labels)
    checked_write(output/'confidence.png',confidence)
    # Review artifacts show the actual final visibility, including excluded objects.
    before = np.dstack([encoded_reference,encoded_alpha])
    comparison = np.concatenate([rgba_preview(before[by0:by1,bx0:bx1]),
                                 rgba_preview(rgba[by0:by1,bx0:bx1])],axis=1)
    checked_write(output/'comparison.jpg',comparison,[cv2.IMWRITE_JPEG_QUALITY,97])
    if config.fusion_mode == 'spatial':
        fx0,fy0,fx1,fy1 = result['fusion_roi_xyxy']
        sx0,sy0=max(0,fx0-x0),max(0,fy0-y0)
        sx1,sy1=min(x1-x0,fx1-x0),min(y1-y0,fy1-y0)
        checked_write(output/'subject-detail.png',rgba[sy0:sy1,sx0:sx1],[cv2.IMWRITE_PNG_COMPRESSION,3])
    info = {'coordinate_system':'normalized uncropped reference canvas',
            'reference_dimensions_wh':[common.shape[1],common.shape[0]],
            'crop_xyxy':result['crop_xyxy'],'background_opacity':opacity,
            'background_margin_px':config.background_margin_px,
            'object_masks':config.object_masks or [],
            'overlap_priority':['exclude','reference','stack'],
            'selection':'Stack polygons join the subject for candidate coverage; reference/exclude override final output.',
            'alpha':'straight / unassociated; fully transparent RGB set to zero',
            'background':'Original reference colors. Margin retains context without expanding focus fusion.',
            'reference_source_index':reference,'invisible_source_index':65535,
            'fully_transparent_pixels':int(invisible.sum()),
            'partially_transparent_pixels':int(((encoded_alpha>0)&(encoded_alpha<65535)).sum()),
            'all_transparent':bool(invisible.all()),
            'limitations':['User polygons are not automatic object segmentation.',
                           'Excluded objects become transparent; no hidden background is reconstructed.',
                           'Boundary colors can retain optical blur or color spill already present in the source.']}
    (output/'compositing.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    result.update(compositing=info,subject_crop_in_stack_xyxy=[bx0,by0,bx1,by1],
                  subject_width=bx1-bx0,subject_height=by1-by0)
    if transparent:
        result['alpha'] = info['alpha']
