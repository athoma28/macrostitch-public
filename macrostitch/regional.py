"""Conservative surface fusion: real source pixels, bounded seams, explicit matte.

Regions and silhouette are a reusable framing recipe, not inferred object geometry.
No sharpening, texture synthesis, deconvolution, or resolution enlargement is used.
"""
from __future__ import annotations
import json
import cv2
import numpy as np
import tifffile
from .imaging import to_linear_bt709, to_srgb16, common_crop


def canvas_crop(common, canvas_mode="common"):
    """Output extent in reference coordinates; validity still gates fusion separately."""
    if canvas_mode == "reference":
        h,w=common.shape[:2]
        return 0,0,w,h
    if canvas_mode == "common":
        return common_crop(common)
    raise ValueError("Invalid canvas_mode")


def luma(image):
    return image[...,0]*.0722 + image[...,1]*.7152 + image[...,2]*.2126


def roi_slices(roi, shape):
    h,w=shape[:2]
    x0,y0,x1,y1=np.rint(np.asarray(roi)*[w,h,w,h]).astype(int)
    if x1<=x0 or y1<=y0: raise ValueError("ROI is smaller than a pixel")
    return slice(y0,y1),slice(x0,x1)


def matte(shape, polygon, feather=1.5, erode=1.):
    """Subpixel signed-distance matte, with finite support and inward erosion."""
    h,w=shape[:2]
    points=np.asarray(polygon,dtype=np.float64)*[w,h]
    binary=np.zeros((h,w),np.uint8)
    cv2.fillPoly(binary,[np.rint(points*256).astype(np.int32)],1,shift=8)
    inside=cv2.distanceTransform(binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    outside=cv2.distanceTransform(1-binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    signed=np.where(binary>0,inside-.5,.5-outside)-erode
    t=np.clip(signed/feather+.5,0,1)
    return t*t*(3-2*t)


def band_weights(height, regions, blend_px):
    """Positive partition of unity; blend_px is the half-width of each seam."""
    y=np.arange(height,dtype=np.float32)+.5
    boundaries=[]
    for region in regions[:-1]:
        edge=region['y'][1]*height
        if blend_px:
            t=np.clip((y-edge)/(2*blend_px)+.5,0,1)
            boundaries.append(t*t*(3-2*t))
        else: boundaries.append((y>=edge).astype(np.float32))
    return np.stack([a-b for a,b in zip([np.ones(height,np.float32)]+boundaries,
                                      boundaries+[np.zeros(height,np.float32)])])


def detail_score(linear, valid=None):
    """Mean strongest 5% native-pixel, noise-suppressed bandpass energy."""
    g=luma(linear)
    energy=(cv2.GaussianBlur(g,(0,0),.8)-cv2.GaussianBlur(g,(0,0),2.4))**2
    values=energy[valid] if valid is not None else energy.ravel()
    if values.size<8: raise ValueError("Focus ROI has too few valid subject pixels")
    count=max(1,int(np.ceil(values.size*.05)))
    return float(np.partition(values.ravel(),values.size-count)[-count:].mean())


def checked_write(path, image, options=()):
    if not cv2.imwrite(str(path),image,list(options)): raise RuntimeError(f"Cannot save {path}")


def rgba_preview(rgba):
    # A display-only composite in encoded sRGB. Saved PNG/TIFF retain straight alpha.
    rgb=rgba[...,:3].astype(np.float32)/65535
    a=rgba[...,3:4].astype(np.float32)/65535
    return np.rint((rgb*a+.25*(1-a))*255).astype(np.uint8)


def write_transparent(output, rgb, alpha, result):
    encoded=np.rint(alpha*65535).astype(np.uint16)
    rgb=rgb.copy(); rgb[encoded==0]=0
    rgba=np.dstack([rgb,encoded])
    # OpenCV omits TIFF ExtraSamples for RGBA; explicitly declare straight alpha.
    tifffile.imwrite(output/'stack.tiff',rgba[...,[2,1,0,3]],photometric='rgb',
                     extrasamples='unassalpha',compression='zlib',metadata=None)
    checked_write(output/'stack.png',rgba,[cv2.IMWRITE_PNG_COMPRESSION,3])
    checked_write(output/'matte.png',encoded)
    ys,xs=np.where(encoded>0)
    if not len(xs): raise ValueError("Subject polygon has no visible pixels in the common crop")
    padding=24
    x0=max(0,int(xs.min())-padding); x1=min(rgb.shape[1],int(xs.max())+padding+1)
    y0=max(0,int(ys.min())-padding); y1=min(rgb.shape[0],int(ys.max())+padding+1)
    tight=rgba[y0:y1,x0:x1]
    checked_write(output/'subject.png',tight,[cv2.IMWRITE_PNG_COMPRESSION,3])
    checked_write(output/'preview.jpg',rgba_preview(tight),[cv2.IMWRITE_JPEG_QUALITY,97])
    result.update(alpha='straight / unassociated; fully transparent RGB set to zero',
                  subject_crop_in_stack_xyxy=[x0,y0,x1,y1],subject_width=x1-x0,subject_height=y1-y0)


def make_transparent(output, common, config, result):
    x0,y0,x1,y1=result['crop_xyxy']
    alpha=matte(common.shape,config.subject_polygon,config.matte_feather_px,config.matte_erode_px)
    if config.canvas_mode=="reference": alpha*=common
    write_transparent(output,cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED),
                      alpha[y0:y1,x0:x1],result)
    foreground=alpha[y0:y1,x0:x1]>0
    labels=cv2.imread(str(output/'source_index.png'),cv2.IMREAD_UNCHANGED)
    labels[~foreground]=65535
    checked_write(output/'source_index.png',labels)
    confidence=cv2.imread(str(output/'confidence.png'),cv2.IMREAD_UNCHANGED)
    confidence[~foreground]=0
    checked_write(output/'confidence.png',confidence)


def fuse_regional(paths, reference, common, config, output, emit, records):
    h,w=common.shape
    from .compositing import controls_active, foreground_matte
    controlled=controls_active(config)
    if controlled:
        alpha=foreground_matte((h,w),config)
    elif config.subject_polygon:
        alpha=matte((h,w),config.subject_polygon,config.matte_feather_px,config.matte_erode_px)
    elif config.subject_only:
        from .imaging import roi_mask
        alpha=roi_mask((h,w),config.roi).astype(np.float32)/255
    else: alpha=np.ones((h,w),np.float32)
    alpha*=common
    interior=cv2.erode((alpha>.999).astype(np.uint8),np.ones((5,5),np.uint8))>0
    scores=[]; tone_values=[]; norms=[]
    for i,path in enumerate(paths):
        image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        normalization=1.
        if config.score_normalization_roi:
            patch=to_linear_bt709(image[roi_slices(config.score_normalization_roi,(h,w))])
            normalization=max(float(np.median(luma(patch))),.01)
        row=[]; tone=[]
        for region in config.focus_regions:
            ys,xs=roi_slices(region['roi'],(h,w))
            # Include filter context outside ROI, then score only the requested surface.
            p=8; y0=max(0,ys.start-p); y1=min(h,ys.stop+p); x0=max(0,xs.start-p); x1=min(w,xs.stop+p)
            linear=to_linear_bt709(image[y0:y1,x0:x1])
            valid=np.zeros(linear.shape[:2],bool)
            valid[ys.start-y0:ys.stop-y0,xs.start-x0:xs.stop-x0]=interior[ys,xs]
            row.append(detail_score(linear,valid)/normalization**2)
            tone_roi=region.get('tone_roi')
            tone.append(float(np.median(luma(to_linear_bt709(image[roi_slices(tone_roi,(h,w))])))) if tone_roi else 1.)
        scores.append(row); tone_values.append(tone); norms.append(normalization)
        emit(stage='surface_scores',completed=i+1,total=len(paths),message='Measuring native detail inside each subject surface')
    scores=np.asarray(scores,dtype=np.float64); tones=np.asarray(tone_values)
    winners=np.argmax(scores,axis=0)
    weight=band_weights(h,config.focus_regions,config.region_blend_px)
    ys,xs=np.where(alpha>0)
    if not len(xs): raise ValueError('Empty subject matte')
    bx0,bx1=int(xs.min()),int(xs.max())+1; by0,by1=int(ys.min()),int(ys.max())+1
    images={int(i):to_linear_bt709(cv2.imread(str(paths[i]),cv2.IMREAD_UNCHANGED)[by0:by1,bx0:bx1]) for i in set(winners)}
    composed=np.zeros((by1-by0,bx1-bx0,3),np.float32)
    info=[]; confidences=[]
    for k,(region,win) in enumerate(zip(config.focus_regions,winners)):
        gain=1.; raw_gain=1.
        if region.get('tone_roi') is not None and region.get('tone_reference') is not None:
            anchor=int(winners[region['tone_reference']])
            raw_gain=float(tones[anchor,k]/max(tones[win,k],1e-6))
            gain=float(np.clip(raw_gain,.8,1.25))
        composed+=images[int(win)]*(weight[k,by0:by1,None,None]*gain)
        rank=np.argsort(-scores[:,k],kind='stable')
        confidence=(float(1-scores[rank[1],k]/scores[win,k]) if scores[win,k]>1e-12 else 0.) if len(paths)>1 else 1.
        confidences.append(confidence)
        info.append({'region':k,'y':region['y'],'roi':region['roi'],'source_index':int(win),
                     'source':records[win]['source'],'frame':records[win]['frame'],'timestamp':records[win]['timestamp'],
                     'score':float(scores[win,k]),'runner_up_source_index':int(rank[min(1,len(rank)-1)]),
                     'relative_score_margin':confidence,'linear_tone_gain':gain,'unclamped_tone_gain':raw_gain,
                     'tone_roi':region.get('tone_roi'),'tone_reference':region.get('tone_reference')})
    clipping=float(((composed<0)|(composed>1)).mean())
    x0,y0,x1,y1=canvas_crop(common,config.canvas_mode)
    reference_image=to_srgb16(to_linear_bt709(cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED)))
    merged=reference_image.copy() if controlled or config.background_mode=='reference' else np.zeros((h,w,3),np.uint16)
    if controlled:
        # The shared compositor applies the single final matte in linear light.
        merged[by0:by1,bx0:bx1]=to_srgb16(composed)
    elif config.background_mode=='reference':
        reference_linear=to_linear_bt709(cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED)[by0:by1,bx0:bx1])
        a=alpha[by0:by1,bx0:bx1,None]
        merged[by0:by1,bx0:bx1]=to_srgb16(composed*a+reference_linear*(1-a))
    else: merged[by0:by1,bx0:bx1]=to_srgb16(composed)
    merged=merged[y0:y1,x0:x1]
    a=alpha[y0:y1,x0:x1]
    label_rows=winners[np.argmax(weight,axis=0)].astype(np.uint16)
    labels=np.broadcast_to(label_rows[:,None],(h,w)).copy()[y0:y1,x0:x1]
    labels[a==0]=65535
    checked_write(output/'source_index.png',labels)
    confidence_rows=np.asarray(confidences)[np.argmax(weight,axis=0)]
    confidence=np.broadcast_to(confidence_rows[:,None],(h,w))[y0:y1,x0:x1]*a
    checked_write(output/'confidence.png',np.rint(confidence*65535).astype(np.uint16))
    result={'crop_xyxy':[x0,y0,x1,y1],'width':x1-x0,'height':y1-y0,
            'color':'sRGB transfer / BT.709 primaries; 16-bit integer outputs from 10-bit source',
            'fusion_mode':'regional','contributing_frames':len(set(winners)),
            'diagnostics':{'ambiguous_foreground_fraction':float((confidence[a>.99]<.05).mean()),
                           'preclip_out_of_range_fraction':clipping}}
    if config.background_mode=='transparent' and not controlled:
        write_transparent(output,merged,a,result)
        original=np.dstack([reference_image[y0:y1,x0:x1],np.rint(a*65535).astype(np.uint16)])
        final=cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
        xx0,yy0,xx1,yy1=result['subject_crop_in_stack_xyxy']
        comparison=np.concatenate([rgba_preview(original[yy0:yy1,xx0:xx1]),rgba_preview(final[yy0:yy1,xx0:xx1])],axis=1)
    else:
        checked_write(output/'stack.tiff',merged,[cv2.IMWRITE_TIFF_COMPRESSION,8])
        checked_write(output/'stack.png',merged,[cv2.IMWRITE_PNG_COMPRESSION,3])
        small=cv2.resize(merged,(min(1600,x1-x0),round((y1-y0)*min(1600,x1-x0)/(x1-x0))),interpolation=cv2.INTER_AREA)
        checked_write(output/'preview.jpg',(small/256).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,97])
        cx0,cx1=max(bx0,x0),min(bx1,x1); cy0,cy1=max(by0,y0),min(by1,y1)
        comparison=(np.concatenate([reference_image[cy0:cy1,cx0:cx1],merged[cy0-y0:cy1-y0,cx0-x0:cx1-x0]],axis=1)/256).astype(np.uint8)
    checked_write(output/'comparison.jpg',comparison,[cv2.IMWRITE_JPEG_QUALITY,97])
    fusion={'method':'Coherent source surfaces, convex smoothstep seams; no sharpening or generated detail',
            'score':'Mean strongest 5% of squared Gaussian(0.8)-Gaussian(2.4) linear-luma bandpass in each interior ROI, divided by normalization-patch median squared',
            'source_index_background_sentinel':65535,'score_normalization_roi':config.score_normalization_roi,
            'region_blend_halfwidth_px':config.region_blend_px,'regions':info,
            'scores_by_selected_frame':scores.tolist(),'score_normalization_medians':norms,
            'matte':'User recipe polygon with inward signed-distance erosion and finite smoothstep feather',
            'limitations':['Score margin is a relative ranking, not a calibrated confidence probability.',
                          'Tone matching is an explicit regional brightness adjustment, not an exposure calibration.',
                          'Native optical bloom, chromatic fringes and blur still present in the best recorded frame remain.']}
    (output/'fusion.json').write_text(json.dumps(fusion,indent=2,allow_nan=False))
    emit(stage='fuse',message='Fused coherent recorded surfaces and saved transparent subject',contributing_frames=len(set(winners)))
    return result
