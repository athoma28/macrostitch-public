"""Native focus evidence with coherent two-dimensional source assignment."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from .imaging import to_linear_bt709, to_srgb16, laplacian_pyramid, reconstruct, roi_mask
from .regional import canvas_crop, matte, checked_write, write_transparent

ENERGY_AGGREGATION_SIGMA = 20.
# The broad footprint already suppresses isolated noise. Subtracting several
# medians here erased real low-contrast plastic detail beside strong metal edges.
FUSION_FLOOR_MULTIPLIER = 1.
FUSION_RELIABILITY_MULTIPLIER = 1.
PALETTE_EQUIVALENCE = .90
PALETTE_MEAN_COVERAGE = .995
PALETTE_P01_COVERAGE = .90
COHERENCE_MAX_SWEEPS = 60


def crop_pixels(path, region):
    """Reuse verified crops keyed by source content and exact pixel bounds."""
    import hashlib
    import os
    import tempfile
    from .cache_io import image_record, _read_record, _signature, _hash

    path=Path(path);ys,xs=region
    if any(part.step not in (None,1) or not isinstance(part.start,int) or
           not isinstance(part.stop,int) or not 0<=part.start<part.stop for part in (ys,xs)):
        raise ValueError('Crop requires positive ordered integer bounds and unit steps')
    prior=_read_record(path)
    decoded=None
    if prior is None:
        decoded=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.dtype!=np.uint16 or decoded.ndim!=3 or decoded.shape[2]!=3:
            raise ValueError(f'Invalid source cache image: {path}')
        shape=decoded.shape
    else:shape=prior['shape']
    source=image_record(path,shape)
    if source is None:raise ValueError(f'Source cache image failed integrity validation: {path}')
    # Shape probing precedes validation; never retain pixels read before identity verification.
    decoded=None
    if ys.stop>shape[0] or xs.stop>shape[1]:raise ValueError('Crop exceeds source image dimensions')
    expected=(ys.stop-ys.start,xs.stop-xs.start,3)
    identity={'source_sha256':source['sha256'],'source_name':path.name,
              'bounds':[xs.start,ys.start,xs.stop,ys.stop]}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cached=path.parent/f'spatial-crop-{key}.npy'
    metadata=Path(str(cached)+'.verified.json')
    try:
        record=json.loads(metadata.read_text())
        stat=_signature(cached)
        if (record['identity']==identity and record['shape']==list(expected)
                and record['dtype']=='uint16' and
                (record['stat']==stat or record['sha256']==_hash(cached))):
            image=np.load(cached,mmap_mode='r',allow_pickle=False)
            if image.shape==expected and image.dtype==np.uint16:
                return image
    except (OSError,ValueError,TypeError,KeyError,EOFError):
        pass
    # Unstamped/legacy arrays are rebuilt once rather than trusting plausible shape.
    if decoded is None:decoded=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    current=image_record(path,shape)
    if decoded is None or current is None or current['sha256']!=source['sha256']:
        raise ValueError('Source cache image changed during crop creation')
    image=decoded[region].copy()
    descriptor,temporary=tempfile.mkstemp(prefix='.spatial-crop-',suffix='.npy',dir=path.parent)
    os.close(descriptor)
    meta_temporary=None
    try:
        np.save(temporary,image,allow_pickle=False)
        os.replace(temporary,cached)
        stat=_signature(cached);checksum=_hash(cached)
        if stat!=_signature(cached):raise RuntimeError('Crop changed while recording cache identity')
        record={'identity':identity,'shape':list(expected),'dtype':'uint16','stat':stat,'sha256':checksum}
        descriptor,meta_temporary=tempfile.mkstemp(prefix='.spatial-crop-',suffix='.json',dir=path.parent)
        with os.fdopen(descriptor,'w') as stream:json.dump(record,stream,sort_keys=True)
        os.replace(meta_temporary,metadata)
    finally:
        Path(temporary).unlink(missing_ok=True)
        if meta_temporary is not None:Path(meta_temporary).unlink(missing_ok=True)
    return image


def select_native(scores, active_mask, max_frames, target, required=()):
    """Reject globally defocused/noisy frames, then cover native structural cells.

    A frame's median cell energy estimates the low-detail floor. This is a
    conservative structure gate, not a calibrated sensor noise model. Coverage
    excludes weak/textureless cells rather than claiming their focus is known.
    """
    mandatory=set(required)
    if len(mandatory)>max_frames or any(type(i) is not int or not 0<=i<len(scores) for i in mandatory):
        raise ValueError('Mandatory native frames exceed budget or reference invalid records')
    flat=np.asarray(scores).reshape(len(scores),-1)
    active=active_mask.ravel()>0
    if not active.any(): raise ValueError('No native subject cells')
    floor=np.maximum(np.median(flat[:,active],axis=1),1e-12)
    contrast=np.quantile(flat[:,active],.99,axis=1)/floor
    eligible=contrast>=8.
    if not eligible.any(): raise ValueError('No frames resolve reliable subject structure')
    corrected=np.maximum(flat-3*floor[:,None],0)*eligible[:,None]
    peak=corrected.max(axis=0)
    active &= peak>4*float(np.median(floor))
    if not active.any(): raise ValueError('No reliable native focus cells')
    raw=corrected[:,active]/peak[active]
    values=np.minimum(raw/.995,1)
    covered=np.zeros(values.shape[1],np.float32); chosen=[]; gains=[]
    for _ in range(min(max_frames,len(scores))):
        if len(set(chosen)|mandatory)>=max_frames:break
        gain=np.maximum(values-covered,0).mean(axis=1);gain[chosen]=-1
        best=int(np.argmax(gain))
        if gain[best]<1e-5: break
        chosen.append(best);gains.append(float(gain[best]));covered=np.maximum(covered,values[best])
        if float(covered.mean())>=target: break
    chosen+=sorted(mandatory-set(chosen))
    if not chosen:raise ValueError('No native frames fit the selection budget')
    covered=values[chosen].max(axis=0)
    return chosen,{'mean_coverage':float(covered.mean()),'raw_mean_coverage':float(raw[chosen].max(axis=0).mean()),
                   'p10_coverage':float(np.quantile(covered,.1)),'target_reached':bool(covered.mean()>=target),
                   'equivalent_focus_tolerance':.005,'textured_cells':int(active.sum()),'marginal_gains':gains,
                   'all_frames_examined':len(scores),'structurally_eligible_frames':int(eligible.sum()),
                   'scope':'Native subject structure only; low-evidence textureless cells excluded',
                   'structure_gate':'99th percentile / median native cell energy >= 8; subtract 3x median floor',
                   'minimum_structure_ratio':8.}


def native_energy(linear):
    g=linear[...,0]*.0722+linear[...,1]*.7152+linear[...,2]*.2126
    band=cv2.GaussianBlur(g,(0,0),.8)-cv2.GaussianBlur(g,(0,0),2.4)
    # Integrate the whole native edge footprint, so displaced defocus fringes
    # cannot independently win at each side of the same physical edge.
    return cv2.GaussianBlur(band*band,(0,0),ENERGY_AGGREGATION_SIGMA),g


def _source_palette(relative,reliable):
    """Cover near-best spatial evidence with a small set of actual source frames."""
    values=np.minimum(relative[:,reliable]/PALETTE_EQUIVALENCE,1)
    covered=np.zeros(values.shape[1],np.float32)
    chosen=[]
    for _ in range(len(values)):
        gains=np.maximum(values-covered,0).mean(axis=1);gains[chosen]=-1
        candidate=int(np.argmax(gains));chosen.append(candidate)
        covered=np.maximum(covered,values[candidate])
        if covered.mean()>=PALETTE_MEAN_COVERAGE and np.quantile(covered,.01)>=PALETTE_P01_COVERAGE:
            break
    return chosen,covered


def _coherent_labels_with_diagnostics(scores,reliable,smoothness):
    if not reliable.any(): raise ValueError('No reliable native fusion pixels')
    peak=scores.max(axis=0)
    relative=scores/np.maximum(peak,1e-12)
    chosen,covered=_source_palette(relative,reliable)
    palette_relative=relative[chosen]
    # Near-equivalent sources incur equal data cost, allowing spatial coherence
    # to choose one. No frame-index averaging or chronological depth assumption.
    unary=np.maximum(PALETTE_EQUIVALENCE-palette_relative,0)/PALETTE_EQUIVALENCE*reliable
    labels=np.argmax(palette_relative,axis=0).astype(np.int32)
    _,nearest=distance_transform_edt(~reliable,return_indices=True)
    labels[~reliable]=labels[tuple(nearest[:,~reliable])]
    h,w=labels.shape;yy,xx=np.indices((h,w))
    # A brightness edge is not evidence that its two sides need different frames.
    # Constant coupling avoids permitting source changes inside a blurred edge.
    weights=[np.ones(labels.shape,np.float32) for _ in range(4)]
    weights[0][0]=0;weights[1][-1]=0;weights[2][:,0]=0;weights[3][:,-1]=0
    for iteration in range(COHERENCE_MAX_SWEEPS):
        changed=0
        for parity in (0,1):
            neighbors=[np.roll(labels,1,0),np.roll(labels,-1,0),np.roll(labels,1,1),np.roll(labels,-1,1)]
            costs=unary.copy()
            for candidate in range(len(chosen)):
                for neighbor,weight in zip(neighbors,weights):costs[candidate]+=smoothness*weight*(candidate!=neighbor)
            current=costs[labels,yy,xx];best=np.argmin(costs,axis=0)
            change=(costs[best,yy,xx]<current-1e-7)&(((yy+xx)%2)==parity)
            changed+=int(change.sum());labels[change]=best[change]
        if not changed:break
    labels=np.asarray(chosen,dtype=np.int32)[labels]
    retained=relative[labels,yy,xx][reliable]
    info={'source_palette':chosen,'equivalent_score_threshold':PALETTE_EQUIVALENCE,
          'palette_mean_coverage_target':PALETTE_MEAN_COVERAGE,'palette_p01_coverage_target':PALETTE_P01_COVERAGE,
          'palette_mean_coverage':float(covered.mean()),'palette_p01_coverage':float(np.quantile(covered,.01)),
          'mean_assigned_relative_score':float(retained.mean()),'p10_assigned_relative_score':float(np.quantile(retained,.1)),
          'constant_neighbor_penalty':smoothness,'max_sweeps':COHERENCE_MAX_SWEEPS,'completed_sweeps':iteration+1,
          'optimizer':'All-palette red/black ICM with strict energy improvements; no global optimality guarantee',
          'score_interpretation':'Relative measured focus energy, not optical sharpness percentage or probability'}
    return labels,relative,info


def coherent_labels(scores, guide, reliable, smoothness=.12):
    """Consolidate equivalent focal choices while retaining strong local evidence.

    guide remains accepted for API compatibility; brightness-edge weighting is
    intentionally omitted because it fragmented the source of physical edges.
    """
    labels,relative,_=_coherent_labels_with_diagnostics(scores,reliable,smoothness)
    return labels,relative


def prepare_local_alignment(paths, records, labels, region, image_shape, recipes):
    """Estimate bounded corrections after source selection, retaining its evidence."""
    from .local_alignment import estimate_local_translations, TranslationConfig
    h,w=image_shape; y0,x0=region[0].start,region[1].start
    used=sorted(int(i) for i in np.unique(labels)); corrections={}; reports=[]
    def box(value):return tuple(round(v*(w if j%2==0 else h)) for j,v in enumerate(value))
    source_hashes={r['source']:r.get('source_sha256') for r in records}
    for recipe in recipes:
        anchors=[i for i,r in enumerate(records) if all(r[k]==v for k,v in recipe['anchor'].items())]
        if len(anchors)!=1:raise ValueError('Local alignment anchor must identify one selected frame')
        crops={i:crop_pixels(paths[i],region) for i in sorted(set(used+[anchors[0]]))}
        report=estimate_local_translations(records,crops,roi=box(recipe['roi']),
            fit_roi=box(recipe['fit_roi']),validation_rois=[box(b) for b in recipe['validation_rois']],
            anchor=anchors[0],crop_origin=(x0,y0),candidate_indices=used,labels=labels,
            source_hashes=source_hashes,config=TranslationConfig(max_shift_px=recipe['max_shift_px'],feather_px=recipe['feather_px']))
        reports.append(report)
        for correction in report['corrections']:
            corrections.setdefault(correction['source_index'],[]).append(correction)
    return corrections,{'method':'Held-out validated local translation, sampled once from extracted originals',
        'focus_evidence':'Source labels and confidence are measured before optional local correction.',
        'boundary':'Existing globally aligned outermost pixel ring is retained exactly.',
        'corrected_frames':len(corrections),'regions':reports}


def corrected_crop(raw, record, corrections, region):
    from .local_alignment import resample_local_correction
    original=cv2.imread(str(record['extracted']),cv2.IMREAD_UNCHANGED)
    if original is None or original.dtype!=np.uint16 or original.ndim!=3 or original.shape[2]!=3:
        raise ValueError('Missing or invalid original frame for local correction')
    result=raw.copy(); oy,ox=region[0].start,region[1].start
    for correction in corrections:
        x0,y0,x1,y1=correction['roi'];target=(slice(y0-oy,y1-oy),slice(x0-ox,x1-ox))
        tile=resample_local_correction(original,correction)
        old=raw[target]
        # remap and warpAffine round coordinates differently even at zero shift.
        # Preserve the existing single-resample boundary rather than introducing a seam.
        tile[0]=old[0];tile[-1]=old[-1];tile[:,0]=old[:,0];tile[:,-1]=old[:,-1]
        result[target]=tile
    return result


def fuse_spatial(paths, reference, common, config, output, emit, records):
    h,w=common.shape
    from .compositing import controls_active, foreground_matte, stack_polygons
    controlled=controls_active(config)
    if stack_polygons(config):
        core=foreground_matte((h,w),config,erode=0)
        hard=cv2.dilate((core>0).astype(np.uint8),np.ones((17,17),np.uint8))
        support=cv2.GaussianBlur(hard.astype(np.float32),(0,0),3)
        interior=cv2.erode(hard,np.ones((25,25),np.uint8))>0
    elif config.subject_polygon:
        # A photograph retains its edge blur; this is a blending support, not a cutout.
        hard=matte((h,w),config.subject_polygon,config.matte_feather_px,0)>0
        hard=cv2.dilate(hard.astype(np.uint8),np.ones((17,17),np.uint8))
        support=cv2.GaussianBlur(hard.astype(np.float32),(0,0),3)
        interior=cv2.erode(hard,np.ones((25,25),np.uint8))>0
    elif config.subject_only:
        hard=roi_mask((h,w),config.roi)>0;interior=hard
        support=cv2.GaussianBlur(hard.astype(np.float32),(0,0),3)
    else:
        support=np.ones((h,w),np.float32);interior=common>0
    support*=common
    ys,xs=np.where(support>1e-6)
    if not len(xs): raise ValueError('Empty spatial fusion support')
    # Extra context protects pyramid reconstruction from the crop boundary.
    x0=max(0,int(xs.min())-64);x1=min(w,int(xs.max())+65)
    y0=max(0,int(ys.min())-64);y1=min(h,int(ys.max())+65)
    region=(slice(y0,y1),slice(x0,x1));inside=interior[region]&common[region].astype(bool)
    if not inside.any(): raise ValueError('No valid subject interior')
    # Each native score plane is needed after label selection for confidence.
    # Allocate its final storage once rather than retaining both a list of
    # planes and a second stacked copy at the end of this loop.
    scores=np.empty((len(paths),y1-y0,x1-x0),np.float32);floors=[];eligibility=[]
    for i,path in enumerate(paths):
        image=to_linear_bt709(crop_pixels(path,region))
        energy,g=native_energy(image)
        floor=max(float(np.median(energy[inside])),1e-12)
        eligible=float(np.quantile(energy[inside],.99))/floor>=8
        scores[i]=np.maximum(energy-FUSION_FLOOR_MULTIPLIER*floor,0)*eligible;floors.append(floor);eligibility.append(eligible)
        emit(stage='spatial_scores',completed=i+1,total=len(paths),message='Measuring native detail and rejecting low-structure noise')
    reliable=(scores.max(axis=0)>FUSION_RELIABILITY_MULTIPLIER*float(np.median(floors))) & (support[region]>.01)
    # Native evidence is aggregated into 8px neighborhoods for coherent physical
    # features. Image pixels are never downsampled for the final merge.
    sh=(max(1,(x1-x0)//8),max(1,(y1-y0)//8))
    small=np.asarray([cv2.resize(s,sh,interpolation=cv2.INTER_AREA) for s in scores])
    reliable_small=(small.max(axis=0)>FUSION_RELIABILITY_MULTIPLIER*float(np.median(floors))) & (cv2.resize(support[region],sh,interpolation=cv2.INTER_AREA)>.5)
    labels,_,coherence=_coherent_labels_with_diagnostics(small,reliable_small,config.spatial_smoothness)
    labels=cv2.resize(labels.astype(np.uint16),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    labels[support[region]<1e-6]=reference
    # Stable labels at native edges, without numerical averaging of source-frame identities.
    peak=scores.max(axis=0);yy,xx=np.indices(labels.shape)
    assigned=scores[labels,yy,xx]
    confidence=np.zeros_like(peak);confidence[reliable]=np.clip(assigned[reliable]/np.maximum(peak[reliable],1e-12),0,1)
    del small,scores
    levels=max(1,min(config.pyramid_levels,int(np.log2(min(labels.shape)))-2))
    sums=None;weight_sums=None
    used=sorted(int(i) for i in np.unique(labels))
    corrections={};local_info=None
    if config.local_alignment_regions:
        corrections,local_info=prepare_local_alignment(paths,records,labels,region,(h,w),config.local_alignment_regions)
        (output/'local-alignment.json').write_text(json.dumps(local_info,indent=2))
        emit(stage='local_alignment',message='Validated local corrections against independent held-out regions',corrected_frames=len(corrections))
    for i in used:
        raw=crop_pixels(paths[i],region)
        if i in corrections:raw=corrected_crop(raw,records[i],corrections[i],region)
        image=to_linear_bt709(raw)
        weight=(labels==i).astype(np.float32)
        # Native finest bands use one source wherever possible. Broader bands
        # blend progressively to avoid exposure/color steps between recorded frames.
        if config.spatial_blend_px>0:weight=cv2.GaussianBlur(weight,(0,0),config.spatial_blend_px)
        bands=laplacian_pyramid(image,levels)
        if sums is None:
            sums=[np.zeros_like(b) for b in bands];weight_sums=[np.zeros(b.shape[:2],np.float32) for b in bands]
        for level,band in enumerate(bands):
            if level:weight=cv2.pyrDown(weight)
            sums[level]+=band*weight[...,None];weight_sums[level]+=weight
    fused=reconstruct([s/np.maximum(t[...,None],1e-12) for s,t in zip(sums,weight_sums)])
    reference_linear=to_linear_bt709(cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED))
    full=reference_linear.copy();a=support[region][...,None]
    full[region]=fused if controlled else fused*a+reference_linear[region]*(1-a)
    full[common==0]=reference_linear[common==0]
    crop=canvas_crop(common,config.canvas_mode);cx0,cy0,cx1,cy1=crop
    merged=to_srgb16(full[cy0:cy1,cx0:cx1])
    result={'crop_xyxy':list(crop),'width':cx1-cx0,'height':cy1-cy0,'fusion_mode':'spatial',
            'color':'sRGB transfer / BT.709 primaries; 16-bit integer from 10-bit video',
            'contributing_frames':len(used),'fusion_roi_xyxy':[x0,y0,x1,y1],
            'locally_corrected_frames':len(corrections),
            'diagnostics':{'reliable_subject_fraction':float(reliable[support[region]>.5].mean()),
                           'preclip_out_of_range_fraction':float(((full<0)|(full>1)).mean())}}
    if config.background_mode=='transparent' and not controlled:
        alpha=matte((h,w),config.subject_polygon,config.matte_feather_px,config.matte_erode_px)*common
        write_transparent(output,merged,alpha[cy0:cy1,cx0:cx1],result)
    else:
        checked_write(output/'stack.tiff',merged,[cv2.IMWRITE_TIFF_COMPRESSION,8])
        checked_write(output/'stack.png',merged,[cv2.IMWRITE_PNG_COMPRESSION,3])
        preview=cv2.resize(merged,(min(1600,cx1-cx0),round((cy1-cy0)*min(1600,cx1-cx0)/(cx1-cx0))),interpolation=cv2.INTER_AREA)
        checked_write(output/'preview.jpg',(preview/256).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,97])
    full_labels=np.full((h,w),reference,np.uint16);full_labels[region]=np.where(support[region]>.5,labels,reference)
    if config.background_mode=='transparent' and not controlled: full_labels[alpha==0]=65535
    checked_write(output/'source_index.png',full_labels[cy0:cy1,cx0:cx1])
    full_confidence=np.zeros((h,w),np.uint16);full_confidence[region]=np.rint(confidence*support[region]*65535).astype(np.uint16)
    checked_write(output/'confidence.png',full_confidence[cy0:cy1,cx0:cx1])
    comparison=np.concatenate([to_srgb16(reference_linear[region]),to_srgb16(full[region])],axis=1)
    checked_write(output/'comparison.jpg',(comparison/256).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,97])
    checked_write(output/'subject-detail.png',to_srgb16(full[region]),[cv2.IMWRITE_PNG_COMPRESSION,3])
    info={'method':'Native edge-footprint focus evidence, consolidated source palette, coherent 2D labels, multiscale seam blending',
          'contributing_sources':[{'source_index':i,'source':records[i]['source'],'frame':records[i]['frame'],
                                  'timestamp':records[i]['timestamp'],'dominant_pixels':int(((labels==i)&(support[region]>.5)).sum())} for i in used],
          'noise_floor_by_selected_frame':floors,'structurally_eligible':eligibility,
          'fusion_floor_multiplier':FUSION_FLOOR_MULTIPLIER,
          'fusion_reliability_multiplier':FUSION_RELIABILITY_MULTIPLIER,
          'spatial_smoothness':config.spatial_smoothness,'finest_seam_sigma_px':config.spatial_blend_px,
          'source_label_support_native_px':8,
          'native_energy_aggregation_sigma_px':ENERGY_AGGREGATION_SIGMA,'label_coherence':coherence,
          'pyramid_levels':levels,'reference_background_index':reference,
          'local_alignment':None if local_info is None else {'artifact':'local-alignment.json','corrected_frames':len(corrections),'focus_evidence':local_info['focus_evidence']},
          'confidence_meaning':'Assigned native structure energy / best observed energy; zero where evidence is weak. Not a probability.',
          'limitations':['Weakly textured surfaces inherit neighboring source labels.',
                         'Multiscale blending may mix sources near boundaries; labels name the dominant finest-scale source.',
                         'No sharpening, super-resolution or synthesized texture. Optical blur and parallax may remain.']}
    (output/'fusion.json').write_text(json.dumps(info,indent=2))
    emit(stage='fuse',message='Saved full photograph with spatial focus stack',contributing_frames=len(used))
    return result
