"""Bounded opt-in experiment, not a production integration.

The integration seam below changes only an all-structural-candidates-rejected
case. Loading every aligned native frame is appropriate for this tiny fixture,
not a production memory strategy. A real integration needs a streaming second
pass and evidence maps keyed by source/frame/geometry.
"""
from __future__ import annotations
import inspect
import cv2
import numpy as np
from macrostitch import spatial
from macrostitch.focus_evidence import coherent_focus_evidence
from macrostitch.imaging import to_linear_bt709, select_coverage


def dense_evidence(images, records, mask):
    """Require two independent same-clip neighbors and broad repeating detail."""
    if len(images)!=len(records) or len(images)<3:
        raise ValueError('At least three identified observations are required')
    identities=[(r['source'],r['frame']) for r in records]
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate captures cannot establish repeatability')
    result=[]; support=[]
    for i, (image, record) in enumerate(zip(images, records)):
        candidates=sorted((j for j,r in enumerate(records) if j!=i and r['source']==record['source']),
                          key=lambda j:abs(records[j]['frame']-record['frame']))[:2]
        if len(candidates)<2:
            result.append(np.zeros(mask.shape,np.float32));support.append(0.);continue
        evidence=coherent_focus_evidence(image,[images[j] for j in candidates],mask=mask,
                                        sigma=8.,minimum_correlation=.5)
        fraction=float(evidence['reliable'][mask].mean())
        result.append(evidence['scores'] if fraction>=.5 else np.zeros(mask.shape,np.float32))
        support.append(fraction)
    scores=np.asarray(result,np.float32)
    if not scores.any():raise ValueError('No broad repeatable dense texture')
    # Static fixed patterns without an observed focus change are insufficient.
    peak=scores.max(axis=0)
    bottom=np.min(np.where(scores>0,scores,np.inf),axis=0)
    varying=mask & ((scores>0).sum(axis=0)>=3) & (peak>1.5*np.maximum(bottom,1e-12)) & (peak>1e-7)
    if varying[mask].mean()<.5:raise ValueError('No broad repeatable focus variation')
    scores[:,~varying]=0
    metadata={'experimental_opt_in':True,'reason':'No structurally eligible frames',
              'method':'Positive signed bandpass correlation across two separate same-clip frames',
              'minimum_correlation':.5,'minimum_supported_fraction':.5,
              'minimum_observed_focus_ratio':1.5,'aggregation_sigma_px':8.,
              'support_fraction_by_frame':support,'noise_floor':'No spatial-median subtraction; correlation-gated explained energy',
              'limitations':['Fixed-pattern and temporally correlated codec noise can imitate real detail.',
                             'Alignment errors can reject real texture; this is not an optical truth test.']}
    return scores,metadata


def select_with_fallback(scores,active,max_frames,target,*,enabled,evidence_provider,required=(),selector=spatial.select_native):
    """Normal path is called verbatim; provider is lazy and opt-in."""
    try:return selector(scores,active,max_frames,target,required=required)
    except ValueError as exc:
        if not enabled or str(exc)!='No frames resolve reliable subject structure':raise
    evidence,metadata=evidence_provider()
    chosen,report=select_coverage(evidence,active,max_frames,target)
    chosen+=sorted(set(required)-set(chosen))
    if len(chosen)>max_frames:
        raise ValueError('Dense fallback prototype cannot fit coverage winners and required anchors')
    # Anchors are recorded observations and count toward both budget and actual
    # measured coverage. This prototype conservatively rejects an overflowing
    # union; it does not silently drop a winning plane to make an anchor fit.
    flat=evidence.reshape(len(evidence),-1)
    valid=active.ravel()>0;peak=flat.max(axis=0)
    valid &= (peak>=float(np.quantile(peak[valid],.15))) & (peak>1e-7)
    raw=flat[chosen][:,valid].max(axis=0)/peak[valid]
    equivalent=np.minimum(raw/.97,1)
    report.update(mean_coverage=float(equivalent.mean()),p10_coverage=float(np.quantile(equivalent,.1)),
                  raw_mean_coverage=float(raw.mean()),target_reached=bool(equivalent.mean()>=target),
                  required_frames=sorted(required))
    report['dense_texture_fallback']=metadata
    return chosen,report


def experimental_fuser():
    """Diagnostic source injection at one proposed hook; package file untouched."""
    source=inspect.getsource(spatial.fuse_spatial)
    # Score storage is now preallocated. Inject at the same semantic boundary,
    # immediately before reliable-pixel gating, rather than a list conversion.
    needle='    reliable=(scores.max(axis=0)>'
    replacement='''    fallback_metadata=None
    if not any(eligibility):
        images=[to_linear_bt709(crop_pixels(p,region)) for p in paths]
        scores,fallback_metadata=dense_evidence(images,records,inside)
        floors=[1e-12]*len(paths)
        eligibility=[bool(s.any()) for s in scores]
    scores=np.asarray(scores,np.float32)
    reliable=(scores.max(axis=0)>'''
    assert source.count(needle)==1
    source=source.replace(needle,replacement)
    needle="    (output/'fusion.json').write_text(json.dumps(info,indent=2))"
    source=source.replace(needle,"    if fallback_metadata is not None:info['dense_texture_fallback']=fallback_metadata\n"+needle)
    namespace=dict(vars(spatial),dense_evidence=dense_evidence)
    exec(compile(source,__file__,'exec'),namespace)
    return namespace['fuse_spatial']
