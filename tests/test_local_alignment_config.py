"""Shared recipe validation before expensive decoding or pixel-sized fitting."""
from copy import deepcopy
import json
import pytest
from macrostitch.local_alignment import normalize_local_regions


def recipe():
    return {'roi':[0,0,1,1], 'fit_roi':[.1,.1,.4,.9],
            'validation_rois':[[.6,.1,.9,.4],[.6,.6,.9,.9]],
            'anchor':{'source':0,'frame':1320}}


def test_json_lists_and_python_tuples_have_identical_canonical_recipes_without_mutation():
    source=recipe();before=deepcopy(source)
    normalized=normalize_local_regions([source])
    assert normalized==normalize_local_regions(normalized)
    assert normalized==normalize_local_regions(json.loads(json.dumps(normalized)))
    assert source==before
    assert isinstance(normalized,tuple) and isinstance(normalized[0]['validation_rois'],tuple)
    assert all(type(v) is float for v in normalized[0]['roi'])
    assert normalized[0]['max_shift_px']==6. and normalized[0]['feather_px']==24
    source['anchor']['frame']=99
    assert normalized[0]['anchor']['frame']==1320
    assert normalize_local_regions(None) is None


@pytest.mark.parametrize('regions',[[],{},'bad',[None],[recipe()]*9])
def test_collection_must_be_one_to_eight_explicit_regions(regions):
    with pytest.raises(ValueError):normalize_local_regions(regions)


@pytest.mark.parametrize('key,value',[
    ('roi',[0,0,0,1]),('roi',[-.1,0,1,1]),('roi',[0,0,1.1,1]),
    ('roi',[0,0,float('nan'),1]),('roi',[0,0,float('inf'),1]),('roi',[0,0,10**400,1]),
    ('roi',[False,0,1,1]),('roi',['0',0,1,1]),('roi',None),
    ('roi',[0,0,1]),('fit_roi',[.8,.1,1,.9]),
    ('validation_rois',[]),('validation_rois',None),
    ('validation_rois',[[.2,.2,.3,.3]]),
    ('anchor',{'source':False,'frame':1}),('anchor',{'source':0.,'frame':1}),
    ('anchor',{'source':0,'frame':-1}),('anchor',{'source':0,'frame':1,'extra':2}),
    ('anchor',{'source':0}),('anchor',None),
    ('max_shift_px',.15),('max_shift_px',6.01),('max_shift_px',float('nan')),('max_shift_px',10**400),
    ('max_shift_px',True),('feather_px',0),('feather_px',65),('feather_px',24.),('feather_px',True),
])
def test_malformed_or_unsafe_measurement_recipe_is_rejected(key,value):
    row=recipe();row[key]=value
    with pytest.raises(ValueError):normalize_local_regions([row])


def test_unknown_and_missing_fields_are_rejected():
    row=recipe();row['estimate_anchor']=True
    with pytest.raises(ValueError):normalize_local_regions([row])
    row=recipe();del row['anchor']
    with pytest.raises(ValueError):normalize_local_regions([row])


def test_measurement_boxes_must_be_inside_support_but_pixel_feather_waits_for_dimensions():
    row=recipe();row['roi']=[.05,.05,.95,.95]
    row['validation_rois']=[[.6,.1,1,.4]]
    with pytest.raises(ValueError,match='within'):normalize_local_regions([row])
    # Touching the normalized support boundary is legal here; the estimator must
    # decide whether the actual pixel dimensions leave the required feather core.
    row=recipe();row['fit_roi']=[0,0,.4,1]
    assert normalize_local_regions([row])[0]['fit_roi']==(0.,0.,.4,1.)


def test_independent_regions_can_touch_but_cannot_correct_the_same_pixels():
    left=recipe();left['roi']=[0,0,.5,1];left['fit_roi']=[.05,.1,.2,.9]
    left['validation_rois']=[[.3,.1,.45,.9]]
    right=recipe();right['roi']=[.5,0,1,1];right['fit_roi']=[.55,.1,.7,.9]
    right['validation_rois']=[[.8,.1,.95,.9]]
    right.update(max_shift_px=.16,feather_px=64)
    assert len(normalize_local_regions([left,right]))==2
    right['roi'][0]=.49
    with pytest.raises(ValueError,match='overlap'):normalize_local_regions([left,right])


def test_fit_and_heldout_may_touch_edges_but_not_share_area():
    row=recipe();row['validation_rois']=[[.4,.1,.8,.9]]
    assert normalize_local_regions([row])
    row['validation_rois'][0][0]=.399
    with pytest.raises(ValueError,match='held out'):normalize_local_regions([row])
