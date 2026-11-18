"""Independent integration checks for source sampling, boundaries and budgets."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from macrostitch import spatial, pipeline, native_scan
from macrostitch.local_alignment import normalize_local_regions
from macrostitch.cache_io import atomic_image_write


def encoded(linear):
    values=np.where(linear<.018,linear*4.5,1.099*np.maximum(linear,0)**.45-.099)
    return np.rint(np.clip(values,0,1)*65535).astype(np.uint16)


def local_fixture(tmp_path):
    rng=np.random.default_rng(841)
    gray=cv2.GaussianBlur(rng.random((320,384),dtype=np.float32),(0,0),1.)*.6+.15
    moving=cv2.warpAffine(gray,np.array([[1,0,2.3],[0,1,-3.1]],np.float32),(384,320),
                          flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    images=[np.repeat(encoded(g)[...,None],3,axis=2) for g in (gray,moving)]
    paths=[];records=[]
    for index,image in enumerate(images):
        path=tmp_path/f'frame{index}.tiff';atomic_image_write(path,image);paths.append(path)
        records.append({'source':0,'frame':index,'timestamp':index/30,'extracted':str(path),
                        'full_transform':np.eye(2,3).tolist(),'source_sha256':'source-video-digest'})
    def unit(box):return [v/(384 if i%2==0 else 320) for i,v in enumerate(box)]
    recipes=normalize_local_regions([{'roi':unit([32,32,352,288]),'fit_roi':unit([64,64,160,250]),
        'validation_rois':[unit([200,64,320,130]),unit([200,170,320,250])],
        'anchor':{'source':0,'frame':0},'feather_px':16}])
    return paths,records,recipes,images


def test_corrected_crop_preserves_exact_exterior_and_samples_original_once(tmp_path):
    rng=np.random.default_rng(882)
    original=rng.integers(5000,55000,(220,260,3),dtype=np.uint16)
    path=tmp_path/'original.tiff';atomic_image_write(path,original)
    full=np.array([[1.02,-.015,3.7],[.015,1.02,-2.4]],np.float64)
    globally_aligned=cv2.warpAffine(original,full,(260,220),flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT_101)
    region=(slice(10,210),slice(20,250));raw=globally_aligned[region].copy();before=raw.copy()
    correction={'accepted':True,'roi':[50,40,210,180],'sample_shift_xy':[2.3,-1.1],
                'feather_px':12,'original_full_transform':full.tolist()}
    result=spatial.corrected_crop(raw,{'extracted':str(path)},[correction],region)
    outside=np.ones(raw.shape[:2],bool);outside[30:170,30:190]=False
    np.testing.assert_array_equal(result[outside],raw[outside])
    np.testing.assert_array_equal(result[30,30:190],raw[30,30:190])
    np.testing.assert_array_equal(result[169,30:190],raw[169,30:190])
    np.testing.assert_array_equal(result[30:170,30],raw[30:170,30])
    np.testing.assert_array_equal(result[30:170,189],raw[30:170,189])
    np.testing.assert_array_equal(raw,before)
    # Independent inverse geometry at fully corrected core pixels. This must
    # match one interpolation of the extracted original, not a second warp of
    # the global alignment image, which loses measurable source detail.
    yy,xx=np.mgrid[60:160,70:190].astype(np.float64)
    inverse=cv2.invertAffineTransform(full);rx=xx+2.3;ry=yy-1.1
    expected=cv2.remap(original,(inverse[0,0]*rx+inverse[0,1]*ry+inverse[0,2]).astype(np.float32),
                       (inverse[1,0]*rx+inverse[1,1]*ry+inverse[1,2]).astype(np.float32),
                       cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
    np.testing.assert_array_equal(result[50:150,50:170],expected)
    twice=cv2.remap(globally_aligned,rx.astype(np.float32),ry.astype(np.float32),cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT_101)
    assert np.mean(abs(twice.astype(float)-expected.astype(float)))>100


def test_prepare_uses_explicit_anchor_outside_label_palette_and_records_provenance(tmp_path):
    paths,records,recipes,_=local_fixture(tmp_path)
    region=(slice(16,304),slice(16,368))
    labels=np.ones((288,352),np.uint16)
    corrections,info=spatial.prepare_local_alignment(paths,records,labels,region,(320,384),recipes)
    assert set(corrections)=={1} and info['corrected_frames']==1
    correction,=corrections[1]
    np.testing.assert_allclose(correction['sample_shift_xy'],[2.3,-3.1],atol=.1)
    proof=info['regions'][0]['provenance']
    assert proof['anchor_index']==0 and proof['candidate_indices']==[1]
    assert proof['crop_origin']==[16,16]
    assert proof['inputs']['0']['source_sha256']=='source-video-digest'
    assert proof['inputs']['1']['source_sha256']=='source-video-digest'


@pytest.mark.parametrize('levels',[1,6])
def test_fused_local_correction_has_bounded_pyramid_influence_and_keeps_focus_provenance(tmp_path,monkeypatch,levels):
    paths,records,recipes,images=local_fixture(tmp_path)
    # Leave enough exterior beyond the full down/up-pyramid filter footprint to
    # make the final-image support assertion non-vacuous, even with six levels.
    padding=128;h,w=320+2*padding,384+2*padding
    for path,image in zip(paths,images):
        atomic_image_write(path,cv2.copyMakeBorder(image,padding,padding,padding,padding,cv2.BORDER_REFLECT_101))
    row=dict(recipes[0])
    def translated(box):
        return tuple((v*(384 if i%2==0 else 320)+padding)/(w if i%2==0 else h) for i,v in enumerate(box))
    row['roi']=translated(row['roi']);row['fit_roi']=translated(row['fit_roi'])
    row['validation_rois']=tuple(translated(box) for box in row['validation_rois'])
    recipes=normalize_local_regions([row])
    count=[0]
    def controlled_focus(image):
        # Two unambiguous recorded source regions create a genuine multiscale
        # source boundary. The integration under test begins after focus scoring.
        index=count[0]%2;count[0]+=1
        energy=np.full(image.shape[:2],1e-6,np.float32)
        if index==0:energy[198:378,378:428]=1
        else:energy[198:378,198:248]=1
        return energy,image[...,0]
    monkeypatch.setattr(spatial,'native_energy',controlled_focus)
    common=np.ones((h,w),np.uint8)
    folders=[tmp_path/'baseline',tmp_path/'corrected']
    for folder,local in zip(folders,[None,recipes]):
        folder.mkdir()
        config=pipeline.Config(fusion_mode='spatial',selection_mode='native',subject_only=False,
            canvas_mode='reference',pyramid_levels=levels,spatial_blend_px=1,local_alignment_regions=local)
        config.validate()
        spatial.fuse_spatial(paths,0,common,config,folder,lambda **event:None,records)
    proof=json.loads((folders[1]/'local-alignment.json').read_text())
    assert proof['corrected_frames']==1
    baseline=cv2.imread(str(folders[0]/'stack.png'),cv2.IMREAD_UNCHANGED)
    corrected=cv2.imread(str(folders[1]/'stack.png'),cv2.IMREAD_UNCHANGED)
    # A five-tap pyrDown/pyrUp pair expands influence by at most this many
    # native pixels. The exact ROI boundary guarantee applies before fusion;
    # the final multiscale image is allowed to change within this finite margin.
    margin=4*(2**(levels-1)-1)
    outside=np.ones(common.shape,bool)
    outside[160-margin:416+margin,160-margin:480+margin]=False
    assert outside.sum()>1000
    np.testing.assert_array_equal(corrected[outside],baseline[outside])
    assert np.any(corrected[208:368,208:278]!=baseline[208:368,208:278])
    for name in ('source_index.png','confidence.png'):
        np.testing.assert_array_equal(cv2.imread(str(folders[0]/name),-1),cv2.imread(str(folders[1]/name),-1))


def test_mandatory_anchor_does_not_waste_remaining_native_frame_budget(tmp_path,monkeypatch):
    source=tmp_path/'source.mov';source.write_bytes(b'fixture')
    metadata={'path':str(source),'sha256':pipeline.digest(source),'width':384,'height':320,'frames':3,'fps':30}
    records=[{'source':0,'frame':i,'accepted':True} for i in range(3)]
    scores=np.ones((3,20,20),np.float32)
    for i in range(3):scores[i,2+i*5:4+i*5,2:4]=1000
    native_dir=tmp_path/'native';native_dir.mkdir();(native_dir/'complete.json').write_text('{}')
    monkeypatch.setattr(pipeline,'probe',lambda *a:metadata)
    monkeypatch.setattr(pipeline,'analyze',lambda *a:(records,scores,0,tmp_path))
    monkeypatch.setattr(native_scan,'native_candidates',lambda *a:(records,scores,0,np.ones((20,20)),native_dir))
    captured=[]
    class StopAfterSelection(Exception):pass
    def capture(sources,selected,*args):
        captured.extend(selected);raise StopAfterSelection
    monkeypatch.setattr(pipeline,'extract',capture)
    recipe={'roi':[0,0,1,1],'fit_roi':[.1,.1,.4,.9],
            'validation_rois':[[.6,.1,.9,.9]],'anchor':{'source':0,'frame':1}}
    config=pipeline.Config(max_frames=3,selection_mode='native',fusion_mode='spatial',local_alignment_regions=[recipe])
    with pytest.raises(StopAfterSelection):pipeline.run_stack([source],tmp_path/'out',config)
    assert {r['frame'] for r in captured}=={0,1,2}
