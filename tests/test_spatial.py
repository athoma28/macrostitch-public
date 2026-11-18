import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from macrostitch import spatial
from macrostitch.pipeline import Config
from macrostitch.imaging import to_linear_bt709,to_srgb16


def test_native_selection_rejects_featureless_and_uniform_noise():
    scores=np.ones((4,10,10),np.float32)
    scores[0]*=100 # bright, featureless energy floor
    scores[1]=np.random.default_rng(2).uniform(5,10,(10,10))
    scores[2,1:3,1:3]=1000
    scores[3,7:9,7:9]=1000 # independent feature unavailable in frame2
    chosen,report=spatial.select_native(scores,np.ones((10,10)),3,.999)
    assert set(chosen)=={2,3}
    assert report['target_reached'] and report['structurally_eligible_frames']==2


def test_native_selection_retains_unsampled_peak():
    scores=np.ones((7,10,10),np.float32)
    scores[0,1:3,1:3]=100
    scores[6,1:3,1:3]=80
    scores[3,7:9,7:9]=200 # native frame absent from hypothetical proxy samples0,6
    chosen,_=spatial.select_native(scores,np.ones((10,10)),3,.99)
    assert 3 in chosen and 0 in chosen


def test_native_selection_rejects_all_featureless():
    with pytest.raises(ValueError,match='reliable subject structure'):
        spatial.select_native(np.ones((3,10,10)),np.ones((10,10)),2,.98)


def test_coherent_labels_preserves_2d_quadrants_and_extends_weak_pixels():
    yy,xx=np.indices((32,40));truth=((yy>=16)*2+(xx>=20)).astype(np.int32)
    scores=np.ones((4,32,40),np.float32)
    for i in range(4):scores[i][truth==i]=10
    reliable=np.ones(truth.shape,bool);reliable[5:8,5:8]=False
    scores[:,~reliable]=0
    labels,_=spatial.coherent_labels(scores,np.zeros(truth.shape,np.float32),reliable,.12)
    np.testing.assert_array_equal(labels,truth)


def test_equivalent_focus_sources_consolidate_without_merging_real_depth_features():
    yy,xx=np.indices((32,40));truth=((yy>=16)*2+(xx>=20)).astype(np.int32)
    scores=np.full((8,32,40),.05,np.float32)
    for group in range(4):
        region=truth==group
        scores[2*group,region]=np.where((yy+xx)[region]%2,1.,.96)
        scores[2*group+1,region]=np.where((yy+xx)[region]%2,.96,1.)
    # Deliberately unordered identities across the two hypothetical clips.
    permutation=np.array([5,0,7,2,1,6,3,4]);scores=scores[permutation]
    groups=permutation//2
    guide=((yy+xx)%2).astype(np.float32) # contrast must not allow checkerboard seams
    labels,_=spatial.coherent_labels(scores,guide,np.ones(truth.shape,bool),.20)
    np.testing.assert_array_equal(groups[labels],truth)
    assert len(np.unique(labels))==4
    for group in range(4):
        assert len(np.unique(labels[truth==group]))==1


def test_edge_footprint_rejects_shifted_blur_and_noise_winning_one_edge_side():
    # A sharp step and a shifted defocused version represent the same physical
    # edge after imperfect alignment. A narrow local score wrongly prefers the
    # shifted blur on its far side, despite the first image having the sharp edge.
    yy,xx=np.indices((192,192))
    sharp=(.2+.4*(xx>=96)).astype(np.float32)
    blurred=cv2.GaussianBlur((.2+.4*(xx>=101)).astype(np.float32),(0,0),3)
    blurred+=np.random.default_rng(9).normal(0,.005,blurred.shape).astype(np.float32)
    narrow=[];wide=[]
    for gray in (sharp,blurred):
        band=(cv2.GaussianBlur(gray,(0,0),.8)-cv2.GaussianBlur(gray,(0,0),2.4))**2
        narrow.append(cv2.GaussianBlur(band,(0,0),4))
        wide.append(spatial.native_energy(np.repeat(gray[...,None],3,axis=2))[0])
    assert np.argmax(narrow,axis=0)[50:142,104:120].mean()>.5
    assert not np.argmax(wide,axis=0)[50:142,80:120].any()


@pytest.mark.parametrize('shape',[(1,1),(2,3)])
def test_coherent_labels_tiny(shape):
    labels,_=spatial.coherent_labels(np.ones((1,*shape),np.float32),np.zeros(shape,np.float32),np.ones(shape,bool))
    assert labels.shape==shape and not labels.any()


def compose_fixture(tmp_path,shape=(64,96)):
    paths=[]
    for i,value in enumerate((16000,32000,48000)):
        path=tmp_path/f'{i}.tiff'
        cv2.imwrite(str(path),np.full((*shape,3),value,np.uint16));paths.append(path)
    common=np.zeros(shape,np.uint8);common[8:-8,8:-8]=1
    config=Config(fusion_mode='spatial',canvas_mode='reference',background_mode='reference',
                  subject_only=False,pyramid_levels=4,spatial_blend_px=0)
    records=[{'source':0,'frame':i,'timestamp':i} for i in range(3)]
    return paths,common,config,records


def test_cached_source_crops_never_alias_distinct_frames_in_one_directory(tmp_path):
    paths,_,_,_=compose_fixture(tmp_path)
    region=(slice(10,30),slice(20,40))
    for index,path in enumerate(paths):
        expected=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[region]
        np.testing.assert_array_equal(spatial.crop_pixels(path,region),expected)
    for path in reversed(paths):
        expected=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[region]
        np.testing.assert_array_equal(spatial.crop_pixels(path,region),expected)


def test_spatial_full_canvas_and_unused_label_context(tmp_path,monkeypatch):
    paths,common,config,records=compose_fixture(tmp_path)
    calls=[]
    def energy(image):
        i=len(calls);calls.append(i)
        e=np.ones(image.shape[:2],np.float32)*1e-6
        if i==1:e[20:35,20:35]=1 # onlysource1 hasreliable structure;source2 unused
        return e,np.zeros(e.shape,np.float32)
    monkeypatch.setattr(spatial,'native_energy',energy)
    out=tmp_path/'out';out.mkdir()
    result=spatial.fuse_spatial(paths,0,common,config,out,lambda **e:None,records)
    merged=cv2.imread(str(out/'stack.png'),cv2.IMREAD_UNCHANGED)
    reference=to_srgb16(to_linear_bt709(cv2.imread(str(paths[0]),cv2.IMREAD_UNCHANGED)))
    assert merged.shape==(64,96,3) and result['crop_xyxy']==[0,0,96,64]
    np.testing.assert_array_equal(merged[common==0],reference[common==0])
    assert merged.min()>0
    assert merged[25,25,0]>reference[25,25,0]
    info=json.loads((out/'fusion.json').read_text())
    assert 2 not in [r['source_index'] for r in info['contributing_sources']]
    assert info['native_energy_aggregation_sigma_px']==20
    assert info['label_coherence']['source_palette']==[1]
    assert info['label_coherence']['equivalent_score_threshold']==.90


def test_real_native_energy_prefers_local_structure_to_brightness():
    flat=np.full((128,128,3),.8,np.float32)
    texture=np.full_like(flat,.2)
    yy,xx=np.indices((24,24))
    texture[40:64,40:64]=(.2+.2*((yy+xx)%2))[...,None]
    a,_=spatial.native_energy(flat);b,_=spatial.native_energy(texture)
    assert b.max()>a.max()+1e-7


def test_weak_focused_texture_survives_adjacent_stronger_focus_plane(tmp_path):
    # Two optical planes: a high-contrast broad feature is sharp in frame0,
    # while much weaker fine texture is sharp in frame1. Shared sensor noise
    # ensures an artificial noisier frame cannot supply the winning signal.
    yy,xx=np.indices((288,288))
    strong=np.full((288,288),.3,np.float32)
    strong[30:80,35:250]=.3+.15*np.sin(xx[30:80,35:250]*np.pi/12)
    weak=np.zeros_like(strong)
    weak[165:255,55:235]=.01*np.sin(xx[165:255,55:235]*np.pi/3)
    noise=np.random.default_rng(411).normal(0,.006,strong.shape).astype(np.float32)
    frames=[strong+cv2.GaussianBlur(weak,(0,0),3)+noise,
            cv2.GaussianBlur(strong,(0,0),3)+weak+noise]
    paths=[]
    for index,linear in enumerate(frames):
        bt709=np.where(linear<.018,linear*4.5,1.099*np.maximum(linear,0)**.45-.099)
        image=np.repeat(np.rint(bt709*65535).astype(np.uint16)[...,None],3,axis=2)
        path=tmp_path/f'plane{index}.tiff';assert cv2.imwrite(str(path),image);paths.append(path)
    output=tmp_path/'weak-texture';output.mkdir()
    config=Config(fusion_mode='spatial',canvas_mode='reference',subject_only=False,
                  pyramid_levels=1,spatial_blend_px=0,spatial_smoothness=.2)
    spatial.fuse_spatial(paths,0,np.ones(strong.shape,np.uint8),config,output,lambda **e:None,
                        [{'source':0,'frame':index,'timestamp':index} for index in range(2)])
    labels=cv2.imread(str(output/'source_index.png'),cv2.IMREAD_UNCHANGED)
    assert (labels[45:65,55:225]==0).mean()>.95
    assert (labels[185:235,80:210]==1).mean()>.95
    actual=cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
    focused=to_srgb16(to_linear_bt709(cv2.imread(str(paths[1]),cv2.IMREAD_UNCHANGED)))
    np.testing.assert_array_equal(actual[200:220,120:160],focused[200:220,120:160])


def test_pipeline_routes_native_candidates_and_spatial_fusion(tmp_path,monkeypatch):
    from macrostitch import pipeline,native_scan
    source=tmp_path/'source.mov';source.write_bytes(b'synthetic')
    extracted=tmp_path/'frame.tiff';cv2.imwrite(str(extracted),np.ones((8,8,3),np.uint16))
    records=[{'source':0,'frame':i,'accepted':True,'extracted':str(extracted),'aligned':str(extracted)} for i in range(3)]
    scores=np.ones((3,10,10),np.float32);scores[1,1:3,1:3]=100;scores[2,7:9,7:9]=100
    native_dir=tmp_path/'native';native_dir.mkdir();(native_dir/'complete.json').write_text('{"all_frames":3}')
    monkeypatch.setattr(pipeline,'probe',lambda p:{'sha256':'fake','width':8,'height':8,'frames':3})
    monkeypatch.setattr(pipeline,'analyze',lambda *a:([records[0]],np.zeros((1,10,10)),0,tmp_path))
    seen=[]
    def native(*args):
        assert len(args[1])==1 and args[4].selection_mode=='native'
        return records,scores,0,np.ones((10,10)),native_dir
    monkeypatch.setattr(native_scan,'native_candidates',native)
    monkeypatch.setattr(pipeline,'select_coverage',lambda *a:pytest.fail('Proxy selection must not run'))
    monkeypatch.setattr(pipeline,'extract',lambda *a:None)
    monkeypatch.setattr(pipeline,'align_selected',lambda *a:([extracted]*3,np.ones((8,8),np.uint8)))
    def fuse(paths,reference,common,config,output,emit,selected):
        seen.extend(r['frame'] for r in selected)
        assert config.canvas_mode=='reference' and reference==0
        for name in ['stack.tiff','stack.png','preview.jpg','comparison.jpg','source_index.png','confidence.png','subject-detail.png']:
            (output/name).write_bytes(b'test')
        (output/'fusion.json').write_text('{}')
        return {'width':8,'height':8}
    monkeypatch.setattr(spatial,'fuse_spatial',fuse)
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *a,**k:'ffmpeg test')
    result=pipeline.run_stack([source],tmp_path/'out',Config(selection_mode='native',fusion_mode='spatial',canvas_mode='reference',max_frames=3))
    assert seen==[0,1,2]
    assert result['candidates']==3 and result['selected_frames']==3
    assert 'native-analysis.json' in result['artifact_sha256']


def test_tiny_fusion_preserves_sparse_native_evidence(tmp_path,monkeypatch):
    path=tmp_path/'tiny.tiff';cv2.imwrite(str(path),np.full((4,4,3),24000,np.uint16))
    def energy(image):
        values=np.full((4,4),1e-6,np.float32);values[1,1]=1
        return values,np.zeros((4,4),np.float32)
    monkeypatch.setattr(spatial,'native_energy',energy)
    output=tmp_path/'tiny-out';output.mkdir()
    config=Config(fusion_mode='spatial',canvas_mode='reference',subject_only=False)
    result=spatial.fuse_spatial([path],0,np.ones((4,4),np.uint8),config,output,
                               lambda **e:None,[{'source':0,'frame':0,'timestamp':0}])
    assert (result['width'],result['height'])==(4,4)
    assert cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED).min()>0
