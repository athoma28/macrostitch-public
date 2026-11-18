"""Experimental dense-texture fallback: real video decoding, both gates, safety."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
import pytest
from macrostitch import pipeline, spatial, native_scan
from macrostitch.imaging import to_linear_bt709

spec=importlib.util.spec_from_file_location('dense_prototype',Path(__file__).parent/'support/dense_fallback.py')
prototype=importlib.util.module_from_spec(spec);spec.loader.exec_module(prototype)


def make_video(root,noise=False,lossy=False):
    root.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(119)
    texture=cv2.GaussianBlur(rng.normal(0,.09,(256,256)).astype(np.float32),(0,0),.65)+.3
    for frame in range(13):
        if noise:image=.3+rng.normal(0,.06,texture.shape).astype(np.float32)
        else:
            left=cv2.GaussianBlur(texture,(0,0),.45+abs(frame-3)*.40)
            right=cv2.GaussianBlur(texture,(0,0),.45+abs(frame-9)*.40)
            image=np.concatenate([left[:,:128],right[:,128:]],axis=1)
            image+=rng.normal(0,.001,image.shape).astype(np.float32)
        encoded=np.where(image<.018,image*4.5,1.099*np.maximum(image,0)**.45-.099)
        cv2.imwrite(str(root/f'f{frame:03d}.png'),np.repeat(np.rint(np.clip(encoded,0,1)*65535).astype(np.uint16)[...,None],3,axis=2))
    video=root/'rack.mkv'
    codec=['-c:v','libx264','-crf','18'] if lossy else ['-c:v','ffv1']
    subprocess.run(['ffmpeg','-y','-v','error','-framerate','12','-i',str(root/'f%03d.png'),
                    *codec,'-pix_fmt','yuv444p','-colorspace','bt709','-color_trc','bt709','-color_primaries','bt709',str(video)],check=True)
    return video


def test_disabled_and_existing_structural_path_do_not_request_temporal_evidence():
    def forbidden():pytest.fail('Existing structural route must not decode temporal evidence')
    scores=np.ones((3,20,20),np.float32)
    scores[1,2:4,2:4]=1000;scores[2,16:18,16:18]=1000
    active=np.ones((20,20),np.uint8)
    expected=spatial.select_native(scores,active,3,.99)
    assert prototype.select_with_fallback(scores,active,3,.99,enabled=True,evidence_provider=forbidden)==expected
    with pytest.raises(ValueError,match='reliable subject structure'):
        prototype.select_with_fallback(np.ones_like(scores),active,3,.99,enabled=False,evidence_provider=forbidden)


def test_fallback_honors_required_anchor_and_never_exceeds_budget():
    raw=np.ones((3,10,10),np.float32);active=np.ones((10,10),np.uint8)
    evidence=np.zeros_like(raw);evidence[0,:,:5]=1;evidence[1,:,5:]=1
    chosen,report=prototype.select_with_fallback(raw,active,3,.99,enabled=True,
        evidence_provider=lambda:(evidence,{'experimental_opt_in':True}),required={2})
    assert set(chosen)=={0,1,2} and report['target_reached'] and report['mean_coverage']==1
    assert report['required_frames']==[2]
    with pytest.raises(ValueError,match='cannot fit'):
        prototype.select_with_fallback(raw,active,2,.99,enabled=True,
            evidence_provider=lambda:(evidence,{}),required={2})


def test_independent_noise_and_static_fixed_pattern_do_not_establish_focus():
    rng=np.random.default_rng(77);mask=np.ones((128,128),bool)
    records=[{'source':0,'frame':i} for i in range(5)]
    images=[rng.normal(.3,.06,mask.shape).astype(np.float32) for _ in records]
    with pytest.raises(ValueError,match='repeatable dense texture'):
        prototype.dense_evidence(images,records,mask)
    # A constant sensor pattern repeats perfectly, but has no focus variation.
    with pytest.raises(ValueError,match='focus variation'):
        prototype.dense_evidence([images[0]]*5,records,mask)
    with pytest.raises(ValueError,match='Duplicate captures'):
        prototype.dense_evidence(images,[records[0]]*5,mask)
    with pytest.raises(ValueError,match='repeatable dense texture'):
        prototype.dense_evidence(images,[{'source':i,'frame':0} for i in range(5)],mask)


@pytest.mark.parametrize('lossy',[False,True])
def test_encoded_independent_noise_still_rejected(tmp_path,lossy):
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg required')
    video=make_video(tmp_path/'noise',noise=True,lossy=lossy)
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','bgr48le','-'])
    frames=np.frombuffer(raw,np.uint16).reshape(-1,256,256,3)
    images=[to_linear_bt709(frame) for frame in frames]
    records=[{'source':0,'frame':i} for i in range(len(images))]
    with pytest.raises(ValueError,match='repeatable dense texture'):
        prototype.dense_evidence(images,records,np.ones((256,256),bool))


def test_experimental_fuser_is_bit_identical_on_existing_structural_path(tmp_path):
    rng=np.random.default_rng(127)
    image=np.full((256,256,3),18000,np.uint16)
    image[80:116,80:116]=rng.integers(10000,55000,(36,36,3),dtype=np.uint16)
    paths=[]
    for index,sigma in enumerate((.4,2.)):
        path=tmp_path/f'{index}.tiff'
        cv2.imwrite(str(path),cv2.GaussianBlur(image,(0,0),sigma));paths.append(path)
    common=np.ones((256,256),np.uint8)
    config=pipeline.Config(subject_only=False,canvas_mode='reference',pyramid_levels=3)
    records=[{'source':0,'frame':i,'timestamp':i} for i in range(2)]
    folders=[tmp_path/'normal',tmp_path/'experimental']
    for folder,fn in zip(folders,[spatial.fuse_spatial,prototype.experimental_fuser()]):
        folder.mkdir();fn(paths,0,common,config,folder,lambda **event:None,records)
    for name in ('stack.png','source_index.png','confidence.png','fusion.json'):
        assert (folders[0]/name).read_bytes()==(folders[1]/name).read_bytes()


@pytest.mark.parametrize('lossy',[False,True])
def test_encoded_dense_focus_sweep_exercises_both_gates_and_complete_pipeline(tmp_path,monkeypatch,lossy):
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg required')
    video=make_video(tmp_path/'input',lossy=lossy)
    config=pipeline.Config(sample_fps=4,proxy_width=256,max_frames=16,roi=(.2,.2,.8,.8),
                           selection_mode='native',fusion_mode='spatial',canvas_mode='reference',
                           subject_only=True,pyramid_levels=1,spatial_blend_px=0,threads=1)
    cache=tmp_path/'cache';cache.mkdir()
    quiet=lambda **event:None
    sources=[pipeline.probe(video)]
    records,_,reference,_=pipeline.analyze(sources,cache,config,quiet)
    records,scores,reference,active,directory=native_scan.native_candidates(sources,records,reference,cache,config,quiet)
    with pytest.raises(ValueError,match='reliable subject structure'):
        spatial.select_native(scores,active,15,.985)
    selected=[dict(r,candidate_index=i) for i,r in enumerate(records)]
    pipeline.extract(sources,selected,cache,config,quiet)
    paths,common=pipeline.align_selected(selected,reference,cache,config,quiet)
    baseline=tmp_path/'baseline';baseline.mkdir()
    with pytest.raises(ValueError,match='reliable native fusion pixels'):
        spatial.fuse_spatial(paths,reference,common,config,baseline,quiet,selected)
    images=[to_linear_bt709(cv2.imread(str(p),cv2.IMREAD_UNCHANGED)) for p in paths]
    mask=np.zeros(common.shape,bool);mask[52:204,52:204]=True;mask &= common>0
    evidence,metadata=prototype.dense_evidence(images,selected,mask)
    # Independently known optical peaks must win on the two physical planes.
    winners=np.argmax(evidence,axis=0)
    assert (abs(winners[75:180,65:105]-3)<=1).mean()>.9
    assert (abs(winners[75:180,150:190]-9)<=1).mean()>.9
    def trial_selector(scores,active,max_frames,target,required=()):
        # Small-fixture proof keeps all native observations for the fusion gate's
        # independent-neighbor check. Production should store evidence maps and
        # retain only greedy coverage winners instead of retaining all frames.
        x0,y0,x1,y1=json.loads((directory/'complete.json').read_text())['native_crop_xyxy']
        reduced=np.asarray([cv2.resize(e[y0:y1,x0:x1],(active.shape[1],active.shape[0]),interpolation=cv2.INTER_AREA) for e in evidence])
        chosen,report=prototype.select_with_fallback(scores,active,max_frames,target,enabled=True,
                                                     evidence_provider=lambda:(reduced,metadata),required=required)
        report['prototype_coverage_winners']=chosen
        report['prototype_retains_all_observations_for_neighbor_check']=True
        retained=list(range(len(scores)))
        assert set(required)<=set(retained) and len(retained)<=max_frames
        return retained,report
    trial_fuser=prototype.experimental_fuser()
    monkeypatch.setattr(spatial,'select_native',trial_selector)
    monkeypatch.setattr(spatial,'fuse_spatial',trial_fuser)
    report=pipeline.run_stack([video],tmp_path/'fallback',config,cache_dir=cache)
    out=tmp_path/'fallback'
    labels=cv2.imread(str(out/'source_index.png'),cv2.IMREAD_UNCHANGED)
    assert (abs(labels[75:180,65:105].astype(int)-3)<=1).mean()>.9
    assert (abs(labels[75:180,150:190].astype(int)-9)<=1).mean()>.9
    assert report['result']['width']==256 and report['result']['height']==256
    fusion=json.loads((out/'fusion.json').read_text())
    assert fusion['dense_texture_fallback']['experimental_opt_in']
    (tmp_path/'proof.json').write_text(json.dumps({'native_gate':'rejected','spatial_gate':'rejected',
        'fallback':report['coverage'],'fusion':fusion},indent=2))
