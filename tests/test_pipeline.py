"""Real decoder → registration → fusion → cache integration, using a tiny focus rack."""
import json
import shutil
import subprocess
import cv2
import numpy as np
import pytest
from macrostitch.pipeline import Config, run_stack


@pytest.mark.skipif(not shutil.which("ffmpeg"),reason="FFmpeg required")
def test_extract_maximum_candidate_pool_without_expression_depth_failure(tmp_path):
    from macrostitch.pipeline import digest, extract
    video=tmp_path/"many-frames.mkv"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i",
        "testsrc2=size=32x24:rate=60:duration=18","-c:v","ffv1",
        "-pix_fmt","yuv444p10le",str(video)],check=True)
    source={"path":str(video),"sha256":digest(video),"width":32,"height":24}
    records=[{"source":0,"frame":frame} for frame in range(0,1024,2)]
    extract([source],records,tmp_path/"cache",Config(threads=1),lambda **event:None)
    assert len(list((tmp_path/"cache"/"frames").rglob("f*.tiff")))==512
    # Compare the last requested frame to an independent, single-frame decode.
    decoded=subprocess.run(["ffmpeg","-v","error","-i",str(video),"-vf",
        "select=eq(n\\,1022),format=rgb48le","-frames:v","1","-f","rawvideo",
        "-pix_fmt","rgb48le","-"],capture_output=True,check=True)
    expected=np.frombuffer(decoded.stdout,"<u2").reshape(24,32,3)[...,::-1]
    actual=cv2.imread(records[-1]["extracted"],cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(not shutil.which("ffmpeg"),reason="FFmpeg required")
def test_video_stack_and_verified_cache(tmp_path):
    frames=tmp_path/"input"; frames.mkdir()
    rng=np.random.default_rng(7)
    texture=rng.integers(45,200,(192,320,3),dtype=np.uint8)
    texture=cv2.GaussianBlur(texture,(0,0),.55)
    cv2.putText(texture,"MACRO",(25,105),cv2.FONT_HERSHEY_SIMPLEX,1.2,(235,235,235),2)
    for i in range(8):
        left=cv2.GaussianBlur(texture,(0,0),.4+i*.5)
        right=cv2.GaussianBlur(texture,(0,0),.4+(7-i)*.5)
        left[:,160:]=right[:,160:]
        cv2.imwrite(str(frames/f"f{i:03}.png"),left)
    video=tmp_path/"rack.mkv"
    subprocess.run(["ffmpeg","-v","error","-framerate","4","-i",str(frames/"f%03d.png"),
        "-c:v","ffv1","-pix_fmt","yuv444p10le","-color_primaries","bt709","-color_trc","bt709",
        "-colorspace","bt709",str(video)],check=True)
    config=Config(sample_fps=4,proxy_width=320,max_frames=6,coverage=.98,roi=(.1,.1,.9,.9),threads=2)
    result=run_stack([video],tmp_path/"out",config)
    assert result["selected_frames"]>=2
    assert result["coverage"]["mean_coverage"]>.95
    image=cv2.imread(result["artifacts"]["stack.png"],cv2.IMREAD_UNCHANGED)
    assert image.dtype==np.uint16 and image.shape[2]==3
    assert image.shape[0]>170 and image.shape[1]>290
    original=result["artifact_sha256"]["stack.png"]
    cached=run_stack([video],tmp_path/"out",config)
    assert cached["cache_hit"] and cached["artifact_sha256"]["stack.png"]==original
    equivalent=Config(sample_fps=4.,proxy_width=320,max_frames=6,coverage=.98,roi=[.1,.1,.9,.9],threads=2)
    assert run_stack([video],tmp_path/"out",equivalent)["cache_hit"]
    legacy=json.loads((tmp_path/"out"/"report.json").read_text())
    legacy["config"]["sample_fps"]=4
    legacy["cache_key"]="legacy-noncanonical-key"
    (tmp_path/"out"/"report.json").write_text(json.dumps(legacy))
    assert run_stack([video],tmp_path/"out",equivalent)["cache_hit"]
    (tmp_path/"out"/"preview.jpg").write_bytes(b"damaged")
    repaired=run_stack([video],tmp_path/"out",config)
    assert not repaired["cache_hit"]
    assert repaired["artifact_sha256"]["stack.png"]==original
    manifest=json.loads((tmp_path/"out"/"selection.json").read_text())
    assert all(record["frame"]<8 for record in manifest["selected"])
    (tmp_path/"out").rename(tmp_path/"relocated")
    relocated=run_stack([video],tmp_path/"relocated",config)
    assert relocated["cache_hit"]
    assert relocated["artifacts"]["stack.png"]==str(tmp_path/"relocated"/"stack.png")


def test_config_rejects_invalid_roi_and_bounds():
    for config in [Config(roi=(.6,0,.2,1)),Config(sample_fps=0),Config(max_frames=1),Config(coverage=.3)]:
        with pytest.raises(ValueError): config.validate()


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_grayscale_video_proxies_are_uint16_bgr_and_reused(tmp_path,monkeypatch):
    from macrostitch import pipeline
    frames=tmp_path/'gray';frames.mkdir()
    yy,xx=np.indices((64,64))
    for index in range(3):
        gray=((xx*700+yy*200+index*1000)%65536).astype(np.uint16)
        assert cv2.imwrite(str(frames/f'f{index:03d}.png'),gray)
    video=tmp_path/'gray.mkv'
    subprocess.run(['ffmpeg','-v','error','-framerate','3','-i',str(frames/'f%03d.png'),
                    '-c:v','ffv1','-pix_fmt','gray16le',str(video)],check=True)
    source=pipeline.probe(video)
    assert source['pixel_format'].startswith('gray')
    config=Config(sample_fps=3,proxy_width=64,threads=1)
    directory,metadata=pipeline.proxies(source,tmp_path/'cache',config,lambda **event:None)
    assert metadata['count']==3
    before={}
    for path in sorted(directory.glob('p*.png')):
        pixels=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        assert pixels.shape==(64,64,3) and pixels.dtype==np.uint16
        np.testing.assert_array_equal(pixels[...,0],pixels[...,1])
        np.testing.assert_array_equal(pixels[...,1],pixels[...,2])
        before[path.name]=(pipeline.digest(path),path.stat().st_mtime_ns)
    monkeypatch.setattr(pipeline,'command',lambda *a,**kw:pytest.fail('Valid grayscale proxies must not decode again'))
    reused,again=pipeline.proxies(source,tmp_path/'cache',config,lambda **event:None)
    assert reused==directory and again==metadata
    assert before=={path.name:(pipeline.digest(path),path.stat().st_mtime_ns) for path in directory.glob('p*.png')}


def test_same_output_serialized_even_with_different_cache_directories(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    import threading
    from macrostitch import pipeline
    source=tmp_path/'source.mov';source.write_bytes(b'fixture')
    output=tmp_path/'out';entered=threading.Event();release=threading.Event();second_attempt=threading.Event()
    calls=[];guard=threading.Lock();real_lock=pipeline.lock
    @contextmanager
    def observed_lock(path):
        if path==output/'.pipeline.lock' and entered.is_set():second_attempt.set()
        with real_lock(path):yield
    def probe(path):
        with guard:calls.append(path);first=len(calls)==1
        if first:
            entered.set();assert release.wait(3)
        return {'sha256':'fixture','width':64,'height':64,'frames':1}
    def stop_after_probe(*args):raise RuntimeError('fixture pipeline stop')
    monkeypatch.setattr(pipeline,'lock',observed_lock)
    monkeypatch.setattr(pipeline,'probe',probe)
    monkeypatch.setattr(pipeline,'analyze',stop_after_probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first=executor.submit(pipeline.run_stack,[source],output,Config(),cache_dir=tmp_path/'cache-a')
        try:
            assert entered.wait(2)
            second=executor.submit(pipeline.run_stack,[source],output,Config(),cache_dir=tmp_path/'cache-b')
            assert second_attempt.wait(2)
            assert len(calls)==1 # Second worker reached output lock but cannot probe yet.
        finally:release.set()
        for future in (first,second):
            with pytest.raises(RuntimeError,match='fixture pipeline stop'):future.result(timeout=3)
    assert len(calls)==2
