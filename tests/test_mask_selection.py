"""Object coverage, scan reuse and real encoded-video output/cache integration."""
from dataclasses import replace
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from macrostitch import native_scan as scan
from macrostitch.compositing import selection_mask
from macrostitch.imaging import select_coverage
from macrostitch.pipeline import Config, run_stack, artifact_names
from test_native_scan import video_fixture  # Reuse real decode fixture and pytest cleanup.


def rectangle(name, action, x0, y0, x1, y1):
    return {'name': name, 'action': action,
            'polygon': ((x0,y0),(x1,y0),(x1,y1),(x0,y1))}


def test_native_extra_object_expands_crop_and_active_cells(tmp_path, video_fixture):
    source, proxies, config = video_fixture
    config.roi = (.1, .2, .4, .7)
    config.subject_polygon = rectangle('subject','stack',.1,.2,.4,.7)['polygon']
    config.validate()
    base = scan.native_candidates([source], proxies, 1, tmp_path/'cache', config, lambda **e: None)
    extended = replace(config, object_masks=(rectangle('extra','stack',.72,.2,.95,.7),))
    extended.validate()
    result = scan.native_candidates([source], proxies, 1, tmp_path/'cache', extended, lambda **e: None)
    before = json.loads((base[4]/'complete.json').read_text())['native_crop_xyxy']
    crop = json.loads((result[4]/'complete.json').read_text())['native_crop_xyxy']
    assert crop[2] > before[2] and crop[2] == source['width']
    active = result[3]
    # Cell centers are mapped through the actual cropped score-grid geometry.
    xs = crop[0] + (np.arange(active.shape[1])+.5)*(crop[2]-crop[0])/active.shape[1]
    ys = crop[1] + (np.arange(active.shape[0])+.5)*(crop[3]-crop[1])/active.shape[0]
    assert active[np.ix_((ys>32)&(ys<80), (xs>96)&(xs<120))].any()
    assert not active[:, (xs>60)&(xs<80)].any()
    assert result[4] != base[4]


def test_presentation_controls_reuse_scan_but_moved_stack_object_invalidates(tmp_path, video_fixture, monkeypatch):
    source, proxies, config = video_fixture
    config.object_masks = (rectangle('extra','stack',.7,.2,.95,.8),)
    config.validate()
    baseline = scan.native_candidates([source], proxies, 1, tmp_path/'cache', config, lambda **e: None)
    protected = rectangle('protect','reference',.1,.1,.3,.3)
    excluded = rectangle('hide','exclude',.4,.4,.6,.6)
    variants = [replace(config, background_opacity=.25), replace(config, background_margin_px=12),
                replace(config, object_masks=config.object_masks+(protected,excluded))]
    with monkeypatch.context() as patch:
        patch.setattr(scan.subprocess, 'Popen', lambda *a, **k: pytest.fail('Presentation changes must reuse native decode'))
        for candidate in variants:
            candidate.validate()
            cached = scan.native_candidates([source], proxies, 1, tmp_path/'cache', candidate, lambda **e: None)
            assert cached[4] == baseline[4]
            np.testing.assert_array_equal(cached[1], baseline[1])
    moved = replace(config, object_masks=(rectangle('extra','stack',.1,.2,.4,.8),))
    moved.validate()
    rebuilt = scan.native_candidates([source], proxies, 1, tmp_path/'cache', moved, lambda **e: None)
    assert rebuilt[4] != baseline[4]
    assert not np.array_equal(rebuilt[3], baseline[3])


def test_proxy_union_requires_frame_sharp_only_in_extra_object():
    config = Config(roi=(.1,.2,.4,.8),
                    subject_polygon=rectangle('subject','stack',.1,.2,.4,.8)['polygon'])
    config.validate()
    scores = np.full((2,100,100), .01, np.float32)
    scores[0,20:81,10:41] = 1
    scores[1,20:81,70:96] = 1
    original, _ = select_coverage(scores, selection_mask((100,100),config), max_frames=2)
    assert original == [0]
    config.object_masks = (rectangle('extra','stack',.7,.2,.95,.8),)
    config.validate()
    selected, coverage = select_coverage(scores, selection_mask((100,100),config), max_frames=2)
    assert set(selected) == {0,1}
    assert coverage['target_reached']


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_real_video_controls_artifacts_and_cache_invalidation(tmp_path):
    frames = tmp_path/'input'; frames.mkdir()
    rng = np.random.default_rng(111)
    texture = rng.integers(40,210,(192,320,3),dtype=np.uint8)
    texture = cv2.GaussianBlur(texture,(0,0),.55)
    cv2.putText(texture,'MASK',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1.2,(240,240,240),2)
    for i in range(6):
        image = cv2.GaussianBlur(texture,(0,0),.5+i*.4)
        image[:,160:] = cv2.GaussianBlur(texture,(0,0),.5+(5-i)*.4)[:,160:]
        assert cv2.imwrite(str(frames/f'f{i:03}.png'),image)
    video = tmp_path/'rack.mkv'
    subprocess.run(['ffmpeg','-v','error','-framerate','4','-i',str(frames/'f%03d.png'),
                    '-c:v','ffv1','-pix_fmt','yuv444p10le','-color_primaries','bt709',
                    '-color_trc','bt709','-colorspace','bt709',str(video)],check=True)
    config = Config(sample_fps=4,proxy_width=320,max_frames=5,threads=1,pyramid_levels=2,
                    canvas_mode='reference',roi=(.1,.1,.9,.9),background_opacity=.25,
                    subject_polygon=rectangle('subject','stack',.2,.2,.7,.8)['polygon'])
    output = tmp_path/'out'
    first = run_stack([video],output,config)
    expected = set(artifact_names(config))
    assert {'compositing.json','reference.png','matte.png','subject.png'} <= expected
    assert set(first['artifact_sha256']) == expected
    assert set(first['artifacts']) == expected | {'report.json'}
    image = cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
    assert image.shape == (192,320,4)
    assert image[10,10,3] == round(.25*65535)
    assert run_stack([video],output,config)['cache_hit']
    opaque = replace(config,background_opacity=.75)
    changed = run_stack([video],output,opaque)
    assert not changed['cache_hit'] and changed['cache_key'] != first['cache_key']
    assert changed['artifact_sha256']['stack.png'] != first['artifact_sha256']['stack.png']
    hidden = replace(opaque,object_masks=(rectangle('remove','exclude',.4,.4,.6,.6),))
    final = run_stack([video],output,hidden)
    assert not final['cache_hit'] and final['cache_key'] != changed['cache_key']
    image = cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
    assert not image[96,160].any()
    assert run_stack([video],output,hidden)['cache_hit']
