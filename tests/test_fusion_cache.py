"""Pre-compositor reuse retains exact pixels, evidence and bounded scratch."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from macrostitch import fusion_cache as cache, pipeline, spatial, storage
from macrostitch.compositing import compose_output
from test_compositing import scene, rectangle, SUBJECT


@pytest.fixture
def snapshot_scene(tmp_path, scene):
    paths, common, _, records = scene
    for i, row in enumerate(records):
        row.update(extracted=str(paths[i]), aligned=str(paths[i]),
                   full_transform=[[1., 0., 0.], [0., 1., 0.]],
                   full_alignment={'correlation': 1.})
    sources = [{'sha256': 'test-video', 'width': 512, 'height': 384}]
    config = pipeline.Config(fusion_mode='spatial', background_opacity=.4,
                             canvas_mode='reference', subject_polygon=SUBJECT,
                             roi=(.3, .3, .65, .75), pyramid_levels=2)
    config.validate()
    return paths, common, records, sources, config


def make_snapshot(tmp_path, fixture, config=None):
    paths, common, records, sources, original = fixture
    config = config or original
    output = tmp_path / 'base'; output.mkdir()
    result = spatial.fuse_spatial(paths, 0, common, config, output, lambda **e: None, records)
    identifier = cache.identity(config, sources, records, 0, pipeline.VERSION)
    root = tmp_path / 'cache'
    cache.save(root, identifier, output, config, records, 0, paths, common, result)
    return root, identifier, output, result


@pytest.mark.parametrize('canvas', ['reference', 'common'])
@pytest.mark.parametrize('variant', ['opacity', 'margin', 'protect', 'exclude', 'overlap', 'empty'])
def test_recomposed_snapshot_is_bit_exact_and_immutable(tmp_path, snapshot_scene, canvas, variant):
    paths, common, records, sources, config = snapshot_scene
    config = replace(config, canvas_mode=canvas)
    root, identifier, original, result = make_snapshot(tmp_path, snapshot_scene, config)
    protect = rectangle('protect', 'reference', .35, .35, .7, .7)
    exclude = rectangle('exclude', 'exclude', .45, .45, .6, .6)
    changes = {'opacity': {'background_opacity': 1.},
               'margin': {'background_opacity': 0., 'background_margin_px': 12.},
               'protect': {'object_masks': (protect,)},
               'exclude': {'object_masks': (exclude,)},
               'overlap': {'object_masks': (exclude, protect)},
               'empty': {'object_masks': (rectangle('all', 'exclude', 0, 0, 1, 1),)}}[variant]
    changed = replace(config, **changes); changed.validate()
    assert cache.identity(changed, sources, records, 0, pipeline.VERSION) == identifier
    directory = root / 'fusion' / identifier
    checksums = json.loads((directory / 'complete.json').read_text())['sha256']
    # Composition destroys output information; preserve and restore from copies.
    compose_output(original, paths, 0, common, config, result)
    restored = tmp_path / 'restored'; restored.mkdir()
    current_records = [{k: v for k, v in row.items() if k not in cache.ALIGNMENT_FIELDS} for row in records]
    hit = cache.restore(root, identifier, restored, changed, current_records, 0, common.shape)
    assert hit is not None and current_records == records
    restored_paths, restored_common, restored_result = hit
    compose_output(restored, restored_paths, 0, restored_common, changed, restored_result)
    fresh = tmp_path / 'fresh'; fresh.mkdir()
    fresh_result = spatial.fuse_spatial(paths, 0, common, changed, fresh, lambda **e: None, records)
    compose_output(fresh, paths, 0, common, changed, fresh_result)
    assert restored_result == fresh_result
    for path in fresh.iterdir():
        assert (restored / path.name).read_bytes() == path.read_bytes(), path.name
    for name, checksum in checksums.items():
        assert pipeline.digest(directory / name) == checksum
    assert (restored / 'stack.png').stat().st_ino != (directory / 'stack.png').stat().st_ino


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'checksum', 'manifest', 'nested_json', 'shape', 'alignment', 'crop', 'symlink'])
def test_damaged_snapshot_is_miss_without_partial_restore(tmp_path, snapshot_scene, damage):
    paths, common, records, _, config = snapshot_scene
    root, identifier, output, _ = make_snapshot(tmp_path, snapshot_scene)
    directory = root / 'fusion' / identifier
    manifest_path = directory / 'complete.json'
    manifest = json.loads(manifest_path.read_text())
    if damage == 'missing':
        (directory / 'stack.png').unlink()
    elif damage == 'truncated':
        (directory / 'common.npy').write_bytes(b'\x93NUMPY')
    elif damage == 'checksum':
        (directory / 'confidence.png').write_bytes(b'wrong')
    elif damage == 'manifest':
        manifest_path.write_text('[]')
    elif damage == 'nested_json':
        manifest_path.write_text('[' * 20000 + ']' * 20000)
    elif damage == 'symlink':
        (directory / 'stack.png').unlink()
        (directory / 'stack.png').symlink_to(output / 'stack.png')
    else:
        state_path = directory / 'state.json'
        state = json.loads(state_path.read_text())
        if damage == 'shape': state['shape'] = [1, 1]
        elif damage == 'alignment': state['alignment'][0]['full_transform'] = [[0, 0, 0], [0, 0, 0]]
        else: state['result']['crop_xyxy'] = [-1, 0, 512, 384]
        state_path.write_text(json.dumps(state))
        manifest['sha256']['state.json'] = pipeline.digest(state_path)
        manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / 'destination'; destination.mkdir()
    before = deepcopy(records)
    assert cache.restore(root, identifier, destination, config, records, 0, common.shape) is None
    assert records == before and not list(destination.iterdir())


def test_looping_fusion_namespace_is_skipped(tmp_path, snapshot_scene):
    paths, common, records, sources, config = snapshot_scene
    root = tmp_path / 'loop-cache'; root.mkdir()
    (root / 'fusion').symlink_to('fusion', target_is_directory=True)
    identifier = cache.identity(config, sources, records, 0, pipeline.VERSION)
    output = tmp_path / 'loop-output'; output.mkdir()
    assert cache.restore(root, identifier, output, config, records, 0, common.shape) is None
    cache.save(root, identifier, output, config, records, 0, paths, common, {})
    assert (root / 'fusion').is_symlink()
    assert not list(output.iterdir())


def test_key_invalidates_every_fusion_dependency_and_guards_activation(snapshot_scene):
    _, _, records, sources, config = snapshot_scene
    def key(candidate=config, videos=sources, rows=records, reference=0, version=pipeline.VERSION):
        return cache.identity(candidate, videos, rows, reference, version)
    original = key()
    assert original
    assert key(replace(config, background_opacity=None)) is None
    assert key(replace(config, fusion_mode='pyramid')) is None
    for changes in ({'background_mode': 'transparent'}, {'canvas_mode': 'common'},
                    {'matte_feather_px': 2.}, {'matte_erode_px': 2.},
                    {'pyramid_levels': 3}, {'spatial_blend_px': 2.},
                    {'spatial_smoothness': .2}, {'max_frames': 50}, {'coverage': .99},
                    {'subject_polygon': ((.2,.2),(.7,.2),(.7,.7))},
                    {'object_masks': (rectangle('more', 'stack', .7,.2,.9,.7),)}):
        changed = replace(config, **changes); changed.validate()
        assert key(changed) != original
    assert key(videos=[dict(sources[0], sha256='changed')]) != original
    assert key(rows=[dict(records[0], frame=19), records[1]]) != original
    assert key(rows=[dict(records[0], transform=[[1,0,2],[0,1,0]]), records[1]]) != original
    assert key(reference=1) != original
    assert key(version='new-version') != original


def test_fusion_retention_is_bounded_on_success_and_failure(tmp_path):
    for failing in (False, True):
        try:
            with storage.cache_retention(tmp_path, 0):
                entry = tmp_path / 'fusion' / 'key'; entry.mkdir(parents=True)
                (entry / 'common.npy').write_bytes(b'scratch')
                if failing: raise RuntimeError('interrupted')
        except RuntimeError:
            pass
        assert not entry.exists()
    for stage in ('frames', 'proxies', 'fusion'):
        entry = tmp_path / stage / 'entry'; entry.mkdir(parents=True)
        (entry / 'data').write_bytes(b'x' * 10)
    storage.prune_cache(tmp_path, 10)
    assert (tmp_path / 'fusion' / 'entry').exists()
    assert not (tmp_path / 'frames' / 'entry').exists()
    assert not (tmp_path / 'proxies' / 'entry').exists()


def test_local_validation_evidence_survives_reuse(tmp_path, snapshot_scene):
    paths, common, records, sources, config = snapshot_scene
    config = replace(config, selection_mode='native', local_alignment_regions=({
        'roi': (.3,.3,.65,.75), 'fit_roi': (.32,.33,.44,.7),
        'validation_rois': ((.47,.33,.63,.7),),
        'anchor': {'source': 0, 'frame': 0}, 'feather_px': 8},))
    config.validate()
    root, identifier, original, _ = make_snapshot(tmp_path, snapshot_scene, config)
    local = (original / 'local-alignment.json').read_bytes()
    changed = replace(config, background_opacity=.9)
    restored = tmp_path / 'restored'; restored.mkdir()
    hit = cache.restore(root, identifier, restored, changed, records, 0, common.shape)
    assert hit is not None
    assert (restored / 'local-alignment.json').read_bytes() == local
    assert json.loads(local)['regions'][0]['provenance']['inputs']
    modified = deepcopy(changed.local_alignment_regions)
    modified[0]['anchor']['frame'] = 1
    assert cache.identity(replace(changed, local_alignment_regions=modified), sources, records, 0, pipeline.VERSION) != identifier


def test_failed_snapshot_publish_cleans_pending_media(tmp_path, snapshot_scene, monkeypatch):
    paths, common, records, _, config = snapshot_scene
    root, identifier, original, result = make_snapshot(tmp_path, snapshot_scene)
    original_manifest = (root / 'fusion' / identifier / 'complete.json').read_bytes()
    copy = cache.shutil.copyfile
    calls = []
    def failing_copy(source, destination):
        calls.append(source)
        if len(calls) == 2: raise OSError('simulated interrupted copy')
        return copy(source, destination)
    monkeypatch.setattr(cache.shutil, 'copyfile', failing_copy)
    with pytest.raises(OSError, match='interrupted'):
        cache.save(root, identifier, original, config, records, 0, paths, common, result)
    assert not list((root / 'fusion').glob('pending-*'))
    assert (root / 'fusion' / identifier / 'complete.json').read_bytes() == original_manifest


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_real_video_reuse_skips_extraction_alignment_fusion_and_keeps_exports(tmp_path, monkeypatch, snapshot_scene):
    frames = tmp_path / 'input'; frames.mkdir()
    image = cv2.imread(str(snapshot_scene[0][1]), cv2.IMREAD_UNCHANGED)
    for i in range(5):
        frame = cv2.GaussianBlur(image, (0, 0), .5 + abs(i - 2) * .7)
        assert cv2.imwrite(str(frames / f'f{i:03}.png'), frame)
    video = tmp_path / 'rack.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-framerate', '4', '-i', str(frames / 'f%03d.png'),
                    '-c:v', 'ffv1', '-pix_fmt', 'yuv444p10le', str(video)], check=True)
    config = pipeline.Config(sample_fps=4, proxy_width=512, max_frames=4, threads=1,
                             pyramid_levels=2, fusion_mode='spatial', canvas_mode='reference',
                             roi=(.3,.3,.65,.75), subject_polygon=SUBJECT, background_opacity=.25)
    root = tmp_path / 'cache'; output = tmp_path / 'first'
    pipeline.run_stack([video], output, config, cache_dir=root)
    changed = replace(config, background_opacity=.75, background_margin_px=5,
                      object_masks=(rectangle('protect', 'reference', .2,.3,.4,.7),
                                    rectangle('exclude', 'exclude', .5,.3,.6,.7)))
    # Prove the snapshot remains sufficient after normal eviction of frame stacks.
    for name in ('frames', 'aligned'):
        shutil.rmtree(root / name)
    events = []
    def forbidden(*args, **kwargs): pytest.fail('Fusion snapshot must bypass full frame work')
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'extract', forbidden)
        patch.setattr(pipeline, 'align_selected', forbidden)
        patch.setattr(spatial, 'fuse_spatial', forbidden)
        reused = pipeline.run_stack([video], tmp_path / 'reuse', changed, cache_dir=root, progress=events.append)
    assert not reused['cache_hit'] and any(e['stage'] == 'fusion_cache' for e in events)
    fresh = pipeline.run_stack([video], tmp_path / 'fresh', changed, cache_dir=root, keep_frames=True)
    for name in pipeline.artifact_names(changed):
        if name not in ('events.json', 'selection.json'):
            assert reused['artifact_sha256'][name] == fresh['artifact_sha256'][name], name
    assert reused['coverage'] == fresh['coverage']
    assert reused['result'] == fresh['result']
    assert json.loads((tmp_path / 'reuse' / 'selection.json').read_text()) == json.loads((tmp_path / 'fresh' / 'selection.json').read_text())
    assert len(list((tmp_path / 'fresh' / 'frames').glob('*.tiff'))) == fresh['selected_frames']
    assert not any(e['stage'] == 'fusion_cache' for e in json.loads((tmp_path / 'fresh' / 'events.json').read_text()))
    # A zero budget deletes this private namespace as well as full frame caches.
    pipeline.run_stack([video], tmp_path / 'zero', changed, cache_dir=root, cache_max_bytes=0)
    assert not list((root / 'fusion').iterdir())
