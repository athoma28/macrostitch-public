"""Native scan acceptance tests with real decoding and independently known focus."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from macrostitch import native_scan as scan
from macrostitch.pipeline import Config, digest


@pytest.fixture
def video_fixture(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required")
    directory = tmp_path / "frames"
    directory.mkdir()
    rng = np.random.default_rng(631)
    texture = rng.integers(9000, 55000, (128, 128), dtype=np.uint16)
    for frame in range(13):
        image = cv2.GaussianBlur(texture, (0, 0), .5 + abs(frame - 3) * .55)
        assert cv2.imwrite(str(directory / f"f{frame:03d}.png"), image)
    video = tmp_path / "rack.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "12", "-i", str(directory / "f%03d.png"),
                    "-c:v", "ffv1", "-pix_fmt", "gray16le", str(video)], check=True)
    proxies = []
    for frame in (0, 6, 12):
        path = tmp_path / f"proxy{frame}.png"
        image = cv2.imread(str(directory / f"f{frame:03d}.png"), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(path), (cv2.resize(image, (64, 64)) / 256).astype(np.uint8))
        proxies.append({"source": 0, "frame": frame, "proxy": str(path),
                        "transform": [[1, 0, 0], [0, 1, 0]], "accepted": True,
                        "alignment": {"correlation": 1}})
    source = {"path": str(video), "sha256": digest(video), "width": 128, "height": 128,
              "frames": 13, "fps": 12}
    config = Config(roi=(.25, .25, .75, .75), threads=1)
    config.validate()
    return source, proxies, config


def test_native_scan_finds_peak_missing_from_six_frame_proxy_and_reuses_valid_cache(tmp_path, video_fixture, monkeypatch):
    source, proxies, config = video_fixture
    events = []
    records, scores, reference, active, directory = scan.native_candidates(
        [source], proxies, 1, tmp_path / "cache", config, lambda **event: events.append(event))
    assert len(records) == 13 and [row["frame"] for row in records] == list(range(13))
    assert records[reference]["frame"] == 6 and records[reference]["proxy"] == proxies[1]["proxy"]
    np.testing.assert_allclose([row["timestamp"] for row in records], np.arange(13) / 12, atol=.0005)
    raw = np.load(directory / "raw_scores.npy")
    raw_quality = raw[:, active > 0].mean(axis=1)
    smoothed_quality = scores[:, active > 0].mean(axis=1)
    assert np.argmax(raw_quality) == 3
    assert np.argmax(smoothed_quality) == 3
    assert raw_quality[3] > 3 * raw_quality[[0, 6, 12]].max()
    metadata = json.loads((directory / "complete.json").read_text())
    assert metadata["all_frames_examined"] and metadata["frame_counts"] == [13]
    assert metadata["native_crop_xyxy"] == [0, 0, 128, 128]
    assert (directory / "decode-0.log").exists()
    assert any(event["completed"] == 13 for event in events)
    monkeypatch.setattr(scan.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("A complete cache must not decode"))
    cached = scan.native_candidates([source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    np.testing.assert_array_equal(cached[1], scores)
    assert cached[0] == records


def test_crop_transform_maps_source_feature_to_reference_and_inverse_union_contains_it():
    # A physical source pixel (80,60) maps to reference (171,117). The native
    # cropped image stores it at (60,30); the target stores it at (71,37).
    full = np.array([[2., 0, 11], [0, 2, -3]])
    local = scan._crop_transform(full, (20, 30), (100, 80))
    np.testing.assert_allclose(local @ [60, 30, 1], [71, 37])
    image = np.zeros((80, 100), np.float32)
    image[30, 60] = 1
    warped = cv2.warpAffine(image, local, (160, 120), flags=cv2.INTER_NEAREST)
    assert warped[37, 71] == 1
    crop = scan._source_crop(np.array([full, [[1, 0, -7], [0, 1, 9]]]), [100, 80, 200, 160], (300, 400))
    assert all(value % 2 == 0 for value in crop)
    for matrix in (full, np.array([[1, 0, -7], [0, 1, 9]], np.float64)):
        for point in ([100, 80, 1], [200, 160, 1]):
            x, y = cv2.invertAffineTransform(matrix) @ point
            assert crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3]


def test_interpolation_preserves_exact_samples_and_clamps_endpoints():
    records = [{"frame": 2, "transform": [[1, 0, 4], [0, 1, -2]]},
               {"frame": 6, "transform": [[1, 0, 12], [0, 1, 6]]}]
    _, matrices, _, _, nearest = scan._interpolate(records, 9)
    np.testing.assert_array_equal(matrices[0], records[0]["transform"])
    np.testing.assert_array_equal(matrices[2], records[0]["transform"])
    np.testing.assert_array_equal(matrices[6], records[1]["transform"])
    np.testing.assert_array_equal(matrices[8], records[1]["transform"])
    np.testing.assert_array_equal(matrices[4], [[1, 0, 8], [0, 1, 2]])
    assert nearest[2] == 0 and nearest[6] == 1


def test_failed_decoder_never_publishes_complete_cache(tmp_path, video_fixture, monkeypatch):
    source, proxies, config = video_fixture
    monkeypatch.setattr(scan, "_timestamps", lambda *args: (list(np.arange(13) / 12), "test exact timeline"))
    source["path"] = str(tmp_path / "missing.mkv")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        scan.native_candidates([source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    directories = list((tmp_path / "cache" / "native-analysis").iterdir())
    assert len(directories) == 1
    assert not (directories[0] / "complete.json").exists()
    assert "No such file" in (directories[0] / "decode-0.log").read_text()


def test_truncated_score_cache_is_recomputed_and_polygon_limits_active_cells(tmp_path, video_fixture):
    source, proxies, config = video_fixture
    config.subject_polygon = ((.25, .25), (.75, .25), (.25, .75))
    records, scores, reference, active, directory = scan.native_candidates(
        [source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    expected = np.array(scores)
    assert active[2, 2] == 255 and active[5, 5] == 0
    with (directory / "scores.npy").open("r+b") as stream:
        stream.truncate(100)
    recovered = scan.native_candidates([source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    np.testing.assert_array_equal(recovered[1], expected)


def test_temporal_smoothing_never_crosses_clip_boundary(tmp_path, video_fixture):
    source, proxies, config = video_fixture
    second = {**source}
    other_proxies = [{**row, "source": 1} for row in proxies]
    _, scores, _, _, _ = scan.native_candidates(
        [source, second], proxies + other_proxies, 1, tmp_path / "cache", config, lambda **event: None)
    # Both clips are identical. Cross-clip convolution would contaminate the
    # strongly different endpoint focus values and break equality near the join.
    np.testing.assert_array_equal(scores[:13], scores[13:])


@pytest.mark.parametrize("corruption", ["nan_score", "negative_score", "wrong_reference", "nan_timestamp", "nan_transform", "empty_mask"])
def test_same_size_corrupt_cache_cannot_change_selection_or_reference(tmp_path, video_fixture, corruption):
    source, proxies, config = video_fixture
    expected_records, scores, reference, active, directory = scan.native_candidates(
        [source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    expected = np.array(scores)
    if corruption in ("nan_score", "negative_score"):
        damaged = np.load(directory / "scores.npy", mmap_mode="r+")
        damaged[3, 3, 3] = np.nan if corruption == "nan_score" else -1
        damaged.flush()
        del damaged
    elif corruption == "wrong_reference":
        metadata = json.loads((directory / "complete.json").read_text())
        metadata["reference_index"] = 0
        (directory / "complete.json").write_text(json.dumps(metadata))
    elif corruption == "empty_mask":
        np.save(directory / "active_mask.npy", np.zeros_like(active))
    else:
        records = json.loads((directory / "records.json").read_text())
        if corruption == "nan_timestamp":
            records[3]["timestamp"] = float("nan")
        else:
            records[3]["transform"][0][0] = float("nan")
        (directory / "records.json").write_text(json.dumps(records))
    recovered = scan.native_candidates([source], proxies, 1, tmp_path / "cache", config, lambda **event: None)
    assert recovered[0] == expected_records and recovered[2] == reference
    np.testing.assert_array_equal(recovered[1], expected)
    np.testing.assert_array_equal(recovered[3], active)


@pytest.mark.parametrize("metadata", ["{broken", '{"timestamp_basis":"sorted packet PTS","timestamps":[null]}'])
def test_damaged_proxy_timestamps_fall_back_to_real_decoded_timeline(tmp_path, video_fixture, metadata):
    source, proxies, _ = video_fixture
    (Path(proxies[0]["proxy"]).parent / "complete.json").write_text(metadata)
    times, basis = scan._timestamps(source, proxies)
    assert basis == "decoded-frame best_effort_timestamp_time"
    np.testing.assert_allclose(times, np.arange(13) / 12, atol=.0005)


@pytest.mark.parametrize("frame", [-1, 1.5, True, 9])
def test_invalid_proxy_indices_cannot_silently_interpolate_other_frames(frame):
    with pytest.raises(ValueError, match="frame indices"):
        scan._interpolate([{"frame": frame, "transform": [[1, 0, 0], [0, 1, 0]]}], 9)


@pytest.mark.parametrize('name',scan.INTEGRITY_FILES)
def test_valid_shape_native_mutation_is_rejected_and_rebuilt(tmp_path,video_fixture,name):
    source,proxies,config=video_fixture
    expected_records,scores,reference,active,directory=scan.native_candidates(
        [source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    expected=np.array(scores);expected_active=active.copy()
    expected_checksums={file:digest(directory/file) for file in scan.INTEGRITY_FILES}
    path=directory/name;before=path.stat()
    if name=='records.json':
        rows=json.loads(path.read_text());rows[3]['transform'][0][2]+=.125
        path.write_text(json.dumps(rows))
    else:
        array=np.load(path,mmap_mode='r+')
        if name=='active_mask.npy':array[3,3]=0
        else:array[3,3,3]+=1.
        array.flush();del array
    # Restore mtime to ensure a simplistic size+mtime cache cannot conceal this.
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    rebuilt=scan.native_candidates([source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    assert rebuilt[0]==expected_records and rebuilt[2]==reference
    np.testing.assert_array_equal(rebuilt[1],expected)
    np.testing.assert_array_equal(rebuilt[3],expected_active)
    assert {file:digest(directory/file) for file in scan.INTEGRITY_FILES}==expected_checksums
    integrity=json.loads((directory/'native-integrity.json').read_text())
    assert all(integrity['files'][file]['sha256']==digest(directory/file) for file in scan.INTEGRITY_FILES)


def test_legacy_adoption_is_explicit_once_and_does_not_claim_historic_integrity(tmp_path,video_fixture,monkeypatch):
    source,proxies,config=video_fixture
    original=scan.native_candidates([source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    directory=original[4];metadata=json.loads((directory/'complete.json').read_text())
    metadata.pop('integrity_schema');(directory/'complete.json').write_text(json.dumps(metadata))
    (directory/'native-integrity.json').unlink()
    monkeypatch.setattr(scan.subprocess,'Popen',lambda *a,**k:pytest.fail('Legacy adoption must not decode'))
    adopted=scan.native_candidates([source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    np.testing.assert_array_equal(adopted[1],original[1])
    integrity=json.loads((directory/'native-integrity.json').read_text())
    assert integrity['origin']=='legacy structural validation'
    assert 'pre-adoption corruption cannot be detected' in integrity['legacy_adoption_limitation']
    assert json.loads((directory/'complete.json').read_text())['integrity_schema']==1
    # Once recorded, deleting the sidecar is damage, not permission to adopt again.
    (directory/'native-integrity.json').unlink()
    assert scan._cached(directory,metadata['identity'],proxies[1]) is None


def test_native_integrity_fast_path_avoids_rehash_and_malformed_sidecar_is_miss(tmp_path,video_fixture,monkeypatch):
    source,proxies,config=video_fixture
    result=scan.native_candidates([source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    directory=result[4];metadata=json.loads((directory/'complete.json').read_text())
    monkeypatch.setattr(scan,'_hash',lambda *a:pytest.fail('Unchanged signatures must not rehash'))
    assert scan._cached(directory,metadata['identity'],proxies[1]) is not None
    (directory/'native-integrity.json').write_text('{broken')
    assert scan._cached(directory,metadata['identity'],proxies[1]) is None


def test_empty_native_array_is_a_cache_miss(tmp_path,video_fixture):
    source,proxies,config=video_fixture
    result=scan.native_candidates([source],proxies,1,tmp_path/'cache',config,lambda **event:None)
    directory=result[4];metadata=json.loads((directory/'complete.json').read_text())
    (directory/'scores.npy').write_bytes(b'')
    assert scan._cached(directory,metadata['identity'],proxies[1]) is None
