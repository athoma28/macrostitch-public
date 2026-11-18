"""Native-pixel provenance, bounded seams, and numerical transparent exports."""
import json
import struct

import cv2
import numpy as np
import pytest

from macrostitch.imaging import common_crop, to_linear_bt709, to_srgb16
from macrostitch.pipeline import Config
from macrostitch.regional import band_weights, fuse_regional, matte, write_transparent


POLYGON = ((.1, .1), (.9, .1), (.9, .9), (.1, .9))
REGIONS = ({"y": (0, .5), "roi": (.2, .2, .8, .4)},
           {"y": (.5, 1), "roi": (.2, .6, .8, .8)})


def regional_config(**kwargs):
    values = dict(fusion_mode="regional", background_mode="transparent",
                  subject_polygon=POLYGON, focus_regions=REGIONS, region_blend_px=3)
    values.update(kwargs)
    config = Config(**values)
    config.validate()
    return config


def sources(tmp_path):
    rng = np.random.default_rng(711)
    sharp = rng.integers(8000, 54000, (128, 128, 3), dtype=np.uint16)
    sharp = cv2.GaussianBlur(sharp, (0, 0), .6)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    first, second = sharp.copy(), sharp.copy()
    first[64:] = blurred[64:]
    second[:64] = blurred[:64]
    paths = [tmp_path / "first.tiff", tmp_path / "second.tiff"]
    for path, image in zip(paths, (first, second)):
        assert cv2.imwrite(str(path), image)
    records = [{"source": 0, "frame": i, "timestamp": float(i)} for i in range(2)]
    return paths, (first, second), records


def run_fusion(tmp_path, config, common=None):
    paths, images, records = sources(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    if common is None:
        common = np.ones((128, 128), np.uint8)
    result = fuse_regional(paths, 0, common, config, output, lambda **event: None, records)
    return output, result, images


@pytest.mark.parametrize("width", [0, .25, 3, 32])
def test_bands_are_nonnegative_partition_even_when_seams_overlap(width):
    regions = [{"y": (0, .2)}, {"y": (.2, .25)}, {"y": (.25, 1)}]
    weights = band_weights(80, regions, width)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
    np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-7)
    # This guarantees no seam overshoot for any convex combination of sources.
    values = np.array([.1, .8, .35])[:, None]
    mixed = (weights * values).sum(axis=0)
    assert mixed.min() >= .1 and mixed.max() <= .8


def test_detail_winners_and_exact_native_pixels_outside_seams(tmp_path):
    config = regional_config()
    output, result, images = run_fusion(tmp_path, config)
    metadata = json.loads((output / "fusion.json").read_text())
    assert [r["source_index"] for r in metadata["regions"]] == [0, 1]
    assert all(r["relative_score_margin"] > .5 for r in metadata["regions"])
    assert all(r["linear_tone_gain"] == 1 for r in metadata["regions"])
    actual = cv2.imread(str(output / "stack.png"), cv2.IMREAD_UNCHANGED)
    assert actual.dtype == np.uint16 and actual.shape == (128, 128, 4)
    expected_sources = [to_srgb16(to_linear_bt709(image)) for image in images]
    np.testing.assert_array_equal(actual[25:55, 25:100, :3], expected_sources[0][25:55, 25:100])
    np.testing.assert_array_equal(actual[75:100, 25:100, :3], expected_sources[1][75:100, 25:100])
    weights = band_weights(128, config.focus_regions, config.region_blend_px)
    expected = to_srgb16(sum(to_linear_bt709(im) * weights[i, :, None, None]
                            for i, im in enumerate(images)))
    np.testing.assert_array_equal(actual[60:68, 25:100, :3], expected[60:68, 25:100])
    assert result["diagnostics"]["preclip_out_of_range_fraction"] == 0
    alpha = actual[..., 3]
    assert alpha.min() == 0 and alpha.max() == 65535
    assert np.any((alpha > 0) & (alpha < 65535))
    assert not actual[alpha == 0, :3].any()
    np.testing.assert_array_equal(alpha, cv2.imread(str(output / "matte.png"), cv2.IMREAD_UNCHANGED))
    np.testing.assert_array_equal(actual, cv2.imread(str(output / "stack.tiff"), cv2.IMREAD_UNCHANGED))


def test_matte_has_finite_support_and_inward_erosion():
    original = matte((100, 100), POLYGON, feather=1.5, erode=0)
    eroded = matte((100, 100), POLYGON, feather=1.5, erode=1)
    assert np.isfinite(original).all()
    assert 0 <= eroded.min() <= eroded.max() <= 1
    assert np.all(eroded <= original)
    assert eroded[50, 50] == 1
    assert not eroded[:8].any() and not eroded[:, :8].any()
    assert not eroded[93:].any() and not eroded[:, 93:].any()


def tiff_extra_samples(path):
    """Inspect the small classic TIFF IFD without a second image dependency."""
    data = path.read_bytes()
    order = "<" if data[:2] == b"II" else ">"
    assert struct.unpack_from(order + "H", data, 2)[0] == 42
    offset = struct.unpack_from(order + "I", data, 4)[0]
    count = struct.unpack_from(order + "H", data, offset)[0]
    for entry in range(count):
        start = offset + 2 + entry * 12
        tag, kind, length = struct.unpack_from(order + "HHI", data, start)
        if tag == 338:
            assert kind == 3 and length == 1
            return struct.unpack_from(order + "H", data, start + 8)[0]
    return None


def test_straight_alpha_preserves_color_and_roundtrips_numerically(tmp_path):
    rgb = np.full((4, 5, 3), (12345, 34567, 54321), np.uint16)
    alpha = np.broadcast_to(np.array([0, .1, .5, .9, 1], np.float32), (4, 5))
    result = {}
    write_transparent(tmp_path, rgb, alpha, result)
    for filename in ("stack.png", "stack.tiff", "subject.png"):
        actual = cv2.imread(str(tmp_path / filename), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual[..., 3], np.rint(alpha * 65535).astype(np.uint16))
        np.testing.assert_array_equal(actual[:, 1:, :3], rgb[:, 1:])
        assert not actual[:, 0, :3].any()
    assert tiff_extra_samples(tmp_path / "stack.tiff") == 2, "TIFF must declare unassociated alpha"


def test_reference_background_keeps_source_and_blends_only_at_matte(tmp_path):
    config = regional_config(background_mode="reference")
    output, result, images = run_fusion(tmp_path, config)
    actual = cv2.imread(str(output / "stack.png"), cv2.IMREAD_UNCHANGED)
    alpha = matte((128, 128), POLYGON, config.matte_feather_px, config.matte_erode_px)
    linear = [to_linear_bt709(im) for im in images]
    weights = band_weights(128, config.focus_regions, config.region_blend_px)
    mixed = sum(im * weights[i, :, None, None] for i, im in enumerate(linear))
    expected = to_srgb16(mixed * alpha[..., None] + linear[0] * (1 - alpha[..., None]))
    assert actual.shape == (128, 128, 3)
    np.testing.assert_array_equal(actual, expected)


def test_reference_comparison_handles_subject_outside_largest_common_rectangle(tmp_path):
    common = np.ones((128, 128), np.uint8)
    common[:64, :20] = 0
    config = regional_config(background_mode="reference", subject_polygon=((0, 0), (1, 0), (1, 1), (0, 1)))
    output, result, _ = run_fusion(tmp_path, config, common)
    x0, y0, x1, y1 = common_crop(common)
    actual = cv2.imread(str(output / "stack.png"), cv2.IMREAD_UNCHANGED)
    assert actual.shape == (y1-y0, x1-x0, 3)
    comparison = cv2.imread(str(output / "comparison.jpg"))
    assert comparison is not None and comparison.shape == (y1-y0, 2*(x1-x0), 3)


def test_textureless_score_ties_are_ambiguous(tmp_path):
    paths, _, records = sources(tmp_path)
    for path in paths:
        assert cv2.imwrite(str(path), np.full((128, 128, 3), 25000, np.uint16))
    output = tmp_path / "flat"
    output.mkdir()
    result = fuse_regional(paths, 0, np.ones((128, 128), np.uint8), regional_config(),
                           output, lambda **event: None, records)
    metadata = json.loads((output / "fusion.json").read_text())
    assert all(region["relative_score_margin"] == 0 for region in metadata["regions"])
    assert result["diagnostics"]["ambiguous_foreground_fraction"] == 1


def test_explicit_tone_gain_is_bounded_reported_and_applied_in_linear_light(tmp_path):
    paths, images, records = sources(tmp_path)
    first = images[0].copy()
    first[64:] = np.rint(first[64:] * .6).astype(np.uint16)
    assert cv2.imwrite(str(paths[0]), first)
    regions = (REGIONS[0], {**REGIONS[1], "tone_roi": (.2, .6, .8, .8), "tone_reference": 0})
    config = regional_config(focus_regions=regions)
    output = tmp_path / "tone"
    output.mkdir()
    fuse_regional(paths, 0, np.ones((128, 128), np.uint8), config, output,
                  lambda **event: None, records)
    metadata = json.loads((output / "fusion.json").read_text())
    assert [r["source_index"] for r in metadata["regions"]] == [0, 1]
    second_region = metadata["regions"][1]
    assert second_region["unclamped_tone_gain"] < .8
    assert second_region["linear_tone_gain"] == .8
    actual = cv2.imread(str(output / "stack.png"), cv2.IMREAD_UNCHANGED)
    expected = to_srgb16(to_linear_bt709(images[1]) * .8)
    np.testing.assert_array_equal(actual[75:100, 25:100, :3], expected[75:100, 25:100])


@pytest.mark.parametrize("changes", [
    {"region_blend_px": float("nan")}, {"matte_feather_px": 0}, {"matte_erode_px": -1},
    {"subject_polygon": None}, {"focus_regions": None}, {"focus_regions": []},
    {"focus_regions": [{"y": (0, .4), "roi": (.2, .2, .8, .4)}]},
    {"focus_regions": [{"y": (0, 1), "roi": (0, 0, 2, 1)}]},
    {"focus_regions": [{"y": (0, 1), "roi": (0, 0, 1, 1), "tone_reference": 1}]},
    {"subject_polygon": ((0, 0), (1, float("nan")), (1, 1))},
])
def test_invalid_regional_configuration_rejected(changes):
    with pytest.raises(ValueError):
        regional_config(**changes)
