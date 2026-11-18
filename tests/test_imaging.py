"""Numerical acceptance checks using known geometry and focus content."""
import cv2
import numpy as np
import pytest

from macrostitch.imaging import (
    common_crop, focus_map, laplacian_pyramid, reconstruct, register,
    scale_transform, select_coverage, to_linear_bt709, to_srgb16,
)


def textured_image():
    rng = np.random.default_rng(7)
    image = np.full((300, 420), 40, np.uint8)
    for _ in range(130):
        x, y = rng.integers([25, 25], [395, 275])
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 10)),
                   int(rng.integers(70, 235)), -1)
    return cv2.GaussianBlur(image, (0, 0), .7)


@pytest.mark.parametrize("angle,scale,offset", [
    (1.5, 1.035, (3.4, -2.7)), (-2., .965, (-4.2, 3.1)),
])
def test_register_recovers_source_to_reference_similarity_with_defocus(angle, scale, offset):
    reference = textured_image()
    expected = cv2.getRotationMatrix2D((210, 150), angle, scale)
    expected[:, 2] += offset
    moving = cv2.warpAffine(reference, cv2.invertAffineTransform(expected),
                            (420, 300), borderMode=cv2.BORDER_REFLECT_101)
    moving = cv2.GaussianBlur(moving, (0, 0), 1.1)
    actual, metrics = register(reference, moving, roi=(.15, .15, .85, .85))
    landmarks = np.array([[40, 40, 1], [380, 40, 1], [210, 150, 1],
                          [40, 260, 1], [380, 260, 1]], np.float32)
    residual = np.linalg.norm(landmarks @ actual.T - landmarks @ expected.T, axis=1)
    assert residual.max() < .25
    assert metrics["ecc_converged"]
    assert metrics["correlation"] > .98


def test_ecc_roi_tracks_subject_in_moving_coordinates():
    """A reference-space inputMask must not make ECC inspect empty background."""
    rng = np.random.default_rng(123)
    reference = np.full((256, 384), 30, np.uint8)
    reference[48:208, 32:104] = rng.integers(40, 220, (160, 72), dtype=np.uint8)
    reference = cv2.GaussianBlur(reference, (0, 0), 1)
    expected = np.float32([[1, 0, -160], [0, 1, 0]])
    moving = cv2.warpAffine(reference, cv2.invertAffineTransform(expected),
                            (384, 256), borderMode=cv2.BORDER_REFLECT_101)
    moving = cv2.GaussianBlur(moving, (0, 0), .6)
    initial = expected.copy()
    initial[0, 2] += 2
    actual, metrics = register(reference, moving,
                               roi=(32/384, 48/256, 104/384, 208/256), initial=initial)
    center = np.array([228, 128, 1], np.float32)
    assert metrics["ecc_converged"]
    assert np.linalg.norm(actual @ center - expected @ center) < .25
    assert metrics["correlation"] > .95


def test_transform_scaling_preserves_normalized_landmarks():
    matrix = np.float32([[1.03, -.02, 3.25], [.02, 1.03, -2.4]])
    scaled = scale_transform(matrix, (300, 420), (1200, 1680))
    point = np.array([91.2, 157.8, 1])
    large_point = point * [4, 4, 1]
    np.testing.assert_allclose(scaled @ large_point, 4 * (matrix @ point), atol=2e-5)


def test_focus_score_prefers_sharp_content_over_defocus():
    image = textured_image()
    sharp = focus_map(image)[30:-30, 30:-30].mean()
    blurry = focus_map(cv2.GaussianBlur(image, (0, 0), 3))[30:-30, 30:-30].mean()
    assert sharp > blurry * 2


def test_coverage_keeps_complementary_planes_and_drops_exact_duplicate():
    # Vary tile contrast to model differently textured regions, independently of focus.
    contrast = np.linspace(.5, 2, 48, dtype=np.float32).reshape(6, 8)
    left = contrast.copy()
    left[:, 4:] *= .12
    right = contrast.copy()
    right[:, :4] *= .12
    blurred = contrast * .08
    scores = np.stack([left, left.copy(), right, blurred])
    chosen, info = select_coverage(scores, np.ones((6, 8), np.uint8), target=.999)
    assert len(chosen) == 2
    assert 2 in chosen
    assert len(set(chosen) & {0, 1}) == 1
    assert 3 not in chosen
    assert info["target_reached"]
    assert info["mean_coverage"] == pytest.approx(1)
    assert info["p10_coverage"] == pytest.approx(1)


def test_coverage_rejects_untextured_stack():
    with pytest.raises(ValueError, match="textured"):
        select_coverage(np.zeros((3, 6, 8), np.float32), np.ones((6, 8), np.uint8))


def test_coverage_accepts_equal_contrast_textured_tiles():
    # A regular pattern can have equally strong peak detail in every spatial tile.
    left = np.full((4, 6), .1, np.float32)
    left[:, :3] = 1
    right = np.fliplr(left).copy()
    chosen, info = select_coverage(np.stack([left, right]), np.ones((4, 6), np.uint8))
    assert set(chosen) == {0, 1}
    assert info["mean_coverage"] == pytest.approx(1)


def test_bt709_decode_and_srgb_encode_preserve_linear_light():
    # BT.709 and sRGB have different transfer curves: encoded values should change.
    linear = np.linspace(0, 1, 10001, dtype=np.float32)
    encoded_709 = np.where(linear < .018, 4.5 * linear, 1.099 * linear**.45 - .099)
    decoded = to_linear_bt709(np.rint(encoded_709 * 65535).astype(np.uint16))
    np.testing.assert_allclose(decoded, linear, atol=1.7e-5)
    encoded_srgb = to_srgb16(linear).astype(np.float64) / 65535
    recovered = np.where(encoded_srgb <= .04045, encoded_srgb / 12.92,
                         ((encoded_srgb + .055) / 1.055)**2.4)
    np.testing.assert_allclose(recovered, linear, atol=1.8e-5)
    assert to_srgb16(np.array([-1., 0., 1., 2.])).tolist() == [0, 0, 65535, 65535]


def legacy_bt709_decode(image):
    """Previous direct conversion is the exact-output compatibility oracle."""
    x = image.astype(np.float32) / 65535.
    return np.where(x < .081, x/4.5, ((x+.099)/1.099)**(1/.45)).astype(np.float32)


def test_bt709_lookup_matches_every_uint16_code_and_strided_images():
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    for image in (codes, codes.T, codes[::-1, ::-1], codes[::3, ::5]):
        actual = to_linear_bt709(image)
        assert actual.dtype == np.float32 and actual.shape == image.shape
        np.testing.assert_array_equal(actual, legacy_bt709_decode(image))
    # Callers own their decoded image and may change it in place.
    decoded = to_linear_bt709(codes)
    decoded[:] = -1
    np.testing.assert_array_equal(to_linear_bt709(codes), legacy_bt709_decode(codes))


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float32, np.float64, '>u2'])
def test_bt709_non_native_uint16_inputs_keep_previous_conversion(dtype):
    image = np.arange(256).reshape(16, 16).astype(dtype)
    np.testing.assert_array_equal(to_linear_bt709(image), legacy_bt709_decode(image))


def test_bt709_lookup_preserves_scalar_array_and_empty_shapes():
    for image in (np.array(42000, np.uint16), np.empty((0, 4, 3), np.uint16)):
        actual = to_linear_bt709(image)
        assert isinstance(actual, np.ndarray) and actual.shape == image.shape
        np.testing.assert_array_equal(actual, legacy_bt709_decode(image))


@pytest.mark.parametrize("levels", [1, 3, 6])
def test_pyramid_reconstructs_random_rgb_with_odd_dimensions(levels):
    image = np.random.default_rng(19).random((97, 131, 3), dtype=np.float32)
    np.testing.assert_allclose(reconstruct(laplacian_pyramid(image, levels)), image, atol=2e-7)


def test_pyramid_constant_rgb_has_no_detail_energy():
    image = np.empty((97, 131, 3), np.float32)
    image[:] = [.03, .22, .81]
    bands = laplacian_pyramid(image, 5)
    for band in bands[:-1]:
        # Several float32 filter accumulations can leave two rounding units.
        np.testing.assert_allclose(band, 0, atol=2e-7)
    np.testing.assert_allclose(reconstruct(bands), image, atol=1e-7)


def brute_force_crop_area(mask):
    h, w = mask.shape
    return max([0] + [(x1-x0)*(y1-y0)
        for y0 in range(h) for y1 in range(y0+1, h+1)
        for x0 in range(w) for x1 in range(x0+1, w+1)
        if mask[y0:y1, x0:x1].all()])


def test_common_crop_is_valid_and_maximal_against_exhaustive_oracle():
    rng = np.random.default_rng(17)
    masks = [np.ones((5, 7), np.uint8), np.eye(5, dtype=np.uint8)]
    masks += [(rng.random((5, 7)) > .3).astype(np.uint8) for _ in range(12)]
    for mask in masks:
        x0, y0, x1, y1 = common_crop(mask)
        assert 0 <= x0 < x1 <= mask.shape[1]
        assert 0 <= y0 < y1 <= mask.shape[0]
        assert mask[y0:y1, x0:x1].all()
        assert (x1-x0)*(y1-y0) == brute_force_crop_area(mask)


def test_common_crop_rejects_empty_intersection():
    with pytest.raises(ValueError, match="no common valid region"):
        common_crop(np.zeros((8, 12), np.uint8))


def test_fusion_preserves_constant_linear_light_and_common_crop(tmp_path):
    from macrostitch.pipeline import Config, fuse

    image = np.empty((64, 96, 3), np.uint16)
    image[:] = [13000, 28000, 47000]
    source = tmp_path / "source.tiff"
    assert cv2.imwrite(str(source), image)
    valid = np.zeros(image.shape[:2], np.uint8)
    valid[5:-5, 7:-7] = 1
    output = tmp_path / "output"
    output.mkdir()
    result = fuse([source], 0, valid, Config(pyramid_levels=4), output, lambda **event: None)
    assert result["crop_xyxy"] == [7, 5, 89, 59]
    merged = cv2.imread(str(output / "stack.tiff"), cv2.IMREAD_UNCHANGED)
    expected = to_srgb16(to_linear_bt709(image[5:-5, 7:-7]))
    assert merged.dtype == np.uint16
    np.testing.assert_allclose(merged, expected, rtol=0, atol=1)


def test_fusion_recovers_complementary_focused_regions(tmp_path):
    from macrostitch.pipeline import Config, fuse

    rng = np.random.default_rng(29)
    sharp = cv2.GaussianBlur(rng.uniform(.1, .9, (128, 192, 3)).astype(np.float32), (0, 0), .6)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    planes = [sharp.copy(), sharp.copy()]
    planes[0][:, 96:] = blurred[:, 96:]
    planes[1][:, :96] = blurred[:, :96]
    paths = []
    for i, plane in enumerate(planes):
        directory = tmp_path / str(i)
        directory.mkdir()
        path = directory / "aligned.tiff"
        assert cv2.imwrite(str(path), np.rint(plane * 65535).astype(np.uint16))
        paths.append(path)
    output = tmp_path / "output"
    output.mkdir()
    fuse(paths, 0, np.ones(sharp.shape[:2], np.uint8), Config(pyramid_levels=4),
         output, lambda **event: None)
    fused = cv2.imread(str(output / "stack.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float64)
    expected = to_srgb16(to_linear_bt709(np.rint(sharp * 65535).astype(np.uint16))).astype(np.float64)
    baseline = to_srgb16(to_linear_bt709(np.rint(planes[0] * 65535).astype(np.uint16))).astype(np.float64)
    # Evaluate away from focal boundaries and filter support at image margins.
    right = np.s_[24:-24, 120:-24]
    fused_error = np.mean((fused[right] - expected[right])**2)
    baseline_error = np.mean((baseline[right] - expected[right])**2)
    assert fused_error < baseline_error * .15
    left = np.s_[24:-24, 24:72]
    assert np.mean((fused[left] - expected[left])**2) < baseline_error * .15
