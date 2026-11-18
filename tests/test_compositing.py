"""User mask controls preserve source colors, output coordinates and straight alpha."""
import cv2
import numpy as np
import pytest
import tifffile

from macrostitch.pipeline import Config, fuse
from macrostitch.regional import fuse_regional
from macrostitch.spatial import fuse_spatial
from macrostitch.imaging import to_linear_bt709, to_srgb16


SUBJECT = ((.30, .30), (.65, .30), (.65, .75), (.30, .75))


def test_srgb_lookup_preserves_all_uint16_values_and_array_ownership():
    from macrostitch.compositing import _srgb_linear
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    for image in (codes, codes.T, codes[::-1, ::-1], codes[::3, ::5]):
        encoded = image.astype(np.float32)/65535
        expected = np.where(encoded <= .04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
        actual = _srgb_linear(image)
        assert actual.dtype == np.float32 and actual.shape == image.shape
        np.testing.assert_array_equal(actual, expected)
        actual[:] = -1
        np.testing.assert_array_equal(_srgb_linear(image), expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float32, np.float64, '>u2'])
def test_srgb_other_dtypes_keep_direct_conversion(dtype):
    from macrostitch.compositing import _srgb_linear
    image = np.arange(256).reshape(16, 16).astype(dtype)
    encoded = image.astype(np.float32)/65535
    expected = np.where(encoded <= .04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
    np.testing.assert_array_equal(_srgb_linear(image), expected)


def test_srgb_lookup_preserves_scalar_array_and_empty_shapes():
    from macrostitch.compositing import _srgb_linear
    for image in (np.array(42000, np.uint16), np.empty((0, 4, 3), np.uint16)):
        actual = _srgb_linear(image)
        encoded = image.astype(np.float32)/65535
        expected = np.where(encoded <= .04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
        assert isinstance(actual, np.ndarray) and actual.shape == image.shape
        np.testing.assert_array_equal(actual, expected)


def rectangle(name, action, x0, y0, x1, y1):
    return dict(name=name, action=action,
                polygon=((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


@pytest.fixture
def scene(tmp_path):
    sharp = np.full((384, 512, 3), 14000, np.uint16)
    # Sparse isolated high-contrast structure supplies a real noise-floor region.
    for x in (232, 412):
        for y in (204,):
            cv2.circle(sharp, (x, y), 4, (53000, 41000, 49000), -1)
    sharp = cv2.GaussianBlur(sharp, (0, 0), .5)
    reference = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    paths = []
    for index, image in enumerate((reference, sharp)):
        directory = tmp_path / f'frame-{index}'
        directory.mkdir()
        path = directory / 'aligned.tiff'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    common = np.zeros((384, 512), np.uint8)
    common[20:-20, 28:-28] = 1
    records = [dict(source=0, frame=i, timestamp=float(i)) for i in range(2)]
    return paths, common, to_srgb16(to_linear_bt709(reference)), records


def render(tmp_path, scene, method='pyramid', canvas='reference', **options):
    from macrostitch.compositing import compose_output
    paths, common, reference, records = scene
    values = dict(fusion_mode=method, canvas_mode=canvas, subject_polygon=SUBJECT,
                  roi=(.3, .3, .65, .75), pyramid_levels=2,
                  focus_regions=({'y': (0, 1), 'roi': (.35, .35, .6, .7)},)
                  if method == 'regional' else None)
    values.update(options)
    config = Config(**values)
    config.validate()
    output = tmp_path / 'out'
    output.mkdir()
    if method == 'pyramid':
        result = fuse(paths, 0, common, config, output, lambda **event: None)
    else:
        function = fuse_regional if method == 'regional' else fuse_spatial
        result = function(paths, 0, common, config, output, lambda **event: None, records)
    compose_output(output, paths, 0, common, config, result)
    image = cv2.imread(str(output / 'stack.png'), cv2.IMREAD_UNCHANGED)
    return image, output, result


@pytest.mark.parametrize('method', ['pyramid', 'regional', 'spatial'])
@pytest.mark.parametrize('canvas', ['common', 'reference'])
def test_opacity_is_straight_alpha_and_crop_coordinates_are_preserved(tmp_path, scene, method, canvas):
    image, output, result = render(tmp_path, scene, method, canvas, background_opacity=.5)
    x0, y0, x1, y1 = result['crop_xyxy']
    assert image.shape == (y1-y0, x1-x0, 4)
    assert (x0 > 0 and y0 > 0) if canvas == 'common' else (x0, y0) == (0, 0)
    # This pixel is valid background, safely away from all feather support.
    y, x = 60-y0, 72-x0
    np.testing.assert_array_equal(image[y, x, :3], scene[2][60, 72])
    assert abs(int(image[y, x, 3])-32768) <= 1
    assert image[192-y0, 232-x0, 3] == 65535
    # Straight alpha reproduces the expected composite on both dark and white grounds.
    rgb = image[y, x, :3].astype(float)/65535
    alpha = float(image[y, x, 3])/65535
    for ground in (0., 1.):
        np.testing.assert_allclose(rgb*alpha + ground*(1-alpha),
                                   scene[2][60, 72]/65535*alpha + ground*(1-alpha))
    matte = cv2.imread(str(output/'matte.png'), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(matte, image[..., 3])
    with tifffile.TiffFile(output/'stack.tiff') as tiff:
        assert int(tiff.pages[0].extrasamples[0]) == 2  # unassociated alpha
        np.testing.assert_array_equal(tiff.asarray()[..., 3], image[..., 3])


@pytest.mark.parametrize('method', ['pyramid', 'regional', 'spatial'])
@pytest.mark.parametrize('reverse', [False, True])
def test_reference_and_exclude_override_stack_independent_of_order(tmp_path, scene, method, reverse):
    masks = [rectangle('stack', 'stack', .2, .2, .85, .85),
             rectangle('protect', 'reference', .35, .35, .7, .7),
             rectangle('remove', 'exclude', .45, .45, .6, .6)]
    image, output, result = render(tmp_path, scene, method, background_opacity=1,
                                   object_masks=tuple(reversed(masks)) if reverse else tuple(masks))
    np.testing.assert_array_equal(image[152, 196, :3], scene[2][152, 196])
    assert image[152, 196, 3] == 65535
    np.testing.assert_array_equal(image[200, 260], np.zeros(4, np.uint16))
    labels = cv2.imread(str(output/'source_index.png'), cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(str(output/'confidence.png'), cv2.IMREAD_UNCHANGED)
    assert labels[152, 196] == 0 and confidence[152, 196] == 0
    assert labels[200, 260] == 65535 and confidence[200, 260] == 0


def test_background_margin_retains_reference_context_without_darkening(tmp_path, scene):
    image, _, _ = render(tmp_path, scene, background_opacity=0, background_margin_px=12)
    # Subject starts at x=154; this is retained native-pixel context, not foreground.
    assert image[192, 145, 3] == 65535
    np.testing.assert_array_equal(image[192, 145, :3], scene[2][192, 145])
    assert image[192, 60, 3] == 0
    assert not image[192, 60, :3].any()


def test_stack_masks_union_subject_and_replace_inferred_roi():
    from macrostitch.compositing import foreground_matte
    mask = rectangle('second', 'stack', .75, .2, .95, .8)
    config = Config(subject_polygon=SUBJECT, object_masks=(mask,))
    config.validate()
    alpha = foreground_matte((100, 100), config)
    assert alpha[50, 50] == 1 and alpha[50, 85] == 1
    config = Config(roi=(.1, .1, .3, .3), object_masks=(mask,))
    config.validate()
    alpha = foreground_matte((100, 100), config)
    assert alpha[20, 20] == 0 and alpha[50, 85] == 1


@pytest.mark.parametrize('options', [
    {'background_opacity': -.1}, {'background_opacity': 1.1},
    {'background_opacity': float('nan')}, {'background_opacity': float('inf')},
    {'background_margin_px': -1}, {'background_margin_px': 513},
    {'background_margin_px': float('nan')},
    {'object_masks': (rectangle('bad', 'unknown', .1, .1, .9, .9),)},
    {'object_masks': (rectangle('same', 'stack', .1, .1, .4, .4),
                      rectangle('same', 'reference', .5, .5, .9, .9))},
    {'object_masks': ({'name': 'line', 'action': 'stack', 'polygon': ((.1,.1),(.2,.2),(.3,.3))},)},
    {'object_masks': ({'name': 'cross', 'action': 'stack', 'polygon': ((.1,.1),(.9,.9),(.1,.9),(.9,.1))},)},
    {'object_masks': (rectangle('outside', 'stack', -.1, .1, .9, .9),)},
])
def test_invalid_compositing_config_rejected(options):
    with pytest.raises(ValueError):
        Config(**options).validate()


def test_legacy_defaults_do_not_activate_controls():
    from macrostitch.compositing import controls_active
    assert not controls_active(Config())
    assert not controls_active(Config(background_mode='transparent', subject_polygon=SUBJECT))
    assert controls_active(Config(background_opacity=1))


@pytest.mark.parametrize('method', ['pyramid', 'regional', 'spatial'])
def test_stack_object_outside_subject_receives_fused_detail(tmp_path, scene, method):
    mask = rectangle('extra', 'stack', .73, .25, .92, .8)
    image, _, _ = render(tmp_path, scene, method, background_opacity=0, object_masks=(mask,))
    patch = image[140:260, 400:448]
    assert np.all(patch[..., 3] == 65535)
    # The sharp source differs substantially from the blurry reference here.
    assert np.mean(np.abs(patch[..., :3].astype(float)-scene[2][140:260, 400:448])) > 100


def test_legacy_compositor_is_byte_preserving_noop(tmp_path, scene):
    from macrostitch.compositing import compose_output
    output = tmp_path/'legacy'
    output.mkdir()
    for name in ('stack.png', 'stack.tiff', 'preview.jpg', 'source_index.png'):
        (output/name).write_bytes(b'legacy output bytes')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    result = {'crop_xyxy': [0, 0, 512, 384]}
    compose_output(output, scene[0], 0, scene[1], Config(), result)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert result == {'crop_xyxy': [0, 0, 512, 384]}


@pytest.mark.parametrize('method', ['pyramid', 'regional', 'spatial'])
def test_all_excluded_is_valid_empty_transparent_output(tmp_path, scene, method):
    mask = rectangle('everything', 'exclude', 0, 0, 1, 1)
    image, output, result = render(tmp_path, scene, method, object_masks=(mask,))
    assert image.shape == (384, 512, 4)
    assert not image.any()
    assert result['compositing']['all_transparent'] is True
    labels = cv2.imread(str(output/'source_index.png'), cv2.IMREAD_UNCHANGED)
    assert np.all(labels == 65535)
    assert cv2.imread(str(output/'subject.png'), cv2.IMREAD_UNCHANGED).shape == image.shape


@pytest.mark.parametrize('method', ['pyramid', 'regional', 'spatial'])
def test_stack_mask_supplies_transparent_subject_without_legacy_polygon(tmp_path, scene, method):
    mask = rectangle('chosen', 'stack', .3, .3, .65, .75)
    image, _, _ = render(tmp_path, scene, method, subject_polygon=None,
                          background_mode='transparent', object_masks=(mask,))
    assert image[192, 232, 3] == 65535
    assert not image[60, 72].any()


@pytest.mark.parametrize('opacity,foreground,protected,expected_label', [
    (0., .4, 0., 1),
    (1., .6, .3, 0),
])
def test_fractional_matte_labels_follow_actual_color_contribution(
        tmp_path, monkeypatch, opacity, foreground, protected, expected_label):
    from macrostitch import compositing
    shape = (12, 16)
    reference = np.full((*shape, 3), 12000, np.uint16)
    fused = np.full((*shape, 3), 48000, np.uint16)
    reference_path = tmp_path / 'reference.tiff'
    assert cv2.imwrite(str(reference_path), reference)
    output = tmp_path / 'output'
    output.mkdir()
    assert cv2.imwrite(str(output / 'stack.png'), fused)
    assert cv2.imwrite(str(output / 'source_index.png'), np.ones(shape, np.uint16))
    assert cv2.imwrite(str(output / 'confidence.png'), np.full(shape, 60000, np.uint16))
    monkeypatch.setattr(compositing, 'foreground_matte',
                        lambda *args, **kwargs: np.full(shape, foreground, np.float32))
    monkeypatch.setattr(compositing, '_action_matte',
                        lambda shape, config, action: np.full(
                            shape, protected if action == 'reference' else 0., np.float32))
    config = Config(background_opacity=opacity)
    result = {'crop_xyxy': [0, 0, shape[1], shape[0]]}
    compositing.compose_output(output, [reference_path], 0, np.ones(shape, np.uint8), config, result)
    labels = cv2.imread(str(output / 'source_index.png'), cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(str(output / 'confidence.png'), cv2.IMREAD_UNCHANGED)
    assert np.all(labels == expected_label)
    assert np.all(confidence > 0) if expected_label == 1 else np.all(confidence == 0)
    if opacity == 0:
        final = cv2.imread(str(output / 'stack.png'), cv2.IMREAD_UNCHANGED)
        np.testing.assert_allclose(final[..., :3], fused, atol=1)
        np.testing.assert_allclose(final[..., 3], round(foreground * 65535), atol=1)
