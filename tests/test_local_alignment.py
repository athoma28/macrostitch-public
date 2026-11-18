import cv2
import numpy as np
import pytest

from macrostitch.local_alignment import (
    TranslationConfig, estimate_local_translations, resample_local_correction,
)


def fixture():
    rng = np.random.default_rng(731)
    image = cv2.GaussianBlur(rng.random((240, 300), dtype=np.float32), (0, 0), 1.)
    moving = cv2.warpAffine(image, np.array([[1, 0, 2.3], [0, 1, -3.1]], np.float32),
                            (300, 240), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    records = [{'source': 0, 'frame': i, 'full_transform': np.eye(2, 3).tolist()} for i in range(2)]
    return records, {0: image, 1: moving}


def estimate(records, crops, **kwargs):
    return estimate_local_translations(
        records, crops, roi=(10, 10, 290, 230), fit_roi=(40, 45, 130, 190),
        validation_rois=[(160, 45, 255, 110), (160, 130, 255, 190)],
        anchor={'source': 0, 'frame': 0}, encoding='linear', **kwargs)


def test_translation_accepted_only_after_independent_validation_and_fingerprinted():
    records, crops = fixture()
    result = estimate(records, crops, source_hashes={0: 'original-video-sha'})
    c, = result['corrections']
    np.testing.assert_allclose(c['sample_shift_xy'], (2.3, -3.1), atol=.08)
    assert all(m['improvement'] > .5 for m in c['measurements'])
    np.testing.assert_allclose(np.array(c['core_full_transform'])[:, 2], (-2.3, 3.1), atol=.08)
    assert result['provenance']['inputs']['1']['source_sha256'] == 'original-video-sha'
    assert len(result['provenance_sha256']) == 64
    other = estimate(records, crops, config=TranslationConfig(min_improvement=.01))
    assert other['provenance_sha256'] != result['provenance_sha256']


@pytest.mark.parametrize('kind', ['unchanged', 'blurred', 'outlier', 'heldout_disagrees', 'flat'])
def test_rejects_unhelpful_or_unsafe_corrections(kind):
    records, crops = fixture()
    if kind == 'unchanged':
        crops[1] = crops[0].copy()
    elif kind == 'blurred':
        crops[1] = cv2.GaussianBlur(crops[1], (0, 0), 8.)
    elif kind == 'outlier':
        crops[1] = cv2.warpAffine(crops[0], np.array([[1, 0, 8], [0, 1, -3]], np.float32), (300, 240))
    elif kind == 'heldout_disagrees':
        crops[1][:, 150:] = crops[0][:, 150:]
    else:
        crops = {0: np.full((240, 300), .5, np.float32), 1: np.full((240, 300), .6, np.float32)}
    result = estimate(records, crops)
    assert not result['corrections']
    assert not result['diagnostics'][1]['accepted']
    if kind == 'heldout_disagrees':
        assert result['diagnostics'][1]['reason'] == 'heldout_not_improved'
    if kind in ('blurred', 'flat'):
        assert result['diagnostics'][1]['reason'] == 'insufficient_structure'
    if kind == 'unchanged':
        assert result['diagnostics'][1]['reason'] == 'no_material_shift'


def test_labels_limit_contributors_and_heldout_regions_cannot_overlap_fit():
    records, crops = fixture()
    result = estimate(records, crops, labels=np.zeros((240, 300), np.uint16))
    assert result['diagnostics'][1]['reason'] == 'no_contributing_footprint'
    with pytest.raises(ValueError, match='held out'):
        estimate_local_translations(records, crops, roi=(0, 0, 300, 240),
                                    fit_roi=(40, 40, 150, 190), validation_rois=[(50, 50, 100, 150)],
                                    anchor=0, encoding='linear')
    with pytest.raises(ValueError, match='six'):
        estimate(records, crops, config=TranslationConfig(max_shift_px=7))


def test_original_single_resample_composition_and_unchanged_exterior():
    records, crops = fixture()
    c, = estimate(records, crops)['corrections']
    # A nonidentity original→reference transform catches incorrect composition.
    original = np.arange(240*300, dtype=np.float32).reshape(240, 300)
    full = np.array([[1.03, -.02, 4], [.02, 1.03, -3]], np.float64)
    c['original_full_transform'] = full.tolist()
    tile = resample_local_correction(original, c, interpolation=cv2.INTER_LINEAR)
    inverse = cv2.invertAffineTransform(full)
    yy, xx = np.mgrid[10:230, 10:290].astype(np.float64)
    dx, dy = c['sample_shift_xy']
    expected = cv2.remap(original,
                        (inverse[0, 0]*(xx+dx)+inverse[0, 1]*(yy+dy)+inverse[0, 2]).astype(np.float32),
                        (inverse[1, 0]*(xx+dx)+inverse[1, 1]*(yy+dy)+inverse[1, 2]).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    np.testing.assert_array_equal(tile[30:-30, 30:-30], expected[30:-30, 30:-30])
    base = cv2.remap(original,
                    (inverse[0, 0]*xx+inverse[0, 1]*yy+inverse[0, 2]).astype(np.float32),
                    (inverse[1, 0]*xx+inverse[1, 1]*yy+inverse[1, 2]).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    np.testing.assert_array_equal(tile[0], base[0])
    np.testing.assert_array_equal(tile[-1], base[-1])
    np.testing.assert_array_equal(tile[:, 0], base[:, 0])
    np.testing.assert_array_equal(tile[:, -1], base[:, -1])
    output = original.copy(); output[10:230, 10:290] = tile
    np.testing.assert_array_equal(output[:10], original[:10])
    np.testing.assert_array_equal(output[:, :10], original[:, :10])


def test_crop_origin_is_reference_space_and_input_pixels_change_fingerprint():
    records, crops = fixture()
    a = estimate(records, crops)
    b = estimate_local_translations(records, crops, roi=(110, 210, 390, 430),
                                   fit_roi=(140, 245, 230, 390),
                                   validation_rois=[(260, 245, 355, 310), (260, 330, 355, 390)],
                                   anchor=0, crop_origin=(100, 200), encoding='linear')
    np.testing.assert_allclose(a['corrections'][0]['sample_shift_xy'], b['corrections'][0]['sample_shift_xy'])
    crops[1] = crops[1].copy(); crops[1][0, 0] += .01
    changed = estimate(records, crops)
    assert changed['provenance']['inputs']['1']['cached_crop_sha256'] != a['provenance']['inputs']['1']['cached_crop_sha256']


def test_provenance_includes_label_footprints_and_source_geometry():
    records, crops = fixture()
    labels = np.ones((240, 300), np.uint16)
    a = estimate(records, crops, labels=labels)
    labels[20, 20] = 0
    b = estimate(records, crops, labels=labels)
    assert a['provenance_sha256'] != b['provenance_sha256']
    records[1]['full_transform'][0][2] = 1.
    c = estimate(records, crops, labels=labels)
    assert c['provenance_sha256'] != b['provenance_sha256']
    records[1]['full_transform'] = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match='invertible'):
        estimate(records, crops)
