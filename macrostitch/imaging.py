"""Numerical primitives. Warps always map source coordinates into reference coordinates."""
from __future__ import annotations
from functools import lru_cache
import cv2
import numpy as np


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def focus_map(image):
    """Locally averaged, noise-suppressed two-scale detail energy."""
    g = gray(image).astype(np.float32)
    if g.max() > 1.5:
        g /= 65535.0 if image.dtype == np.uint16 else 255.0
    fine = cv2.GaussianBlur(g, (0, 0), .65)
    mid = cv2.GaussianBlur(g, (0, 0), 1.6)
    coarse = cv2.GaussianBlur(g, (0, 0), 3.2)
    energy = (fine - mid) ** 2 + .35 * (mid - coarse) ** 2
    return cv2.GaussianBlur(energy, (0, 0), 2.2)


def roi_mask(shape, roi):
    h, w = shape[:2]
    x0, y0, x1, y1 = roi or (.1, .1, .9, .85)
    mask = np.zeros((h, w), np.uint8)
    mask[round(y0*h):round(y1*h), round(x0*w):round(x1*w)] = 255
    return mask


def scale_transform(matrix, source_shape, destination_shape):
    """Conjugate a transform when both images are resized to a different grid."""
    sy = destination_shape[0] / source_shape[0]
    sx = destination_shape[1] / source_shape[1]
    s = np.diag([sx, sy, 1.])
    m = np.eye(3)
    m[:2] = matrix
    return (s @ m @ np.linalg.inv(s))[:2].astype(np.float32)


def register(reference, moving, roi=None, initial=None, max_width=480):
    """Feature initialization + ECC refinement, restricted to a similarity warp.

    ECC estimates an affine warp; projecting it onto a similarity prevents focus
    blur from inducing shear. Candidate quality is measured after projection.
    """
    original_shape = reference.shape[:2]
    factor = min(1., max_width / original_shape[1])
    size = (round(original_shape[1]*factor), round(original_shape[0]*factor))
    a = cv2.resize(gray(reference), size).astype(np.float32) / 255.
    b = cv2.resize(gray(moving), size).astype(np.float32) / 255.
    mask = roi_mask(a.shape, roi)
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    b = cv2.GaussianBlur(b, (0, 0), 1.2)
    forward = np.eye(2, 3, dtype=np.float32)
    if initial is not None:
        forward = scale_transform(initial, original_shape, a.shape)
    else:
        sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=.015)
        ka, da = sift.detectAndCompute((a*255).astype(np.uint8), mask)
        kb, db = sift.detectAndCompute((b*255).astype(np.uint8), mask)
        if da is not None and db is not None and len(da)>4 and len(db)>4:
            matches = cv2.BFMatcher().knnMatch(db, da, k=2)
            good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .78*n.distance]
            if len(good)>=5:
                m, _ = cv2.estimateAffinePartial2D(np.float32([kb[q.queryIdx].pt for q in good]),
                    np.float32([ka[q.trainIdx].pt for q in good]), method=cv2.RANSAC, ransacReprojThreshold=2.)
                if m is not None and .8 < np.linalg.det(m[:,:2]) < 1.25:
                    forward = m.astype(np.float32)
    def correlation(matrix):
        aligned = cv2.warpAffine(b, matrix, size)
        valid = cv2.warpAffine(np.ones(b.shape,np.uint8), matrix, size)>0
        region = (mask>0) & valid
        av, bv = a[region], aligned[region]
        value = float(np.corrcoef(av,bv)[0,1]) if len(av)>32 and min(av.std(),bv.std())>1e-5 else 0.
        return value if np.isfinite(value) else 0.
    initial_quality = correlation(forward)
    inverse = cv2.invertAffineTransform(forward)
    input_mask = cv2.warpAffine(mask, inverse, size, flags=cv2.INTER_NEAREST)
    converged = False
    try:
        _, inverse = cv2.findTransformECC(a, b, inverse, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 65, 2e-5), input_mask, 5)
        candidate = cv2.invertAffineTransform(inverse)
        u, s, vt = np.linalg.svd(candidate[:,:2])
        linear = u @ vt * s.mean()
        # Preserve the affine map's position at the subject centroid.
        ys, xs = np.where(mask>0)
        center = np.array([xs.mean(), ys.mean()])
        candidate[:,2] += (candidate[:,:2] - linear) @ center
        candidate[:,:2] = linear
        if np.linalg.det(linear)>0 and .82 < s.mean() < 1.22 and correlation(candidate)>=initial_quality:
            forward = candidate.astype(np.float32)
            converged = True
    except cv2.error:
        pass
    return scale_transform(forward, a.shape, original_shape), {"correlation":correlation(forward),
        "initial_correlation":initial_quality,"ecc_converged":converged}


def select_coverage(scores, mask, max_frames=48, target=.985):
    """Greedy spatial coverage with a robust best-of-neighbors focus envelope."""
    flat = scores.reshape(len(scores), -1).astype(np.float32)
    active = mask.ravel()>0
    peak = flat.max(axis=0)
    if not active.any():
        raise ValueError("ROI has no focus cells; enlarge it or increase proxy resolution")
    active &= (peak >= float(np.quantile(peak[active], .15))) & (peak > 1e-7)
    if not active.any():
        raise ValueError("No textured subject region found; choose a larger ROI or sharper clip")
    raw_values = np.minimum(flat[:,active]/np.maximum(peak[active], 1e-12), 1)
    # Differences below 3% of local focus energy are equivalent for selection.
    # This stops sensor noise and tiny within-plane fluctuations consuming the budget.
    values = np.minimum(raw_values/.97, 1)
    # Each textured spatial cell gets equal say, preventing one bright edge dominating.
    covered = np.zeros(values.shape[1],np.float32)
    chosen, gains = [], []
    for _ in range(min(max_frames,len(values))):
        gain = np.maximum(values-covered,0).mean(axis=1)
        gain[chosen] = -1
        best = int(np.argmax(gain))
        if gain[best] < .00025:
            break
        chosen.append(best)
        gains.append(float(gain[best]))
        covered = np.maximum(covered,values[best])
        if covered.mean()>=target:
            break
    return chosen, {"mean_coverage":float(covered.mean()), "p10_coverage":float(np.quantile(covered,.1)),
                    "raw_mean_coverage":float(raw_values[chosen].max(axis=0).mean()),
                    "equivalent_focus_tolerance":.03,
                    "textured_cells":int(active.sum()), "marginal_gains":gains,
                    "target_reached":bool(covered.mean()>=target)}


def _decode_bt709(image):
    x = image.astype(np.float32) / 65535.
    return np.where(x < .081, x/4.5, ((x+.099)/1.099)**(1/.45)).astype(np.float32)


@lru_cache(maxsize=1)
def _bt709_uint16_lut():
    # Evaluate the same float32 expression once for every possible code value.
    # A read-only table prevents one caller from altering later conversions.
    table = _decode_bt709(np.arange(65536, dtype=np.uint16))
    table.flags.writeable = False
    return table


def to_linear_bt709(image):
    if image.dtype == np.uint16:
        result = _bt709_uint16_lut()[image]
        return np.asarray(result) if image.ndim == 0 else result
    return _decode_bt709(image)


def to_srgb16(image):
    x = np.clip(image,0,1)
    y = np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)
    return np.rint(np.clip(y,0,1)*65535).astype(np.uint16)


def laplacian_pyramid(image, levels):
    gaussian = [image]
    for _ in range(levels-1):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    return [gaussian[i]-cv2.pyrUp(gaussian[i+1],dstsize=(gaussian[i].shape[1],gaussian[i].shape[0]))
            for i in range(levels-1)] + [gaussian[-1]]


def reconstruct(pyramid):
    result = pyramid[-1]
    for band in pyramid[-2::-1]:
        result = cv2.pyrUp(result,dstsize=(band.shape[1],band.shape[0]))+band
    return result


def common_crop(mask):
    """Largest all-valid axis-aligned rectangle (histogram stack, exact)."""
    height = np.zeros(mask.shape[1],np.int32)
    best = (0,0,0,0)
    area = 0
    for y,row in enumerate(mask):
        height = (height+1)*(row>0)
        stack = []
        for x in range(len(height)+1):
            h = int(height[x]) if x<len(height) else 0
            start = x
            while stack and stack[-1][1]>h:
                left,hh=stack.pop()
                if hh*(x-left)>area:
                    area=hh*(x-left); best=(left,y-hh+1,x,y+1)
                start=left
            if not stack or stack[-1][1]<h:
                stack.append((start,h))
    if area==0:
        raise ValueError("Selected frames have no common valid region")
    return best
