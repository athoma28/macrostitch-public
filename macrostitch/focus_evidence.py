"""Experimental temporal focus evidence; intentionally not wired into production.

Spatially uniform texture can be real detail. This module tests whether signed
bandpass structure repeats across registered neighboring observations instead of
assuming the spatial median focus energy is noise. It cannot distinguish fixed
pattern sensor noise from fixed scene texture, and bad alignment reduces support.
"""
from __future__ import annotations

import cv2
import numpy as np


def _bandpass(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        image = image[..., 0] * .0722 + image[..., 1] * .7152 + image[..., 2] * .2126
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError("Expected finite linear grayscale or BGR pixels")
    return cv2.GaussianBlur(image, (0, 0), .8) - cv2.GaussianBlur(image, (0, 0), 2.4)


def coherent_focus_evidence(current, neighbors, *, mask=None, sigma=8., minimum_correlation=.25):
    """Estimate repeatable bandpass energy from separate aligned observations.

    Inputs are linear-light grayscale or BGR pixels on the same native grid.
    Neighbors must be different captured frames with comparable scene structure,
    never duplicate copies of current. No image pixels are averaged or modified.

    For each neighbor, local positive covariance squared / neighbor variance is
    the energy explained by repeating structure. The median across neighbors
    reduces isolated coincidences. A heuristic correlation gate avoids admitting
    weak accidental matches. This is a diagnostic, not calibrated optical truth.
    """
    if not np.isfinite(sigma) or sigma <= 0 or not 0 < minimum_correlation < 1:
        raise ValueError("Invalid evidence support or correlation threshold")
    band = _bandpass(current)
    neighbors = list(neighbors)
    if not neighbors:
        raise ValueError("At least one separate neighboring frame is required")
    if mask is None:
        mask = np.ones(band.shape, bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != band.shape or not mask.any():
        raise ValueError("Evidence mask must match the image and contain pixels")
    weights = mask.astype(np.float32)
    support = cv2.GaussianBlur(weights, (0, 0), sigma)
    def mean(values):
        return cv2.GaussianBlur(values * weights, (0, 0), sigma) / np.maximum(support, 1e-12)
    average = mean(band)
    energy = np.maximum(mean(band * band) - average * average, 0)
    correlations = []
    projections = []
    for image in neighbors:
        other = _bandpass(image)
        if other.shape != band.shape:
            raise ValueError("Neighbor dimensions must match the current frame")
        other_average = mean(other)
        other_energy = np.maximum(mean(other * other) - other_average * other_average, 0)
        covariance = mean(band * other) - average * other_average
        correlation = np.clip(covariance / np.sqrt(np.maximum(energy * other_energy, 1e-20)), 0, 1)
        correlations.append(correlation)
        projections.append(energy * correlation * correlation)
    coherence = np.median(np.asarray(correlations), axis=0)
    explained = np.median(np.asarray(projections), axis=0)
    reliable = mask & (support >= .5) & (coherence >= minimum_correlation) & (energy > 1e-12)
    explained[~reliable] = 0
    return {"scores": explained, "bandpass_variance": energy, "coherence": coherence,
            "reliable": reliable, "minimum_correlation": minimum_correlation,
            "aggregation_sigma_px": sigma, "neighbor_count": len(neighbors)}
