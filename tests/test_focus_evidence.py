"""Real native patterns distinguish repeating scene detail from independent noise."""
import cv2
import numpy as np
import pytest

from macrostitch.focus_evidence import coherent_focus_evidence
from macrostitch.spatial import native_energy


def surface():
    yy,xx=np.indices((192,192))
    return (.3+.06*np.sin(xx*np.pi/4)+.025*np.sin(yy*np.pi/7)).astype(np.float32)


def test_uniform_real_texture_has_focus_evidence_despite_low_spatial_energy_ratio():
    sharp=surface()
    blurred=cv2.GaussianBlur(sharp,(0,0),2.5)
    interior=np.s_[30:-30,30:-30]
    # Reproduce the existing eligibility failure on real image patterns, not
    # hand-authored score arrays: spatial median is genuine signal here.
    for image in (sharp,blurred):
        energy,_=native_energy(np.repeat(image[...,None],3,axis=2))
        assert np.quantile(energy[interior],.99)/np.median(energy[interior])<2
    rng=np.random.default_rng(17)
    sharp_neighbors=[cv2.GaussianBlur(sharp,(0,0),sigma)+rng.normal(0,.003,sharp.shape).astype(np.float32)
                     for sigma in (.5,.8)]
    weak_neighbors=[cv2.GaussianBlur(sharp,(0,0),sigma)+rng.normal(0,.003,sharp.shape).astype(np.float32)
                    for sigma in (2.2,2.7)]
    focused=coherent_focus_evidence(sharp,sharp_neighbors)
    defocused=coherent_focus_evidence(blurred,weak_neighbors)
    assert focused['reliable'][interior].mean()>.99
    assert focused['scores'][interior].mean()>10*defocused['scores'][interior].mean()


def test_uniform_independent_sensor_noise_is_not_repeatable_detail():
    rng=np.random.default_rng(120)
    frames=[(.3+rng.normal(0,.03,(192,192))).astype(np.float32) for _ in range(3)]
    evidence=coherent_focus_evidence(frames[0],frames[1:])
    assert evidence['bandpass_variance'][30:-30,30:-30].mean()>1e-5
    assert evidence['reliable'][30:-30,30:-30].mean()<.01
    assert evidence['scores'][30:-30,30:-30].mean()<1e-7


def test_brighter_noisier_observation_cannot_beat_repeatable_sharp_texture():
    rng=np.random.default_rng(921)
    sharp=surface()
    noisy=(.6+rng.normal(0,.08,sharp.shape)).astype(np.float32)
    stable=coherent_focus_evidence(sharp,[sharp+rng.normal(0,.003,sharp.shape).astype(np.float32)
                                        for _ in range(2)])
    unstable=coherent_focus_evidence(noisy,[(.6+rng.normal(0,.08,sharp.shape)).astype(np.float32)
                                          for _ in range(2)])
    assert stable['scores'][30:-30,30:-30].mean()>100*unstable['scores'][30:-30,30:-30].mean()


def test_masked_evidence_does_not_assign_focus_outside_requested_surface():
    image=surface();mask=np.zeros(image.shape,bool);mask[35:155,45:145]=True
    result=coherent_focus_evidence(image,[cv2.GaussianBlur(image,(0,0),.6)],mask=mask)
    assert not result['scores'][~mask].any()
    assert result['reliable'][55:135,65:125].all()


def test_evidence_requires_a_separate_observation_and_matching_geometry():
    with pytest.raises(ValueError,match='neighboring frame'):
        coherent_focus_evidence(surface(),[])
    with pytest.raises(ValueError,match='dimensions'):
        coherent_focus_evidence(surface(),[np.zeros((10,10),np.float32)])
