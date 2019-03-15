from scipy.signal import tf2zpk
from typing import List
import math
from dataclasses import dataclass


# Ordinary frequency in cycles/second (Hz)
class OrdFreq(float):
    def w(self) -> 'AngularFreq': return AngularFreq(self * 2 * math.pi)


# Angular frequency in radians/second
class AngularFreq(float):
    def f(self) -> OrdFreq: return OrdFreq(self / (2 * math.pi))


@dataclass(frozen=True)
class LPFSpec:
    passband_corner:        AngularFreq
    stopband_corner:        AngularFreq
    stopband_atten:         float  # In dB down from 0dB
    passband_ripple:        float  # In dB
    group_delay_variation:  float  # In seconds, maximum variation of group delay from low-freq to passband corner

    def __post_init__(self):
        assert self.passband_ripple > 0, "Can't have 0 ripple in the passband"
        assert self.stopband_atten > 0, "Attenuation should be given as a positive number " \
                                        "(-50dBc => stopband_atten = 50)"


@dataclass(frozen=True)
class ZPK:
    Z: List[complex]  # Zeros of TF
    P: List[complex]  # Poles of TF
    K: float          # DC Gain


@dataclass(frozen=True)
class BA:
    B: List[complex]  # Numerator polynomial coefficients
    A: List[complex]  # Denominator polynomial coefficients

    def to_zpk(self) -> ZPK:
        zpk = tf2zpk(self.B, self.A)
        return ZPK(zpk[0], zpk[1], zpk[2])

    def order(self) -> int: return len(self.to_zpk().P)

    def dc_gain(self) -> float: return np.abs(self.B[0] / self.A[0])
