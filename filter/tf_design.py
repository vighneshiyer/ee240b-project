from enum import Enum
from typing import List
import numpy as np
from scipy.signal import iirdesign, iirfilter, freqs, tf2zpk
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
plt.style.use('ggplot')


@dataclass(frozen=True)
class LPFSpec:
    passband_corner:        float  # In radians/sec
    stopband_corner:        float  # In radians/sec
    stopband_atten:         float  # In dB down from 0dB
    passband_ripple:        float  # In dB
    group_delay_variation:  float  # In seconds, maximum variation of group delay from low-freq to passband corner

    def __post_init__(self):
        assert self.passband_ripple > 0, "Can't have 0 ripple in the passband"
        assert self.stopband_atten > 0, "Attenuation should be given as a positive number " \
                                        "(-50dBc => stopband_atten = 50)"


class FilterType(Enum):
    BUTTERWORTH = 'butter'
    CHEBYSHEV1 = 'cheby1'
    CHEBYSHEV2 = 'cheby2'
    ELLIPTIC = 'ellip'
    BESSEL = 'bessel'


@dataclass(frozen=True)
class ZPK:
    Z: List[complex]
    P: List[complex]
    K: float


@dataclass(frozen=True)
class BA:
    B: List[complex]
    A: List[complex]

    def to_zpk(self) -> ZPK:
        zpk = tf2zpk(self.B, self.A)
        return ZPK(zpk[0], zpk[1], zpk[2])

    def order(self) -> int: return len(self.A)

    def dc_gain(self) -> float: return np.abs(self.B[0] / self.A[0])


# Design a minimum order LPF of filter type 'ftype'.
# The function will increase the filter order until the stopband attenuation spec is met.
def design_lpf(spec: LPFSpec, ftype: FilterType) -> BA:
    #assert ftype != FilterType.BESSEL, "iirdesign doesn't support Bessel automatic order selection"
    if ftype != FilterType.BESSEL:
        return iirdesign(
            wp=spec.passband_corner,
            ws=spec.stopband_corner,
            gpass=spec.passband_ripple,
            gstop=spec.stopband_atten,
            analog=True,
            ftype=ftype.value,
            output='ba'
        )
    else:
        order = 1
        while True:
            b, a = iirfilter(
                N=order,
                Wn=[spec.passband_corner],
                rp=spec.passband_ripple,
                rs=spec.stopband_atten,
                btype='lowpass',
                analog=True,
                ftype=ftype.value,
                output='ba')
            w, h = freqs(b, a, worN=[spec.stopband_corner])
            if -20*np.log10(abs(h[0])) > spec.stopband_atten:
                return order, b, a
            else:
                order = order + 1


spec = LPFSpec(2*math.pi*20e6, 2*math.pi*200e6, 55, 1, 3e-9)
b, a = design_lpf(spec, ftype=FilterType.BUTTERWORTH)
order = len(a)
w, h = freqs(b, a, worN=np.logspace(6, 10, 1000))
f = w / (2*math.pi)
db = 20*np.log10(abs(h))
width, height = figaspect(1/6)
fig, ax = plt.subplots(figsize=(width,height))
plt.semilogx(f, db, linewidth=1.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude response (dB)')
plt.axvline(x=20e6, linestyle='--', linewidth=0.7, color='b')
plt.axvline(x=200e6, linestyle='--', linewidth=0.7, color='g')
plt.axhline(y=-3, linestyle='--', linewidth=0.7, color='r')
plt.axhline(y=-55, linestyle='--', linewidth=0.7, color='m')
plt.grid(True)
ax.fill([f[0], 20e6, 20e6, f[0]], [3, 3, -3, -3], alpha=0.3)
ax.fill([200e6, f[-1], f[-1], 200e6], [-55, -55, db[-1], db[-1]], alpha=0.3)
plt.legend(['Filter Gain', 'Passband Corner', 'Stopband Corner', '-3dB Passband Attenuation',
            '-55dB Stopband Attenuation', 'Passband Region', 'Stopband Region'])
plt.tight_layout()
plt.show()
z, p, k = tf2zpk(b, a)
print(order, z, p, k)
print(b, a)
