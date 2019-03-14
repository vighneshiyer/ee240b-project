from enum import Enum
from typing import List, Dict
import numpy as np
from scipy.signal import iirdesign, iirfilter, freqs, tf2zpk
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect


@dataclass(frozen=True)
class OrdFreq:
    freq: float

    def to_w(self) -> 'AngularFreq': return AngularFreq(self.freq * 2 * math.pi)

    def w(self) -> float: return self.freq * 2 * math.pi


@dataclass(frozen=True)
class AngularFreq:
    freq: float

    def to_f(self) -> OrdFreq: return OrdFreq(self.freq / (2 * math.pi))

    def f(self) -> float: return self.freq / (2 * math.pi)


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


class FilterType(Enum):
    BUTTERWORTH = 'butter'
    CHEBYSHEV1 = 'cheby1'
    CHEBYSHEV2 = 'cheby2'
    ELLIPTIC = 'ellip'
    BESSEL = 'bessel'


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


# Design a minimum order LPF of filter type 'ftype'.
# The function will increase the filter order until the stopband attenuation spec is met.
def design_lpf(spec: LPFSpec, ftype: FilterType) -> BA:
    # For a non-Bessel filter type, the regular scipy iirdesign function works and calculates the minimal order
    if ftype != FilterType.BESSEL:
        f = iirdesign(
            wp=spec.passband_corner.freq,
            ws=spec.stopband_corner.freq,
            gpass=spec.passband_ripple,
            gstop=spec.stopband_atten,
            analog=True,
            ftype=ftype.value,
            output='ba'
        )
        return BA(f[0], f[1])
    # For a Bessel filter iirfilter needs to be used with manual control of order
    else:
        order = 1
        pass_corner = spec.passband_corner.freq
        while order < 10:
            b, a = iirfilter(
                N=order,
                Wn=[pass_corner],
                rp=spec.passband_ripple,
                rs=spec.stopband_atten,
                btype='lowpass',
                analog=True,
                ftype=ftype.value,
                output='ba')
            w, h = freqs(b, a, worN=[spec.passband_corner.freq, spec.stopband_corner.freq])
            if -20*np.log10(abs(h[1])) < spec.stopband_atten:
                order = order + 1
            elif 20*np.log10(abs(h[0])) < -3:
                pass_corner = pass_corner + 1e6
            else:
                return BA(b, a)


def plot_filters(filters: Dict[FilterType, BA], spec: LPFSpec):
    plt.style.use('ggplot')
    width, height = figaspect(1/3)
    fig, ax = plt.subplots(figsize=(width, height))
    plot_w = np.logspace(
            start=math.log10(spec.passband_corner.freq) - 0.5,
            stop=math.log10(spec.stopband_corner.freq) + 0.5,
            num=1000)
    for ftype, ba in filters.items():
        w, h = freqs(ba.B, ba.A, worN=plot_w)
        f = w / (2*math.pi)
        db = 20*np.log10(abs(h))
        plt.semilogx(f, db, linewidth=2)
    # Passband region
    ax.fill([
        AngularFreq(plot_w[0]).f(),
        spec.passband_corner.f(),
        spec.passband_corner.f(),
        AngularFreq(plot_w[0]).f()
    ], [spec.passband_ripple, spec.passband_ripple, -3, -3], alpha=0.1)
    # Stopband region
    ax.fill([
        spec.stopband_corner.f(),
        AngularFreq(plot_w[-1]).f(),
        AngularFreq(plot_w[-1]).f(),
        spec.stopband_corner.f()], [-spec.stopband_atten, -spec.stopband_atten, db[-1], db[-1]], alpha=0.1)
    plt.axvline(x=spec.passband_corner.f(), linestyle='--', linewidth=0.7, color='b')
    plt.axvline(x=spec.stopband_corner.f(), linestyle='--', linewidth=0.7, color='g')
    plt.axhline(y=-3, linestyle='--', linewidth=0.7, color='r')
    plt.axhline(y=-spec.stopband_atten, linestyle='--', linewidth=0.7, color='m')
    plt.legend([*[x.value for x in filters.keys()],
                'Passband Corner', 'Stopband Corner', '-3dB Passband Attenuation',
                '-55dB Stopband Attenuation', 'Passband Region', 'Stopband Region'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude response (dB)')
    plt.ylim([-spec.stopband_atten - 20, spec.passband_ripple + 2])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
