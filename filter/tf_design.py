from filter.specs import LPFSpec, AngularFreq, BA
from enum import Enum
from typing import List, Dict
import numpy as np
from scipy.signal import iirdesign, iirfilter, freqs
import math
import matplotlib.figure
from joblib import Memory

"""
Design of the low-pass filter transfer function from specification to the zeros, poles, and DC gain required
to achieve these specs using various filter types.
"""

cachedir = './cache'
memory = Memory(cachedir, verbose=0)


class FilterType(Enum):
    BUTTERWORTH = 'butter'
    CHEBYSHEV1 = 'cheby1'
    CHEBYSHEV2 = 'cheby2'
    ELLIPTIC = 'ellip'
    BESSEL = 'bessel'


# Design a minimum order LPF of filter type 'ftype'.
@memory.cache
def design_lpf(spec: LPFSpec, ftype: FilterType) -> BA:
    # For a non-Bessel filter type, the regular scipy iirdesign function works and calculates the minimal order
    # But iteration on the passband corner is required to meet the group delay requirement
    if ftype != FilterType.BESSEL:
        pass_corner = spec.passband_corner
        while pass_corner < spec.stopband_corner:
            f = iirdesign(
                wp=pass_corner,
                ws=spec.stopband_corner,
                gpass=spec.passband_ripple,
                gstop=spec.stopband_atten,
                analog=True,
                ftype=ftype.value,
                output='ba'
            )
            ba = BA(f[0], f[1])
            if group_delay_variation(ba, spec) < spec.group_delay_variation:
                return ba
            else:
                pass_corner = pass_corner + 1e6
    # For a Bessel filter iirfilter needs to be used with manual control of order
    # Don't bother to control for the group delay spec since Bessel is very flat
    else:
        order = 1
        pass_corner = spec.passband_corner
        while order < 10:
            f = iirfilter(
                N=order,
                Wn=[pass_corner],
                rp=spec.passband_ripple,
                rs=spec.stopband_atten,
                btype='lowpass',
                analog=True,
                ftype=ftype.value,
                output='ba')
            w, h = freqs(f[0], f[1], worN=[spec.passband_corner, spec.stopband_corner])
            ba = BA(f[0], f[1])
            if -20*np.log10(abs(h[1])) < spec.stopband_atten:
                order = order + 1
            elif 20*np.log10(abs(h[0])) < -3 or group_delay_variation(ba, spec) > spec.group_delay_variation:
                pass_corner = pass_corner + 1e6
            else:
                return ba


# Calculate a reasonable frequency range of analysis
def freq_range(spec: LPFSpec, num=1000) -> List[AngularFreq]:
    return np.logspace(
        start=math.log10(spec.passband_corner) - 0.5,
        stop=math.log10(spec.stopband_corner) + 0.5,
        num=num)


def plot_filters_gain(filters: Dict[FilterType, BA], spec: LPFSpec, ax: matplotlib.figure.Axes):
    plot_w = freq_range(spec)
    for ftype, ba in filters.items():
        w, h = freqs(ba.B, ba.A, worN=plot_w)
        f = w / (2*math.pi)
        db = 20*np.log10(abs(h))
        ax.semilogx(f, db, linewidth=2, label=ftype.value)
    ax.fill([
        AngularFreq(plot_w[0]).f(),
        spec.passband_corner.f(),
        spec.passband_corner.f(),
        AngularFreq(plot_w[0]).f()
    ], [spec.passband_ripple, spec.passband_ripple, -3, -3], alpha=0.1, label='Passband Region')
    ax.fill([
        spec.stopband_corner.f(),
        AngularFreq(plot_w[-1]).f(),
        AngularFreq(plot_w[-1]).f(),
        spec.stopband_corner.f()
    ], [-spec.stopband_atten, -spec.stopband_atten, db[-1], db[-1]], alpha=0.1, label='Stopband Region')
    ax.axvline(x=spec.passband_corner.f(), linestyle='--', linewidth=0.7, color='b', label='Passband Corner')
    ax.axvline(x=spec.stopband_corner.f(), linestyle='--', linewidth=0.7, color='g', label='Stopband Corner')
    ax.axhline(y=-3, linestyle='--', linewidth=0.7, color='r', label='-3dB Passband Attenuation')
    ax.axhline(y=-spec.stopband_atten, linestyle='--', linewidth=0.7, color='m', label='-55dB Stopband Attenuation')
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|H(s)| [dB]')
    ax.set_title('Filter Gain')
    ax.set_ylim(-spec.stopband_atten - 20, spec.passband_ripple + 2)
    ax.grid(True)


def plot_filters_group_delay(filters: Dict[FilterType, BA], spec: LPFSpec, ax: matplotlib.figure.Axes):
    for ftype, ba in filters.items():
        w, gdelay = group_delay(ba, spec)
        ax.semilogx(w[1:] / (2*math.pi), gdelay*1e9, linewidth=2, label=ftype.value)
    ax.axvline(x=20e6, linestyle='--', linewidth=0.7, color='g', label='Passband Corner')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Group Delay [ns]')
    ax.set_title('Group Delay')
    ax.legend()
    ax.set_ylim(0, 30)
    ax.grid(True)


# Computes the group delay for filter BA for a reasonable frequency range based on the spec
def group_delay(ba: BA, spec: LPFSpec) -> (List[AngularFreq], List[float]):
    w = freq_range(spec)
    w, h = freqs(ba.B, ba.A, worN=w)
    gdelay = (-np.diff(np.unwrap(np.angle(h))) / np.diff(w))
    return w, gdelay


# Find the index in array that is closest to the value v
# From: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest_idx(a: np.ndarray, v):
    a = np.asarray(a)
    idx = (np.abs(a - v)).argmin()
    return idx


# Calculates the maximum group delay variation across the passband
def group_delay_variation(ba: BA, spec: LPFSpec) -> float:
    w, gdelay = group_delay(ba, spec)
    passband_idx = find_nearest_idx(w, spec.passband_corner)
    max_gdelay = np.max(gdelay[0:passband_idx])
    min_gdelay = np.min(gdelay[0:passband_idx])
    return np.abs(max_gdelay - min_gdelay)
