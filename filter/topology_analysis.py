import ahkab
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import f
from sympy import I
from scipy.constants import k
import scipy, scipy.interpolate
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from joblib import Memory
from filter.specs import LPFSpec, OrdFreq

cachedir = './cache'
memory = Memory(cachedir, verbose=1)

sp.init_printing()

nonideal_dict = {
    'Av': 100,
    'RO': 200,
    'Cload': 40e-15
}

ac_params = {
    'start': 1e3,
    'stop': 1e9,
    'pts': 100
}


class FT(Enum):  # filter topology
    SK = 'sallen-key'
    MFB = 'multiple-feedback'


@dataclass(frozen=True)
class SallenKeySpec:
    r1: float
    r2: float
    c1: float
    c2: float
    e1: float
    ro: Optional[float] = None  # output resistance of opamp
    e2: Optional[float] = None  # gain for GBW
    rp: Optional[float] = None  # for GBW
    cp: Optional[float] = None  # for GBW


def build_lpf(cascade: List[FT], fspecs: List, ro=False, gbw=False):
    """
    :param cascade: list of filter topologies in the cascade
    :param fspecs: list of filter specs
    :param ro: consider ro?
    :param gbw: consider GBW?
    :return: filter, dict of subs for symbolic expressions, list noise sources
    """
    subs_dict = {}
    noise_srcs = []
    filt = ahkab.Circuit('LPF')
    for i, t in enumerate(cascade):
        filt, s, n = attach_stage(filt, t, fspecs[i], len(cascade), i, ro, gbw)
        subs_dict.update(s)
        noise_srcs = noise_srcs + n
    print(filt)
    print(subs_dict)
    print(noise_srcs)
    return filt, subs_dict, noise_srcs


@memory.cache
def attach_stage(c: ahkab.Circuit, topology: FT, fspec, stages: int, pos: int, ro=False, gbw=False):
    """
    :param c: circuit to append to
    :param topology: filter topology to add
    :param fspec: components specs
    :param stages: total # of stages (to get in/out node correct)
    :param pos: position in cascade
    :param ro: consider ro?
    :param gbw: consider GBW?
    :return: filter, dict of subs for symbolic expressions, list noise sources
    """
    subs_dict = {}
    noise_srcs = []
    p = str(pos)
    if pos == 0:
        in_node = 'in'
        c.add_vsource('V1', in_node, c.gnd, dc_value=0, ac_value=1)  # source
    else:
        in_node = 'int_'+str(pos-1)
    if pos == stages-1:
        out_node = 'out'
        c.add_capacitor('CL', out_node, c.gnd, nonideal_dict['Cload'])  # load
        subs_dict['CL'] = nonideal_dict['Cload']
    else:
        out_node = 'int_'+str(pos)

    if topology == FT.SK:
        if not isinstance(fspec, SallenKeySpec):
            raise ValueError("Wrong specs given for Sallen-Key Filter!")

        # Add components
        c.add_resistor('R1_'+p, in_node, 'n1_'+p, fspec.r1)
        subs_dict['R1_'+p] = fspec.r1
        c.add_isource('INR1_'+p, in_node, 'n1_'+p, dc_value=0, ac_value=0)
        noise_srcs.append('INR1_'+p)
        c.add_resistor('R2_'+p, 'n1_'+p, 'n2_'+p, fspec.r2)
        subs_dict['R2_'+p] = fspec.r2
        c.add_isource('INR2_'+p, 'n1_'+p, 'n2_'+p, dc_value=0, ac_value=0)
        noise_srcs.append('INR2_'+p)
        c.add_capacitor('C1_'+p, 'n1_'+p, out_node, fspec.c1)
        subs_dict['C1_'+p] = fspec.c1
        c.add_capacitor('C2_'+p, 'n2_'+p, c.gnd, fspec.c2)
        subs_dict['C2_'+p] = fspec.c2

        #if gbw:

        if ro:
            c.add_vcvs('E1_'+p, 'ne_'+p, c.gnd, 'n2_'+p, out_node, fspec.e1)
            subs_dict['E1_'+p] = fspec.e1
            c.add_resistor('RO_'+p, out_node, 'ne_'+p, fspec.ro)
            subs_dict['RO_'+p] = fspec.ro
            c.add_isource('INE1_'+p, 'ne_'+p, c.gnd, dc_value=0, ac_value=0)
        else:
            c.add_vcvs('E1_'+p, out_node, c.gnd, 'n2_'+p, out_node, fspec.e1)
            subs_dict['E1_'+p] = fspec.e1
            c.add_isource('INE1_'+p, out_node, c.gnd, dc_value=0, ac_value=0)
        noise_srcs.append('INE1_'+p)

    return c, subs_dict, noise_srcs


def run_ac(circuit):
    """
    :param circuit: ahkab circuit
    :return: results for AC analysis
    """
    opa = ahkab.new_op()
    aca = ahkab.new_ac(ac_params['start'], ac_params['stop'], ac_params['pts'])
    return ahkab.run(circuit, [opa, aca])['ac']


@memory.cache
def run_sym(circuit, source, print_tf=False):
    """
    :param circuit: ahkab circuit
    :param source: name of the source for analysis
    :param print_tf: print transfer function?
    :return: results for symbolic analysis
    """
    if not isinstance(source, str):
        raise ValueError('Source name must be a string! e.g. \'V1\'')

    r, tf = ahkab.run(circuit, ahkab.new_symbolic(source=source, verbose=0))['symbolic']

    tfs = tf['VOUT/'+str(source)]

    if print_tf:
        print("DC gain: {} dB".format(20*sp.log(tf['VOUT/'+str(source)]['gain0'], 10)))
        print("Transfer function:")
        sp.pprint(tfs['gain'])
        for i, z in enumerate(tfs['zeros']):
            print("Zero #{}:".format(i))
            sp.pprint(z)
        for i, p in enumerate(tfs['poles']):
            print("Pole #{}:".format(i))
            sp.pprint(p)

    return tfs


def subs_syms(tf, subs):
    """
    :param tf: transfer function
    :param subs: dict of all possible values to substitute
    :return: dict of only values to substitue for free_symbols
    """
    subs_dict = {}
    for sym in tf['gain'].free_symbols:
        if str(sym) == 's':
            subs_dict[sym] = I*2*np.pi*f
        else:
            subs_dict[sym] = subs[str(sym)]

    return subs_dict


def check_specs(circuit, rac, tf, spec: LPFSpec, subs, noise_srcs):
    """
    :param circuit: ahkab circuit
    :param rac: AC analysis result
    :param tf: transfer function w/ gain, poles, zeros
    :param spec: target specs
    :param subs: dictionary of component value substitutions
    :param noise_srcs: list of noise sources
    :return:
    """
    design_pass = True

    # Substitute design variables into transfer function
    subs_dict = subs_syms(tf, subs)

    hs = sp.lambdify(f, tf['gain'].subs(subs_dict))
    print(hs)

    # Print poles/zeros
    for i, z in enumerate(tf['zeros']):
        zero = z.subs(subs_dict)/2/np.pi
        print("Zero #{}: {} + {}j ({} Hz)".format(i, sp.re(zero), sp.im(zero), sp.Abs(zero)))
    for i, p in enumerate(tf['poles']):
        pole = p.subs(subs_dict)/2/np.pi
        print("Pole #{}: {} + {}j ({} Hz)".format(i, sp.re(pole), sp.im(pole), sp.Abs(pole)))

    # Passband/stopband
    spec_test = -20*np.log10(np.abs(hs(spec.passband_corner.f())))
    if spec_test > 3:
        design_pass = False
        print('Fails passband attenuation: {} dB > 3 dB spec'.format(spec_test))
    else:
        print('Passes passband attenuation!')
    spec_test = -20*np.log10(np.abs(hs(spec.stopband_corner.f())))
    if spec_test < spec.stopband_atten:
        design_pass = False
        print('Fails stopband attenuation: {} dB < {} dB spec'.format(spec_test, spec.stopband_atten))
    else:
        print('Passes stopband attenuation!')

    # Group Delay (interpolate @ passband)
    grp_del = (-np.diff(np.unwrap(np.angle(hs(rac.get_x())))) / np.diff(rac['f']))
    grp_del_norm = grp_del - grp_del[0]
    grp_del_interp = scipy.interpolate.interp1d(rac['f'][1:], grp_del_norm)
    spec_test = grp_del_interp(spec.passband_corner.f())
    if spec_test > spec.group_delay_variation:
        design_pass = False
        print('Fails group delay: {} ns > {} ns spec'.format(
            spec_test*1e9, spec.group_delay_variation*1e9))
    else:
        print('Passes group delay!')

    # Gain ripples

    # Dynamic range
    vi2 = 0
    print("Noise simulation temp: {} K".format(ahkab.constants.Tref))
    kT4 = 4*k*ahkab.constants.Tref

    # Assume opamp noise can be modeled as single transistor noise current w/ some gm
    gamma = 2/3
    gm = 0.005

    # Integrate input-referred noise power using symbolic analysis
    for s in noise_srcs:
        if s.startswith('INR'):
            in2 = kT4/subs[s[2:]]
        elif s.startswith('INE'):
            in2 = kT4*gamma*gm
        tfn = run_sym(circuit, s)
        vi2 += sp.integrate(sp.Abs(((tfn['gain']/tf['gain']).subs(subs_syms(tfn, subs)))) * in2,
                            (f, 1, spec.passband_corner.f()))

    print("Total input referred noise power: {} V^2".format(vi2))

    # Assume allowable swing (zero-peak) is VDD/2 - V*
    vstar = 0.1
    dr = 10**(spec.dynamic_range/10)
    vi_min = sp.sqrt(vi2*dr*2)

    if vi_min > 0.6-vstar:
        print('Fails dynamic range: voltage swing of {} V not attainable!'.format(vi_min))
        design_pass = False
    else:
        print('Passes dynamic range!')

    return design_pass

spec = LPFSpec(
    passband_corner=OrdFreq(20e6).w(),
    stopband_corner=OrdFreq(200e6).w(),
    stopband_atten=55,
    passband_ripple=1,
    group_delay_variation=3e-9,
    dynamic_range=50
)

Rbase = 900
m = 1.5
Cbase = 9e-12
n = 1.5
sk0spec = SallenKeySpec(
    r1=Rbase*m,
    r2=Rbase/m,
    c1=Cbase*n,
    c2=Cbase/n,
    e1=nonideal_dict['Av'],
    ro=nonideal_dict['RO']
)

lpf, subs, nsrcs = build_lpf([FT.SK, FT.SK], [sk0spec, sk0spec])
rac = run_ac(lpf)
tf = run_sym(lpf, 'V1', True)
check_specs(lpf, rac, tf, spec, subs, nsrcs)
