import ahkab
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import f
from sympy import I
from scipy.constants import k
import scipy, scipy.interpolate
from enum import Enum
from filter.specs import LPFSpec, AngularFreq
from joblib import Memory

cachedir = './cache'
memory = Memory(cachedir, verbose=1)

sp.init_printing()

filter_list = ['sallen-key', 'multiple-feedback']

nonideal_dict = {
    'Av': 100,
    'RO': 200,
    'Cload': 40e-15
}

specs_dict = {
    'gain': 0,             #dB
    'passband': 20e6,      #Hz
    'pass_att': 3,         #dB
    'stopband': 200e6,     #Hz
    'stop_att': 55,        #dB
    'grp_del': 3,          #ns
    'gain_ripple': 1,      #dB
    'snr': 50              #dB
}

ac_params = {
    'start': 1e3,
    'stop': 1e9,
    'pts': 100
}

class FilterTopology(Enum):
    SK = 'sallen-key'
    MFB = 'multiple-feedback'

@memory.cache
def build_sk2(d, ro = False, gbw = None):
    """
    :param d: design_dict of all components & values (must be complete!)
    :param ro: True if ro should be taken into account
    :param gbw: value of finite GBW
    :return: ahkab circuit for Sallen-Key filter of order 2.
    """
    sk2 = ahkab.Circuit('2nd Order Sallen-Key LPF')
    sk2.add_capacitor('CL', 'out', sk2.gnd, nonideal_dict['Cload']) #load

    sk2.add_resistor('R1', 'in', 'n1', d['R1'])
    sk2.add_isource('INR1', 'in', 'n1', dc_value=0, ac_value=0) #R1 noise current
    sk2.add_resistor('R2', 'n1', 'n2', d['R2'])
    sk2.add_isource('INR2', 'n1', 'n2', dc_value=0, ac_value=0) #R2 noise current
    sk2.add_capacitor('C1', 'n1', 'out', d['C1'])
    sk2.add_capacitor('C2', 'n2', sk2.gnd, d['C2'])

    #if gbw is not None:

    if ro:
        sk2.add_vcvs('E1', 'ne', sk2.gnd, 'n2', 'out', d['E1'])
        sk2.add_resistor('RO', 'out', 'ne', d['RO']) #DANGER: this blows up the solution
        sk2.add_isource('INE1', 'ne', sk2.gnd, dc_value=0, ac_value=0) #op-amp total output noise current
    else:
        sk2.add_vcvs('E1', 'out', sk2.gnd, 'n2', 'out', d['E1'])
        sk2.add_isource('INE1', 'out', sk2.gnd, dc_value=0, ac_value=0) #op-amp total output noise current

    sk2.add_vsource('V1', 'in', sk2.gnd, dc_value=0, ac_value=1)

    print(sk2)
    return sk2

def run_ac(circuit):
    """
    :param circuit: ahkab circuit
    :return: results for AC analysis
    """
    opa = ahkab.new_op()
    aca = ahkab.new_ac(ac_params['start'], ac_params['stop'], ac_params['pts'])
    return ahkab.run(circuit, [opa, aca])['ac']

@memory.cache
def run_sym(circuit, source, print_tf = False):
    """
    :param circuit: ahkab circuit
    :param source: name of the source for analysis
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

def subs_syms(tf, d):
    """
    :param tf: transfer function
    :param d: design dictionary of values for R's & C's
    :return: dictionary with values to substitute for dictionary
    """
    subs_dict = {}
    for sym in tf['gain'].free_symbols:
        if str(sym) == 's':
            subs_dict[sym] = I*2*np.pi*f
        else:
            subs_dict[sym] = d[str(sym)]

    return subs_dict

def check_specs(rac, tf, d, specs):
    """
    :param rac: AC analysis result
    :param tf: transfer function w/ gain, poles, zeros
    :param d: design dictionary of values for R's & C's
    :param specs: target specs dictionary
    :return:
    """
    design_pass = True

    # Substitute design variables into transfer function
    subs_dict = subs_syms(tf, d)

    hs = sp.lambdify(f, tf['gain'].subs(subs_dict))

    # Print poles/zeros
    for i, z in enumerate(tf['zeros']):
        zero = z.subs(subs_dict)/2/np.pi
        print("Zero #{}: {} + {}j ({} Hz)".format(i, sp.re(zero), sp.im(zero), sp.Abs(zero)))
    for i, p in enumerate(tf['poles']):
        pole = p.subs(subs_dict)/2/np.pi
        print("Pole #{}: {} + {}j ({} Hz)".format(i, sp.re(pole), sp.im(pole), sp.Abs(pole)))

    # Passband/stopband
    spec_test = -20*np.log10(np.abs(hs(specs['passband'])))
    if spec_test > specs['pass_att']:
        design_pass = False
        print('Fails passband attenuation: {} dB > {} dB spec'.format(spec_test, specs['pass_att']))
    else:
        print('Passes passband attenuation!')
    spec_test = -20*np.log10(np.abs(hs(specs['stopband'])))
    if spec_test < specs['stop_att']:
        design_pass = False
        print('Fails stopband attenuation: {} dB < {} dB spec'.format(spec_test, specs['stop_att']))
    else:
        print('Passes stopband attenuation!')

    # Group Delay (interpolate @ passband)
    grp_del = (-np.diff(np.unwrap(np.angle(hs(rac.get_x())))) / np.diff(rac['f'])) * 1e9
    grp_del_norm = grp_del - grp_del[0]
    grp_del_interp = scipy.interpolate.interp1d(rac['f'][1:], grp_del_norm)
    spec_test = grp_del_interp(specs['passband'])
    if spec_test > specs['grp_del']:
        design_pass = False
        print('Fails group delay: {} ns > {} ns spec'.format(spec_test, specs['grp_del']))
    else:
        print('Passes group delay!')

    # Gain ripples

    return design_pass

def check_dyn_range(circuit, srcs, tf, d, specs):
    """
    :param circuit: ahkab circuit
    :param srcs: List of all noise sources (strings)
    :param tf: Overall filter transfer function
    :param d: Design dict
    :param specs: Specs dict
    :return: total integrated noise in passband
    """
    design_pass = True
    vi2 = 0

    print("Noise simulation temp: {} K".format(ahkab.constants.Tref))
    kT4 = 4*k*ahkab.constants.Tref

    # Assume opamp noise can be modeled as single transistor noise current w/ some gm
    gamma = 2/3
    gm = 0.005

    for s in srcs:
        if s.startswith('INR'):
            in2 = kT4/d[s[2:]]
        elif s.startswith('INE'):
            in2 = kT4*gamma*gm

        tfn = run_sym(circuit, s)
        subs_dict = subs_syms(tfn, d)

        vi2 += sp.integrate(sp.Abs(((tfn['gain']/tf['gain']).subs(subs_dict))) * in2, (f, 1, specs['passband']))

    print("Total input referred noise power: {} V^2".format(vi2))

    # Assume allowable swing (zero-peak) is VDD/2 - V*
    vstar = 0.1
    snr = 10**(specs['snr']/10)
    vi_min = sp.sqrt(vi2*snr*2)

    if vi_min > 0.6-vstar:
        print('Fails dynamic range: voltage swing {} not attainable!'.format(vi_min))
        design_pass = False
    else:
        print('Passes dynamic range!')

    return design_pass


Rbase = 900
m = 1.5
Cbase = 9e-12
n = 1.5
design_dict = { #keys must match the instance names of each component in design
    'R1': Rbase*m,
    'R2': Rbase/m,
    'C1': Cbase*n,
    'C2': Cbase/n,
    'CL': nonideal_dict['Cload'],
    'E1': nonideal_dict['Av'],
    'E2': nonideal_dict['Av'],
    'RO': nonideal_dict['RO']
}
lpf = build_sk2(design_dict)
rac = run_ac(lpf)
tf = run_sym(lpf, 'V1', True)
check_specs(rac, tf, design_dict, specs_dict)
check_dyn_range(lpf, ['INR1', 'INR2', 'INE1'], tf, design_dict, specs_dict)
