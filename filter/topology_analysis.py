import ahkab
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import f
from sympy import I
from scipy.constants import k
import scipy, scipy.interpolate
from enum import Enum
from typing import Optional
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
    'grp_del': 3           #ns
}

ac_params = {
    'start': 1e3,
    'stop': 1e9,
    'pts': 100
}

class FilterTopology(Enum):
    SK = 'sallen-key'
    MFB = 'multiple-feedback'

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
    sk2.add_isource('InR1', 'in', 'n1', dc_value=0, ac_value=0) #R1 noise current
    sk2.add_resistor('R2', 'n1', 'n2', d['R2'])
    sk2.add_isource('InR2', 'n1', 'n2', dc_value=0, ac_value=0) #R2 noise current
    sk2.add_capacitor('C1', 'n1', 'out', d['C1'])
    sk2.add_capacitor('C2', 'n2', sk2.gnd, d['C2'])

    #if gbw is not None:

    if ro:
        sk2.add_vcvs('E1', 'ne', sk2.gnd, 'n2', 'out', d['E1'])
        sk2.add_resistor('RO', 'out', 'ne', d['RO']) #DANGER: this blows up the solution
        sk2.add_isource('InE1', 'ne', sk2.gnd, dc_value=0, ac_value=0) #op-amp total output noise current
    else:
        sk2.add_vcvs('E1', 'out', sk2.gnd, 'n2', 'out', d['E1'])
        sk2.add_isource('InE1', 'out', sk2.gnd, dc_value=0, ac_value=0) #op-amp total output noise current

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

def run_sym(circuit, source, print_tf = False):
    """
    :param circuit: ahkab circuit
    :param source: name of the source for analysis
    :return: results for symbolic analysis
    """
    if not isinstance(source, str):
        raise ValueError('Source name must be a string! e.g. \'V1\'')

    r, tf = ahkab.run(circuit, ahkab.new_symbolic(source=source, verbose=4))['symbolic']

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

def check_specs(tf, d=None, specs=None):
    """
    :param tf: transfer function w/ gain, poles, zeros
    :param d: design dictionary of values for R's & C's
    :param specs: target specs dictionary
    :return:
    """
    design_pass = True

    # Get the design symbols by iterating through expression
    #H = tf['gain']
    #tf['gain'].free_symbols
    subs_dict = {}

    for sym in tf['gain'].free_symbols:
        print(sym)
        if str(sym) == 's':
            subs_dict[sym] = I*2*np.pi*f
        else:
            subs_dict[sym] = d[str(sym)]
    print(subs_dict)

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
tf_v1 = run_sym(lpf, 'V1', True)
check_specs(tf_v1, design_dict)

#    if topology not in filter_list:
#        raise ValueError("'%s' is invalid filter topology." % topology)

