from filter.topology_analysis import OTA3Spec, FT, build_lpf, run_sym
from filter.specs import BA
import sympy as sp
import numpy as np
from typing import Set, Tuple
from joblib import Memory

cachedir = './cache'
memory = Memory(cachedir, verbose=1)


#@memory.cache
def construct_ideal_lut(desired_filter: BA) -> (Set[Tuple[float, float, float]], sp.Expr):
    """
    Given a desired filter transfer function, constructs a 3 passive 2nd order OTA filter circuit, derives
    its symbolic poles, and computes a LUT representing possible combinations of R/C/gm which achieve the pole locations.
    :param desired_filter: Desired filter TF from scipy.iirdesign / design_lpf()
    :return: A set of potential solutions and the symbolic filter circuit transfer function
    """
    nonideal_dict = {
        'Av': 100,
        'RO': 200,
        'Cload': 40e-15,
        'gamma': 2,  # short-channel?
        'gm': 0.01  #
    }

    Rbase = 900
    Cbase = 9e-12
    ota3spec = OTA3Spec(
        r1=Rbase,
        c1=Cbase,
        c2=Cbase,
        gm=nonideal_dict['gm'],
        ro=nonideal_dict['Av']/nonideal_dict['gm'],
        bw=1e8
    )
    des_poles = desired_filter.to_zpk().P[:2]
    lpf, subs, nsrcs = build_lpf([FT.OTA3], [ota3spec], ro=False, cl=False)
    tf = run_sym(lpf, 'V1', True)

    # Step 1: take the ideal transfer function, and for reasonable ranges of gm, R, solve for the necessary C
    # to build a LUT of potential design points
    C1_0, C2_0, G1_0, R1_0 = sp.symbols('C1_0 C2_0 G1_0 R1_0')
    C, R, gm = sp.symbols('C R g_m', real=True, positive=True)
    sym_poles = [p.subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm}) for p in tf['poles']]

    def produce_valid_solutions():
        for R_val in np.geomspace(start=100, stop=50e3, num=10):
            for C_val in np.geomspace(start=200e-15, stop=10e-12, num=10):
                # TODO: is the right approach to solve for equality of the magnitude of the poles?
                sym_pole = sp.Abs(sym_poles[0].subs({R: R_val, C: C_val}))
                sol = sp.solveset(sym_pole - abs(des_poles[0]), gm, domain=sp.S.Reals)
                yield (R_val, C_val, abs(sol.args[0]))
    lut = set(produce_valid_solutions())
    # Verify that a point in this ideal LUT constructs a filter TF that matches our desired specs
    lut_list = list(lut)
    hs_sym = tf['gain'].subs({C1_0: lut_list[0][1], C2_0: lut_list[0][1], R1_0: lut_list[0][0], G1_0: lut_list[0][2]})

    hs_num, hs_denom = sp.fraction(hs_sym)
    #sp.pprint([sp.Poly(hs_num, s).all_coeffs(), sp.Poly(denom, x).all_coeffs()]
    sp.pprint(hs_sym)
    return lut
