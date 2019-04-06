from typing import List, Tuple
import numpy as np
import sympy as sp
from filter.specs import BA
from filter.topology_construction import OTA3Spec, build_lpf, FT, run_sym


# Building a LUT of OTA3 design points (R, C, gm) combos that best fit the transfer function
# This function used equations from the book "Continuous-Time Active Filter Design"
# Bad idea because this doesn't generalize across topologies
def construct_ideal_lut_textbook(desired_filter: BA) -> List[Tuple[float, float, float]]:
    """
    Given a desired filter transfer function, constructs a 3 passive 2nd order OTA filter circuit, derives
    its symbolic poles, and computes a LUT representing possible combinations of R/C/gm which achieve the pole locations.
    :param desired_filter: Desired filter TF from scipy.iirdesign / design_lpf()
    :return: A set of potential solutions and the symbolic filter circuit transfer function
    """
    def solutions():
        w0 = np.sqrt(desired_filter.B[0])
        Q = w0 / desired_filter.A[1]
        for C_val in np.geomspace(start=10e-15, stop=10e-12, num=100):
            yield(float((2 * Q) / (w0 * C_val)), float(C_val), float(2*Q*w0*C_val))
    return list(solutions())


# Similar function, except tries to construct the (R,C,gm) LUT using symbolic methods
# Bad idea because we just tried to equate the symbolic poles to the numerical desired poles
# This gave values of gm with small imag parts due to numerical precision loss
# Also just equating poles doesn't equate the transfer functions together, this is a bad idea
def construct_ideal_lut(desired_filter: BA) -> (List[Tuple[float, float, float]]):
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
    C, R, gm = sp.symbols('C R g_m', real=True)
    s = sp.symbols('s')
    sym_poles = [p.subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm}) for p in tf['poles']]

    sym_hs = tf['gain'].subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm})
    sp.pprint(sym_hs)
    sp.pprint(sym_poles)
    def produce_valid_solutions():
        for C_val in np.geomspace(start=200e-15, stop=10e-12, num=5):
            for R_val in np.geomspace(start=1, stop=10e6, num=5):
                # TODO: is the right approach to solve for equality of the magnitude of the poles?
                sym_pole_subs = [p.subs({C: C_val, R: R_val}) for p in sym_poles]
                sol = sp.nonlinsolve([
                    sym_pole_subs[0] - des_poles[0],
                    sym_pole_subs[1] - des_poles[1],
                    #sym_pole_subs[0] * sym_pole_subs[1] - desired_filter.to_zpk().K
                ], [gm])
                print(sol)
    lut = set(produce_valid_solutions())

    # Verify that a point in this ideal LUT constructs a filter TF that matches our desired specs
    lut_list = list(lut)
    hs_sym = tf['gain'].subs({C1_0: lut_list[0][1], C2_0: lut_list[0][1], R1_0: lut_list[0][0], G1_0: lut_list[0][2]})
    hs_num, hs_denom = sp.fraction(hs_sym)

    hs_a = sp.Poly(hs_denom, s).all_coeffs()
    factor = hs_a[0]
    hs_a = np.array([x / factor for x in hs_a]).astype(np.float64)
    hs_b = sp.Poly(hs_num, s).all_coeffs()
    hs_b = np.array([x / factor for x in hs_b]).astype(np.float64)
    hs_ba = BA(B=hs_b, A=hs_a)
    sp.pprint(hs_sym)
    print(hs_ba)
    return lut
