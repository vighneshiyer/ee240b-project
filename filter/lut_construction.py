from filter.specs import BA
import sympy as sp
import numpy as np
from typing import List, Tuple
from joblib import Memory
import csv

cachedir = './cache'
memory = Memory(cachedir, verbose=1)


def construct_ideal_lut(desired_filter: BA) -> (List[Tuple[float, float, float]], sp.Expr):
    """
    Given a desired filter transfer function, constructs a 3 passive 2nd order OTA filter circuit, derives
    its symbolic poles, and computes a LUT representing possible combinations of R/C/gm which achieve the pole locations.
    :param desired_filter: Desired filter TF from scipy.iirdesign / design_lpf()
    :return: A set of potential solutions and the symbolic filter circuit transfer function
    """
    def solutions():
        w0 = np.sqrt(desired_filter.B[0])
        Q = w0 / desired_filter.A[1]
        for C_val in np.geomspace(start=200e-15, stop=10e-12, num=100):
            yield(float((2 * Q) / (w0 * C_val)), float(C_val), float(2*Q*w0*C_val))
    return list(solutions())
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
    C, R, gm = sp.symbols('C R g_m', real=True)
    s = sp.symbols('s')
    sym_poles = [p.subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm}) for p in tf['poles']]

    sym_hs = tf['gain'].subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm})
    sym_hs_num, sym_hs_denom = sp.fraction(sym_hs)
    sym_hs_a = sp.Poly(sym_hs_denom, s).all_coeffs()
    sym_hs_b = sp.Poly(sym_hs_num, s).all_coeffs()
    w0 = sp.sqrt(sym_hs_b[0])
    Q = w0 / sym_hs_a[1]
    sp.pprint(w0)
    sp.pprint(Q)
    sp.pprint(sym_hs)
    def produce_valid_solutions():
        for C_val in np.geomspace(start=200e-15, stop=10e-12, num=5):
            # TODO: is the right approach to solve for equality of the magnitude of the poles?
            sym_pole_subs = [p.subs({C: C_val}) for p in sym_poles]
            sol = sp.nonlinsolve([
                sym_pole_subs[0] - des_poles[0],
                sym_pole_subs[1] - des_poles[1],
                #sym_pole_subs[0] * sym_pole_subs[1] - desired_filter.to_zpk().K
            ], [gm, R])
            print(sol)
            #print(R_val, C_val, sol.args[0][0])
            #yield (R_val, C_val, sol.args[0][0])
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
    """

def construct_ota_lut(w: int=1) -> [Tuple[float, float, float]]:
    # Header: vstar	idc gm ro av wbw Cgg Cdd Css vgs drain_eff
    with open('filter/nmoschar.csv') as csv_file:
        header = csv_file.readline().strip().split(',')
        data = []
        for line in csv_file:
            data.append(np.array(list(map(np.float64, line.strip().split(',')))))

    scale = np.array([1, w, w, 1/w, 1, 1, w, w, w, 1, 1])

    scaled_data = list(map(lambda x: x * scale, data))

    return header, scaled_data
