from typing import Dict, List
from filter.specs import LPFSpec, BA
import sympy as sp
from functools import reduce
import inspect
from scipy.optimize import minimize
import numpy as np


def fit_filter_circuit(spec: LPFSpec, ba: BA, tfs: Dict[str, List[sp.Expr]], design_dict):
    """
    Use a generic optimizer to fit the transfer function 'tfs' to the filter spec 'spec' and desired transfer function 'ba'
    :param spec:
    :param ba:
    :param tfs:
    :return:
    """
    sym_poles = list(map(lambda p: (
        sp.lambdify(p.free_symbols, sp.re(p)),
        sp.lambdify(p.free_symbols, sp.im(p))), tfs['poles']))  # type: List[(Function, Function)]
    goal_poles = ba.to_zpk().P[0:2]
    print(goal_poles)

    def cost(y) -> float:
        R1 = y[0]
        R2 = y[1]
        C1 = y[2]
        C2 = y[3]
        actual_poles = map(lambda e: [
            e[0](R1=R1, R2=R2, C1=C1, C2=C2),
            e[1](R1=R1, R2=R2, C1=C1, C2=C2)], sym_poles)
        abs_diffs = map(lambda x: np.abs((np.real(x[0]) - x[1][0]) + (np.imag(x[0]) - x[1][1])*1j), zip(goal_poles, actual_poles))
        cost_val = reduce(lambda a, b: a + b, abs_diffs)
        return cost_val

    res = minimize(cost, [design_dict['R1'], design_dict['R2'], design_dict['C1'], design_dict['C2']], method='Nelder-Mead')
    print(list(map(lambda p: p[0](R1=res.x[0], R2=res.x[1], C1=res.x[2], C2=res.x[3]), sym_poles)))
    print(list(map(lambda p: p[1](R1=res.x[0], R2=res.x[1], C1=res.x[2], C2=res.x[3]), sym_poles)))
    print(res)