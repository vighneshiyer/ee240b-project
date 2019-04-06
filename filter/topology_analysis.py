from typing import List
import sympy as sp
from scipy.signal import freqs
import scipy.optimize
import numpy as np
from functools import partial

from filter.specs import BA
from filter.topology_construction import build_lpf, run_sym
from filter.topologies import Topology


class TopologyAnalyzer:
    def __init__(self, cascade: List[Topology]):
        self.cascade = cascade
        self.circuit, self.subs_dict, self.noise_srcs = build_lpf(cascade)
        self.sym_tf = run_sym(self.circuit, 'V1')
        self.sym_tf_symbols = list(filter(lambda s: str(s) != 's', self.sym_tf['gain'].free_symbols))
        self.sym_gain_lambda = sp.lambdify(self.sym_tf_symbols + [sp.symbols('s')], self.sym_tf['gain'])
        sp.pprint(self.sym_tf['gain'])

    def eval_tf(self, w: List[float], variables: List[float]) -> List[complex]:
        return list(map(lambda x: self.sym_gain_lambda(*variables, s=1j*x), w))

    def initial_guess(self, exclude: List[str]) -> List[float]:
        def inner():
            for s in self.sym_tf_symbols:
                if str(s)[0].upper() in exclude:
                    continue
                if str(s)[0:1].upper() == 'RO':
                    yield self.cascade[0].values.ro_base
                elif str(s)[0].upper() == 'R':
                    yield self.cascade[0].values.Rbase
                elif str(s)[0].upper() == 'C':
                    yield self.cascade[0].values.Cbase
                elif str(s)[0].upper() == 'G':
                    yield self.cascade[0].values.gm_base
        return list(inner())

    def construct_lut(self, desired_filter: BA) -> List[List[float]]:
        w, h = freqs(desired_filter.B, desired_filter.A)

        def cost(y: List[float], C_val) -> float:
            sym_gain = list(map(lambda x: self.sym_gain_lambda(C1_0=C_val, C2_0=C_val, G1_0=y[1], R1_0=y[0], s=1j*x), w))
            return np.linalg.norm(sym_gain - h)

        def gen_lut():
            for C_val in np.geomspace(start=1e-15, stop=1e-12, num=10):
                partial_cost = partial(cost, C_val=C_val)
                res = scipy.optimize.minimize(partial_cost, x0=[10e3, 30e-6], method='Nelder-Mead',
                                              options={'maxfev': 1000, 'xatol': 1e-3, 'fatol': 1e-6, 'adaptive': True})
                print("C: {}, R: {}, gm: {}".format(C_val, res.x[0], res.x[1]))
                yield (C_val, C_val, res.x[0], res.x[1])
        return list(gen_lut())
