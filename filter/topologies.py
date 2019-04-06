from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sympy as sp
from scipy.signal import freqs
import numpy as np
from functools import partial
import scipy.optimize
import ahkab

from filter.specs import BA


# Base class for component and non-ideality values for a given topology stage
class TopologyValues:
    def __init__(self):
        self.Rbase = 10e3
        self.Cbase = 100e-15
        self.Av_base = 30  # from HW1
        self.Cload_base = 40e-15
        self.gamma_base = 2  # short-channel?
        self.gm_base = 50e-6
        self.ro_base = self.Av / self.gm


class SallenKeyValues(TopologyValues):
    def __init__(self, ro=False, bw=False):
        super().__init__()
        self.r1 = self.Rbase
        self.r2 = self.Rbase
        self.c1 = self.Cbase
        self.c2 = self.Cbase
        self.e1 = 100e3
        if ro:
            self.ro: Optional[float] = self.ro_base
        else:
            self.ro: Optional[float] = None
        assert bw is False, "TODO"
        self.bw: Optional[float] = None


class MFBValues(TopologyValues):
    def __init__(self):
        super().__init__()
        self.r1 = self.Rbase
        self.r2 = self.Rbase
        self.r3 = self.Rbase
        self.c1 = self.Cbase
        self.c2 = self.Cbase
        self.e1 = 100e3
        self.ro: Optional[float] = None
        self.bw: Optional[float] = None


class OTA3Values(TopologyValues):
    def __init__(self):
        super().__init__()
        self.r1 = self.Rbase
        self.c1 = self.Cbase
        self.c2 = self.Cbase
        self.gm = self.gm_base
        self.ro = self.ro_base
        self.bw: Optional[float] = None


class OTA4Spec(TopologyValues):
    def __init__(self):
        super().__init__()
        self.r1 = self.Rbase
        self.r2 = self.Rbase
        self.c1 = self.Cbase
        self.c2 = self.Cbase
        self.gm = self.gm_base
        self.ro = self.ro_base
        self.bw: Optional[float] = None


class Topology:
    def __init__(self, values: TopologyValues):
        self.values = values
        self.Rbase = 900
        self.Cbase = 9e-12
        self.sym_tf = None  # type: Dict[str, sp.Expr]
        self.sym_tf_symbols = []  # type: List[sp.Expr]

    def initial_guess(self, exclude: List[str]) -> List[float]:
        def inner():
            for s in self.sym_tf_symbols:
                if str(s)[0].upper() in exclude:
                    continue
                if str(s)[0:1].upper() == 'RO':
                    yield self.nonideal_dict['Av'] / self.nonideal_dict['gm']
                elif str(s)[0].upper() == 'R':
                    yield self.Rbase
                elif str(s)[0].upper() == 'C':
                    yield self.Cbase
                elif str(s)[0].upper() == 'G':
                    yield self.nonideal_dict['gm']
        return list(inner())

    def eval_tf(self, w: List[float], variables: List[float]) -> List[complex]:
        pass

    def construct_stage(self, c: ahkab.Circuit, in_node: str, out_node: str, suffix: str, ro: bool = False):
        pass


class OTA3(Topology):
    def __init__(self, values):
        super().__init__(values)
        self.spec = OTA3Spec(
            r1=self.Rbase,
            c1=self.Cbase,
            c2=self.Cbase,
            gm=self.nonideal_dict['gm'],
            ro=self.nonideal_dict['Av']/self.nonideal_dict['gm'],
            bw=1e8
        )

    def eval_tf(self, w: List[float], variables: List[float]) -> List[complex]:
        return list(map(lambda x: self.sym_gain_lambda(*variables, s=1j*x), w))

    def construct_lut(self, desired_filter: BA) -> List[Tuple[float, float, float]]:
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

    def construct_stage(self, c: ahkab.Circuit, in_node: str, out_node: str, suffix: str, ro: bool = False):
        # Add components
        c.add_resistor('R1_'+suffix, out_node, 'n1_'+suffix, self.spec.r1)
        c.add_isource('INR1_'+suffix, out_node, 'n1_'+suffix, dc_value=0, ac_value=0)
        c.add_capacitor('C1_'+suffix, out_node, c.gnd, self.spec.c1)
        c.add_capacitor('C2_'+suffix, 'n1_'+suffix, c.gnd, self.spec.c2)

        # Add OTA
        c.add_vccs('G1_'+suffix, c.gnd, 'n1_'+suffix, in_node, out_node, self.spec.gm)
        c.add_isource('ING1_'+suffix, c.gnd, 'n1_'+suffix, dc_value=0, ac_value=0)
        if ro:
            c.add_resistor('RO_'+suffix, 'n1_'+suffix, c.gnd, self.spec.ro)


class OTA4(Topology):
    def __init__(self):
        super().__init__()
        self.Rbase = 900
        self.Cbase = 9e-12
        self.spec = OTA4Spec(
            r1=self.Rbase,
            r2=self.Rbase,
            c1=self.Cbase,
            c2=self.Cbase,
            gm=self.nonideal_dict['gm'],
            ro=self.nonideal_dict['Av'] / self.nonideal_dict['gm'],
            bw=1e8
        )
        self.circuit, self.subs_dict, self.noise_srcs = build_lpf([FT.OTA4], [self.spec])
        self.sym_tf = run_sym(self.circuit, 'V1')
        self.sym_tf_symbols = list(filter(lambda s: str(s) != 's', self.sym_tf['gain'].free_symbols))
        self.sym_gain_lambda = sp.lambdify(self.sym_tf_symbols + [sp.symbols('s')], self.sym_tf['gain'])

    def eval_tf(self, w: List[float], variables: List[float]) -> List[complex]:
        return list(map(lambda x: self.sym_gain_lambda(*variables, s=1j*x), w))

    def construct_lut(self, desired_filter: BA) -> List[Tuple[float, float, float]]:
        w, h = freqs(desired_filter.B, desired_filter.A)

        def cost(y: List[float], C_val) -> float:
            sym_gain = list(map(lambda x: self.sym_gain_lambda(C1_0=C_val, C2_0=C_val, G1_0=y[1], R1_0=y[0], R2_0=y[0], s=1j*x), w))
            residuals = np.abs(np.subtract(sym_gain, h))
            weights = [1]*200
            return sum(np.multiply(residuals, weights))

        def gen_lut():
            for C_val in np.geomspace(start=100e-15, stop=1e-12, num=5):
                partial_cost = partial(cost, C_val=C_val)
                res = scipy.optimize.minimize(partial_cost, x0=[10e3, 100e-6], method='Nelder-Mead',
                               options={'maxfev': 1000, 'xatol': 1e-6, 'fatol': 1e-12, 'adaptive': True})
                print("C: {}, R: {}, gm: {}".format(C_val, res.x[0], res.x[1]))
                yield (C_val, res.x[0], res.x[0], res.x[1])
        return list(gen_lut())


