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
        self.ro_base = self.Av_base / self.gm_base


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
        self.ro: Optional[float] = None
        self.bw: Optional[float] = None


class OTA4Values(TopologyValues):
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

    def construct_stage(self, circuit: ahkab.Circuit, in_node: str, out_node: str, suffix: str):
        pass


class OTA3(Topology):
    def __init__(self, values: OTA3Values):
        super().__init__(values)

    def construct_stage(self, circuit: ahkab.Circuit, in_node: str, out_node: str, suffix: str):
        # Add components
        circuit.add_resistor('R1_'+suffix, out_node, 'n1_'+suffix, self.values.r1)
        circuit.add_isource('INR1_'+suffix, out_node, 'n1_'+suffix, dc_value=0, ac_value=0)
        circuit.add_capacitor('C1_'+suffix, out_node, circuit.gnd, self.values.c1)
        circuit.add_capacitor('C2_'+suffix, 'n1_'+suffix, circuit.gnd, self.values.c2)

        # Add OTA
        circuit.add_vccs('G1_'+suffix, circuit.gnd, 'n1_'+suffix, in_node, out_node, self.values.gm)
        circuit.add_isource('ING1_'+suffix, circuit.gnd, 'n1_'+suffix, dc_value=0, ac_value=0)
        if self.values.ro is not None:
            circuit.add_resistor('RO_'+suffix, 'n1_'+suffix, circuit.gnd, self.values.ro)


class OTA4(Topology):
    def __init__(self, values: OTA4Values):
        super().__init__(values)

