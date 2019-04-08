from typing import List, Dict
import sympy as sp
from scipy.signal import freqs
import scipy.optimize
import numpy as np
from functools import partial
from joblib import Memory

from filter.specs import BA
from filter.topology_construction import build_lpf, run_sym
from filter.topologies import Topology

cachedir = './cache'
memory = Memory(cachedir, verbose=1)


class TopologyAnalyzer:
    def __init__(self, cascade: List[Topology]):
        self.cascade = cascade
        self.circuit, self.subs_dict, self.noise_srcs = build_lpf(cascade)
        self.sym_tf = run_sym(self.circuit, 'V1')
        sp.pprint(self.sym_gain)

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

# TODO: there should be a topology analyzer per topology to perform topology-specific fitting

