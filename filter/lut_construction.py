import numpy as np
from typing import List, Tuple
from joblib import Memory
from scipy.signal import freqs
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from filter.specs import BA
from filter.topology_construction import Topology

cachedir = './cache'
memory = Memory(cachedir, verbose=1)


# Generically optimize a topology for a desired transfer response
def naive_topology_optim(desired_filter: BA, topology: Topology):
    w, h = freqs(desired_filter.B, desired_filter.A)

    def cost(y: List[float]) -> float:
        sym_gain = topology.eval_tf(w, y)
        return np.linalg.norm(sym_gain - h)

    res = minimize(cost, x0=topology.initial_guess([]), method='Nelder-Mead',
                   options={'maxfev': 10000, 'xatol': 1e-3, 'fatol': 1e-12, 'adaptive': False})
    print(res)
    print(topology.sym_tf_symbols)

    real_h = (topology.eval_tf(w, res.x))
    plt.semilogx(w, 20*np.log10(np.abs(real_h)))
    plt.show()


def construct_ota_lut(w: int=1) -> Tuple[List[str], List[List[float]]]:
    # Header: vstar	idc gm ro av wbw Cgg Cdd Css vgs drain_eff
    with open('filter/nmoschar.csv') as csv_file:
        header = csv_file.readline().strip().split(',')
        data = []
        for line in csv_file:
            data.append(np.array(list(map(np.float64, line.strip().split(',')))))

    scale = np.array([1, w, w, 1/w, 1, 1, w, w, w, 1, 1])
    scaled_data = list(map(lambda x: x * scale, data))
    return header, scaled_data
