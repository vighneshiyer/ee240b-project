from filter.topology_construction import build_lpf, run_sym
from filter.topologies import Topology


class TopologyAnalyzer():
    def __init__(self, topology: Topology):
        self.topology = topology
        self.circuit, self.subs_dict, self.noise_srcs = build_lpf([topology], [topology.values])
        self.sym_tf = run_sym(self.circuit, 'V1')
        self.sym_tf_symbols = list(filter(lambda s: str(s) != 's', self.sym_tf['gain'].free_symbols))
        self.sym_gain_lambda = sp.lambdify(self.sym_tf_symbols + [sp.symbols('s')], self.sym_tf['gain'])
