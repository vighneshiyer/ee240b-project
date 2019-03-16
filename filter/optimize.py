from typing import Dict, List, Callable, TypeVar
from filter.specs import BA
from filter.topology_analysis import SallenKeySpec
import sympy as sp
from functools import reduce
import inspect
from scipy.optimize import minimize, basinhopping
import numpy as np
from collections import OrderedDict


T = TypeVar('T')


def flatten(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]


def fit_filter_circuit(desired_tf: BA, symbolic_tf: Dict[str, List[sp.Expr]], design_vars: List[SallenKeySpec]) -> Dict[str, float]:
    """
    Use a generic optimizer to fit a symbolic transfer function to a desired transfer function
    :param desired_tf: The system-level transfer function that is desired with p poles and z zeros
    :param symbolic_tf: A dictionary with keys 'poles' and 'zeros' which contain lists of
        symbolic expressions for the poles and zeros respectively
    :param design_vars: A list of Sallen Key design variables, used as an initial optimizer guess.
        the list should contain a set of design variable structures for each filter stage
    :return: Dict from variable name to optimized value
    """
    sym_poles = list(map(lambda p: sp.lambdify(p.free_symbols, p, dummify=False), symbolic_tf['poles']))  # type: List[Callable]
    sym_poles_vars = list(map(lambda p: inspect.getfullargspec(p).args, sym_poles))  # type: List[List[str]]
    goal_poles = desired_tf.to_zpk().P
    assert len(sym_poles) == len(goal_poles),\
        "Number of symbolic poles {} isn't equal to number of poles in desired tf {}".format(len(sym_poles), len(goal_poles))

    var_assns = OrderedDict()
    for expr_vars in sym_poles_vars:
        for var in expr_vars:
            stage_num = int(str(var)[-1])  # TODO: generalize this, or make the stage parameter passing more systematic
            var_name = str(var)[:-2].lower()
            var_assns[var] = getattr(design_vars[stage_num], var_name)
    assert set(var_assns.keys()) == set(flatten(sym_poles_vars))

    # y is a list of variable assignments in the same order as var_assns
    def cost(y) -> float:
        this_assn = OrderedDict()
        for i, val in enumerate(y):
            this_assn[list(var_assns.keys())[i]] = val
        # This dictionary comprehension removes the keys that don't map to any variables in each pole expression
        actual_poles = map(lambda e: e[0](**{k: v for k,v in this_assn.items() if k in e[1]}), zip(sym_poles, sym_poles_vars))
        abs_diffs = map(lambda x: np.abs(x[0] - x[1]), zip(goal_poles, actual_poles))
        cost_val = reduce(lambda a, b: a**2 + b**2, abs_diffs)
        return cost_val

    print("Calling optimizer")
    # TODO: this call is non-deterministic, given the same cost function and x0, different optimization results are reached
    # This seems to happen because the solver shrinks the simplex from different corners every time
    # The best way to fix this is to make the cost function more determined, right now there are too many variables that are related
    res = minimize(cost, x0=list(var_assns.values()), method='Nelder-Mead', options={'maxfev': 10000, 'xatol': 1e-3, 'fatol': 1e-12, 'adaptive': False})
    #res = basinhopping(cost, x0=list(var_assns.values()), minimizer_kwargs={'method': 'Nelder-Mead'})
    print(res)
    final_assn = {k: v for k, v in zip(var_assns.keys(),res.x)}
    return final_assn
