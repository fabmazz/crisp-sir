import numpy as np
from numba import njit, float64, int_
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.core import types

float_array = types.float64[:]
spec = [
    ("i", types.int_),
    ("neighs", types.DictType(types.int_, types.int_)),
    ("loglambs", types.ListType(float_array))
]
@jitclass(spec)
class Node:
    def __init__(self, i) -> None:
        self.i = i
        self.neighs = Dict.empty(key_type=types.int_,
            value_type=types.int_
        )
        self.loglambs = List.empty_list(float_array)

    def _check_neigh(self,j):
        if j in self.neighs:
            return self.neighs[j]
        else:
            return -1

    def set_loglambs(self, j:int, lambs:np.ndarray):
        if j in self.neighs:
            idx_j = self.neighs[j]
            self.loglambs[idx_j] = lambs
        else:
            idx_j = len(self.loglambs)
            self.neighs[j] = idx_j
            self.loglambs[idx_j].append(lambs)

    def add_contact(self, j:int, t:int, loglam: float, T:int):
        if j in self.neighs:
            idx_j = self.neighs[j]
        else:
            idx_j = len(self.loglambs)
            self.neighs[j] = idx_j
            self.loglambs.append(np.zeros(T))
        loglambs = self.loglambs[idx_j]
        loglambs[t] = loglam