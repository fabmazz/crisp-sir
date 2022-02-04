import numpy as np
from numba import njit, float64, int_
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.core import types

float_array = types.float64[:]
spec = [
    ("i", types.int_),
    ("neighs_out", types.DictType(types.int_, types.int_)),
    ("loglambs", types.ListType(float_array)),
    ("neighs_in", types.Set(types.int_)),
    ("sum_lam",types.DictType(types.int_, types.float64)) ,
    ("lam_cs_f",types.DictType(types.int_, float_array)) ,
    ("lam_cs_b",types.DictType(types.int_, float_array)),
]
@jitclass(spec)
class Node:
    def __init__(self, i) -> None:
        self.i = i
        self.neighs_out = Dict.empty(key_type=types.int_,
            value_type=types.int_
        )
        self.loglambs = List.empty_list(float_array)
        self.neighs_in = set((1,))
        self.neighs_in.remove(1)
        self.sum_lam = Dict.empty(key_type=types.int_,
            value_type=types.float64
        )
        self.lam_cs_f = Dict.empty(key_type=types.int_,
            value_type=float_array
        )
        self.lam_cs_b = Dict.empty(key_type=types.int_,
            value_type=float_array
        )

    def _check_neigh(self,j):
        if j in self.neighs_out:
            return self.neighs_out[j]
        else:
            return -1

    def set_loglambs(self, j:int, lambs:np.ndarray):
        if j in self.neighs_out:
            idx_j = self.neighs_out[j]
            self.loglambs[idx_j] = lambs
        else:
            idx_j = len(self.loglambs)
            self.neighs_out[j] = idx_j
            self.loglambs[idx_j].append(lambs)

    def add_contact(self, j:int, t:int, loglam: float, T:int):
        if j in self.neighs_out:
            idx_j = self.neighs_out[j]
        else:
            idx_j = len(self.loglambs)
            self.neighs_out[j] = idx_j
            self.loglambs.append(np.zeros(T))
        loglambs = self.loglambs[idx_j]
        loglambs[t] = loglam
    def add_incoming_c(self, j:int):
        if j not in self.neighs_in:
            self.neighs_in.add(j)

    def calc_cache(self, T:int):
        for j in self.neighs_out:
            idj = self.neighs_out[j]
            self.lam_cs_f[j] = np.zeros(T)
            self.lam_cs_b[j] = np.zeros(T)
            lam_f = self.lam_cs_f[j]
            lam_b = self.lam_cs_b[j]
            lam_b[T-1] = self.loglambs[idj][T-1]
            lam_f[0] = self.loglambs[idj][0]
            for t in range(1,T):
                lam_f[t] = lam_f[t-1] + self.loglambs[idj][t]
                lam_b[T-t-1] = lam_b[T-t] + self.loglambs[idj][T-t-1]
            
            self.sum_lam[j] = self.loglambs[idj].sum()

kt = Node.class_type.instance_type
#@njit()
def make_nodes_contacts(cts, T, nodes=None):
    if nodes is None:
        nodes =  Dict.empty(key_type=types.int_,
                            value_type=kt)
    for c in range(len(cts)):
        #i = int(ct.i)
        t=int(cts[c][0])
        i=int(cts[c][1])
        j=int(cts[c][2])
        lam = float(cts[c][3])
        if i not in nodes:
            nodes[i] = Node(i)
        node = nodes[i]
        node.add_contact(j,t, np.log(1-lam), T)
        if j not in nodes:
            nodes[j] = Node(j)
        nodes[j].add_incoming_c(i)
        
    return nodes