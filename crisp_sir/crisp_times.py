
import re

import numpy as np
import numba as nb

from numba.typed import Dict, List

#@nb.njit()
## do not jit since it is computed once
def make_obs_dict(obs_list, conf_mat, T):
    """
    obs_list: observations, (node, state, time)
    conf_mat: confusion matrix p[obs][state]
    T: max time (==t_limit)
    """
    obs_dict = Dict.empty(nb.int_, nb.float64[:,:])

    for o in obs_list:
        i = int(o[0])
        s = int(o[1])
        t = int(o[2])
        if i not in obs_dict:
            obs_dict[i] = np.zeros((T+2,T+2))
        probs = np.log(conf_mat[s])
        o_mat = obs_dict[i]
        o_mat[t+1:] += probs[0]
        for t0 in range(t+1):
            di = t-t0
            o_mat[t0][:di+1] += probs[2]
            o_mat[t0][di+1:] += probs[1]

    return obs_dict

@nb.njit()
def calc_logput(nodes, times, T, i):
    logput = np.zeros(T)

    for j in nodes[i].neighs_in:
        idx_i = nodes[j].neighs_out[i]
        #print(j, idx_i,nodes[j].loglambs[idx_i].shape)
        t_s = times[j][0]
        t_e = t_s + times[j][1]
        logput[t_s:t_e] += nodes[j].loglambs[idx_i][t_s:t_e]

    return logput

@nb.njit()
def c_logfvu(v,u, T, times, nodes, p0):
    r = np.full(T,np.log(1-p0))
    for k in nodes[v].neighs_in:
        if k == u:
            #print("eq")
            continue
        t0k = times[k][0]
        idx_v = nodes[k].neighs_out[v]
        lambs_kv = nodes[k].loglambs[idx_v]
        for t in range(t0k, min(t0k+times[k][1],T)):
            r[t]+= lambs_kv[t]
    return r