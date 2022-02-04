
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
def c_logfvu(v,u, times, nodes, p0):
    #r = np.full(T,np.log(1-p0))
    r = np.log(1-p0)
    t0v = times[v][0]
    
    for k in nodes[v].neighs_in:
        if k == u:
            #print("eq")
            continue
        t0k = times[k][0]
        idx_v = nodes[k].neighs_out[v]
        lambs_kv = nodes[k].loglambs[idx_v]
        #for t in range(t0k, min(t0k+times[k][1],T)):
        #    r[t]+= lambs_kv[t]
        if t0v >= t0k+1 and t0v <= t0k+times[k][1]:
            r+= lambs_kv[t0v-1]
    return r


@nb.njit()
def calc_logB(nodes, times, u, T, p0):

    logB = np.zeros((T+2,T+1))
    nodd = nodes[u]
    for v in nodd.neighs_out:
        idx_v = nodd.neighs_out[v]
        #print(v)
        t0v = times[v][0]
        if t0v<=T and t0v>=1 and nodd.loglambs[idx_v][t0v-1] != 0:
            ## second part
            ## t0v <= T since it has to become infected, and not remain S
            ## t0v >= 1 since it has to be infected, not be the source
            fvu = np.exp(c_logfvu(v,nodd.i, t0v-1, times, nodes, p0))
            lam_uvt = nodd.loglambs[idx_v][t0v-1]
            #print(fvu)
        else:
            fvu = -2
            lam_uvt=-2
        for t0 in range(T+2):
            for di in range(1,T+2):
                t_max = min(t0+di-1,t0v-2)
                if t0 <= T:
                    ## if t0=T+1 it can't infect anyone
                    ## if t0=T it could, but we have no contacts at t=T
                    ll = nodd.sum_lam[v]
                    logB[t0,di-1] += ll
                    if t0 > 0:
                        logB[t0,di-1] -= nodd.lam_cs_f[v][t0-1]
                    if t_max+1 < T:
                        logB[t0,di-1] -= nodd.lam_cs_b[v][t_max+1]
                        
                if fvu >0 and t0v >=t0+1 and t0v <= t0+di:
                    logB[t0][di-1] += np.log(1-fvu*np.exp(lam_uvt))-np.log(1-fvu)

    return logB