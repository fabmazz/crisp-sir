from collections import defaultdict
import numpy as np
from numba import njit
from .base import get_state_time

def make_mat_obs(p_wrong_obs):
    return np.diag(np.ones(3)-3*p_wrong_obs)+np.ones((3,3))*p_wrong_obs

@njit()
def observ_term_i(obs_list_i, mat_pobs, t0, d_inf, log=True):
    """
    Give the observation term, 
    mat_po represents the confusion matrix
    observations must be (state,time) sorted by time index
    mat_pobs is index by [obs][state]
    """
    tot = 0
    for tup in obs_list_i:
        st = int(tup[0])
        t = int(tup[1])
        state_u = get_state_time(t, t0, d_inf)
        tot += np.log(mat_pobs[st][state_u])
    if log:
        return tot #np.exp(tot)
    else:
        return np.exp(tot)

@njit()
def _calc_logpobs_u(T_crisp, obs_u, mat_obs):
    logpobs_u = np.full((T_crisp,T_crisp+1),np.nan)
    for t0 in range(0, T_crisp):
            for dinf in range(1,min(T_crisp-t0+1,T_crisp+1)):
                logpobs_u[t0,dinf] = observ_term_i(obs_u,mat_obs,t0,dinf)
    return logpobs_u

def make_observat_term(observ_list, N, T, mat_obs):
    """
    Observations in format
    (node, state, time)
    """
    logpobsall= {}
    obs_by_node = defaultdict(list)
    T_crisp = T+2
    ## nota che T=t_limit+1 e' il numero dei tempi del sistema
    if len(observ_list) < 1:
        return logpobsall
    for obs_tup in observ_list:
        node = obs_tup[0]
        
        obs_by_node[node].append(tuple(obs_tup)[1:])
        
        if obs_tup[2] > T:
            raise ValueError("observation time too large")

    for node in obs_by_node.keys():
        obsl = obs_by_node[node]
        if len(obsl) == 0:
            continue

        #print(node,obs_u)
        obs_u = np.array(obsl).astype(int)
        

        logpobsall[node] = _calc_logpobs_u(T_crisp=T_crisp, obs_u=obs_u, mat_obs=mat_obs)
        
    return logpobsall