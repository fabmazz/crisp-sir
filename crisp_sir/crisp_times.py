from warnings import warn
import numpy as np
import numba as nb

from numba.typed import Dict

from .base import sample, geometric_p_cut, get_state_time

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
            #di goes from 1 to T+1
            obs_dict[i] = np.zeros((T+2,T+1))
        probs = np.log(conf_mat[s])
        o_mat = obs_dict[i]
        o_mat[t+1:] += probs[0]
        for t0 in range(t+1):
            di = t-t0
            o_mat[t0][:di] += probs[2]
            o_mat[t0][di:] += probs[1]

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
def c_logfvu(v,u, t, times, nodes, p0):
    #r = np.full(T,np.log(1-p0))
    r = np.log(1-p0)
    
    for k in nodes[v].neighs_in:
        if k == u:
            #print("eq")
            continue
        t0k = times[k][0]
        dinf_k = times[k][1]
        idx_v = nodes[k].neighs_out[v]
        lambs_kv = nodes[k].loglambs[idx_v]
        ## check if node k is infected
        if t >= t0k and t <= t0k+dinf_k-1:
            r+= lambs_kv[t]
    return r

@nb.njit()
def pi0(gamma, t0):
    if t0 == 0:
        return gamma
    else:
        return 1-gamma

@nb.njit()
def calc_logA(nodes, times, u, T, logp0s, logpdinf, p0, gamma):
    logA = np.zeros((T+2,T+1))

    logput = calc_logput(nodes, times, T, u)

    Ku = logput.cumsum()

    loglinf = np.log( 1-(1-p0)*np.exp(logput) ) - np.log(p0)
    for t0 in range(T+2):
        loga1 = np.log(pi0(gamma,t0))
        if t0>=1:
            loga1 += logp0s[t0]
            if t0<= T:
                loga1 +=loglinf[t0-1]
        if t0>=2:
            loga1 += Ku[t0-2]
        for dinf in range(1,T+2):
            logA[t0,dinf-1] = loga1 + logpdinf[dinf]
    return logA

@nb.njit()
def _c_lambsB(nodd, v, t0, t_max, T):
    ll = nodd.sum_lam[v]
    if t0 > 0:
        ll -= nodd.lam_cs_f[v][t0-1]
    if t_max+1 < T:
        ll -= nodd.lam_cs_b[v][t_max+1]

    return ll

@nb.njit()
def _calc_sum_nu(nus, t0, di, t0v):
    """
    Compute the extact sum in the first part
    of log B, without optimizations
    nus=loglambdas
    """
    r = 0.
    for t in range(t0, t0+di-1):
        if t <= t0v-2:
            r+= nus[t]
    return r

@nb.njit()
def calc_logB(nodes, times, u, T, p0):
    """
    times is the actual sample
    """

    logB = np.zeros((T+2,T+1))
    nodd = nodes[u]
    for v in nodd.neighs_out:
        idx_v = nodd.neighs_out[v]
        #print(v)
        t0v = times[v][0]
        nus = nodd.loglambs[idx_v]
        if t0v<=T and t0v>=1 and nodd.loglambs[idx_v][t0v-1] != 0:
            ## second part
            ## t0v <= T since it has to become infected, and not remain S
            ## t0v >= 1 since it has to be infected, not be the source
            fvu = np.exp(c_logfvu(v,u,t=t0v-1, times=times, nodes=nodes, p0=p0))
            lam_uvt = nodd.loglambs[idx_v][t0v-1]
            #print(fvu)
        else:
            fvu = -2
            lam_uvt=-2
        for t0 in range(T+2):
            for di in range(1,T+2):
                """t_max = min(t0+di-1,t0v-2)
                if t0 <= T:
                    ## if t0=T+1 it can't infect anyone
                    ## if t0=T it could, but we have no contacts at t=T
                    #ll = nodd.sum_lam[v]
                    logB[t0,di-1] += _c_lambsB(nodd, v, t0, t_max, T)
                """
                logB[t0, di-1] += _calc_sum_nu(nus,t0, di, t0v)
                        
                if fvu >0 and t0v >=t0+1 and t0v <= t0+di:
                    logB[t0][di-1] += np.log(1-fvu*np.exp(lam_uvt))-np.log(1-fvu)

    return logB


def crisp_step_probs(nodes, times_st, idx_u, T, logp0s, logpdinf, logC_dict, params):

    logp_sam = calc_logA(nodes, times_st, idx_u, T, logp0s, logpdinf=logpdinf,
        p0=params.pautoinf, gamma=params.p_seed)

    logp_sam += calc_logB(nodes, times_st, idx_u, T, p0=params.pautoinf)

    if idx_u in logC_dict:
        logp_sam+= logC_dict[idx_u]


    return np.exp(logp_sam-logp_sam.max()) #np.exp(logp_sam)

    

def sample_state(probs):

    #logprobs = crisp_step_probs(nodes, times_st, idx_u, T, logp0s, logpdinf, logC_dict, params)

    c_cont = probs.flags.c_contiguous == True
    f_cont = probs.flags.f_contiguous == True

    shape = tuple(probs.shape)

    ##flatten the array in place
    ## probs.shape = (np.prod(shape))
    pr_flat = probs.reshape(np.prod(shape))

    idx = sample(pr_flat)
    if c_cont:
        i1 = int(idx/shape[1])
        i2 = idx % shape[1]
    elif f_cont:
        warn("using fortran array order")
        i1 = int(idx/shape[0])
        i2 = idx % shape[0]
    else:
        raise RuntimeError("Cannot understand shape")

    return i1, i2, pr_flat[idx]

@nb.njit()
def record_stats(stats,u, t0, dinf, T):
    for t in range(T+1):
        s = get_state_time(t, t0, dinf)
        stats[u,t, s]+=1

    

def run_crisp(nodes, pars, seed, nsteps, obs_logC_term=None, burn_in=0, debug=False, state_times=None):
    T = pars.T
    N = pars.N

    p0 = pars.pautoinf
    mu = pars.mu
    try:
        seed.random()
        rng = seed
    except AttributeError:
        if debug: print("New generator")
        rng = np.random.RandomState(np.random.PCG64(seed))

    if state_times is None:
        state_times = rng.randint(0, T+2, (N,2))


    logp0s = np.log(geometric_p_cut(p0, T+2))
    logpdI = np.log(geometric_p_cut(mu, T+2))

    if obs_logC_term is None:
        obs_logC_term = make_obs_dict(list(), None, T=T)

    changes = []

    stats_times = np.zeros((N,2,T+2), dtype=np.int_)
    stats_st = np.zeros((N,T+1,3), dtype=np.int_)
    ##compute cache
    for i in nodes:
        nodes[i].calc_cache(T)
    
    for i_s in range(nsteps):
        u = int(rng.random()*N)

        probs = crisp_step_probs(nodes, state_times, u, T,
            logp0s=logp0s, logpdinf=logpdI, logC_dict= obs_logC_term, params=pars)

        t0, dinf, pr_nor = sample_state(probs=probs)
        ## the matrix of probs is t0=(0,T+2) and dinf = (0,T+1)
        ## shift extracted dinf
        dinf +=1
        state_times[u,0] = t0
        state_times[u, 1] = dinf

        #stats_times[u,0, t0] +=1
        #stats_times[u,1,dinf] += 1
        if i_s >= burn_in:
            record_stats(stats_st,u, t0, dinf, T)

        changes.append((u, t0, dinf, pr_nor))


    return state_times, stats_st, changes