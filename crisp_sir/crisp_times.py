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
        #for t0 in range(T+2):
        #    for dinf in range(1,T+2):
        #        o_mat[t0][dinf-1] += probs[get_state_time(t,t0,dinf)]

    """for iu in obs_dict:
        mat = obs_dict[iu]
        for t0 in range(T+2):
            mat[t0][T+2-t0:] = -np.inf"""
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

    log_put = calc_logput(nodes, times, T, u)

    Ku = log_put.cumsum()
    logp0 = np.log(1-p0)

    loglinf = np.log( 1-(1-p0)*np.exp(log_put) ) 
    # - np.log(p0)
    for t0 in range(T+2):
        loga1 = np.log(pi0(gamma,t0))
        if t0>=1 and t0<= T:
            # we have no contacts at t=T,
            # so the last time it can be infected
            # is t=T-1
            loga1 +=loglinf[t0-1]
        if t0>=2:
            loga1 += (t0-1)*logp0
            loga1 += Ku[t0-2]
        #lim = min(T+3-t0, T+2)
        for dinf in range(1,T+2):
            #if dinf == lim-1:
            #    logA[t0,dinf-1] = loga1 + np.log(np.exp(logpdinf[dinf:]).sum())
            #else:
                logA[t0,dinf-1] = loga1 + logpdinf[dinf]
        #logA[t0][T+2-t0:] = np.nan
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
    for t in range(t0, t0+di):
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
            lam_uvt=np.nan
        for t0 in range(T+2):
            #l = min(T+3-t0, T+2)
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



    return np.exp(logp_sam-np.max(logp_sam)) #np.exp(logp_sam)

    

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

def sample_probs_nan(probs):
    
    idx_good = np.logical_not(np.isnan(probs))
    idcs = np.where(idx_good)
    pr_sam = probs[idx_good]

    idx_sam = sample(pr_sam)
    return idcs[0][idx_sam], idcs[1][idx_sam], pr_sam[idx_sam]    

@nb.njit()
def record_stats(stats,i, t0, dinf, T, value):
    for t in range(T+1):
        s = get_state_time(t, t0, dinf)
        stats[i,t, s]+=value

    

def run_crisp(nodes, pars, seed, nsteps, obs_logC_term=None, burn_in=0, debug=False, init_state=None):
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

    if init_state is None:
        conf_times = rng.randint(int(0.2*T),int(0.8*T), (N,2))
    else:
        conf_times = init_state.copy()

    if burn_in > nsteps:
        raise ValueError("Burn-in steps higher than MC steps")


    logp0s = np.log(geometric_p_cut(p0, T+2))
    logpdI = np.log(geometric_p_cut(mu, T+2))

    if obs_logC_term is None:
        obs_logC_term = make_obs_dict(list(), None, T=T)

    changes = []


    stats_times = np.zeros((N,T+2,T+1), dtype=np.int_)
    count_states_SIR = np.zeros((N,T+1,3), dtype=np.int_)
    ##compute cache
    for i in nodes:
        nodes[i].calc_cache(T)
    
    for i_s in range(nsteps):
        u = int(rng.random()*N)

        probs = crisp_step_probs(nodes, conf_times, u, T,
            logp0s=logp0s, logpdinf=logpdI, logC_dict= obs_logC_term, params=pars)

        probs /= np.sum(probs)
        assert probs.shape == (T+2,T+1)
        t0, dinf, pr_nor = sample_state(probs=probs)
        ## the matrix of probs is t0=(0,T+2) and dinf = (0,T+1)
        ## shift extracted dinf
        dinf +=1
        #if pr_nor > 0.9 and u not in obs_logC_term:
        #    print(f"move {i_s}:  {u} to t0 {t0}, dinf {dinf} has p {pr_nor}")
        #    print("conf: ", conf_times)
        conf_times[u,0] = t0
        conf_times[u, 1] = dinf


        if i_s >= burn_in:
            stats_times[u, t0, dinf-1] += 1
            #record_stats(count_states_SIR,u, t0, dinf, T)

        changes.append((u, t0, dinf, pr_nor))


    return conf_times, stats_times, changes


@nb.njit()
def transform_counts(counts_ts, N, T):
    counts_SIR = np.zeros((N,T+1,3),dtype=np.int_)
    for i in range(N):
        for t0 in range(T+2):
            for dinf in range(1,T+2):
                v = counts_ts[i][t0][dinf-1]
                record_stats(counts_SIR, i, t0, dinf, T, v)
    return counts_SIR