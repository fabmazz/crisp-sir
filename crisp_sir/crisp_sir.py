import time
import numpy as np
from numba import njit, jit


from .base import get_state_time, sample, set_numba_seed
from .observ import *





def geom_prob(p0,T,nosum=False):
    probs = (1-p0)**(np.arange(0,T)-1)*p0
    probs[0] = 0.
    if not nosum:
        probs[-1] = 1-sum(probs[:-1])
    return probs

@njit()
def calc_logp_ut(contacts, u, allstates, lamda, verbose=False):
    """
    Precompute p_ut
    """
    T = allstates.shape[1]
    logp_ut = np.zeros(T)
    contacts_u = contacts[(contacts[:,2]==u)]
    if verbose:
        print("Num c: ", len(contacts_u))
    
    for v,tup in enumerate(contacts_u):
        t = int(tup[0])
        ## from v to u
        v = int(tup[1])
        #lam = tup[3]
        if allstates[v,t] == 1:
            logp_ut[t]+= np.log(1-lamda)
    return logp_ut

@njit()
def p_no_inf(v,t, conts, state, lam, pautoinf, idx_inf=None, v_inf=None):
    """
    Compute the probability of no infection at different times
    """
    nskip = 0
    logp_noinf = np.log(1-pautoinf)
    for tup in conts:
        if t!=int(tup[0]) or v!= int(tup[2]):
            nskip+=1
            continue
        i = int(tup[1])
        if i == idx_inf:
            if v_inf:
                logp_noinf += np.log(1-lam)
        #lam=tup[3]
        elif state[i,t] == 1:
            logp_noinf += np.log(1-lam)
    return np.exp(logp_noinf), nskip
@njit()
def _num_neigh_inf(v,t,contacts,state):
    """
    Count the number of infected neighbors at time t
    """
    nskip = 0
    nc = 0
    for tup in contacts:
        if t!=int(tup[0]) or v!= int(tup[2]):
            nskip+=1
            continue
        i = int(tup[1])
        if state[i,t] == 1:
            nc +=1
    return nc, nskip
@njit()
def calc_B_ratio(contacts, u, lamda, p0, T, state, verbose=False):
    """
    Compute the ratio of Bs
    """
    logp_b = np.zeros(T)
    #logp_b_noinf = np.ones_like(T_crisp)
    conts_cache = None
    told = -1
    contacts_u_out = contacts[contacts[:,1]==u]
    #
    totnskip = 0

    for tup in contacts_u_out:
        
        t = int(tup[0])
        if t > told:
            told = t
            #conts_cache = contacts[contacts[:,0]==t]
            conts_cache = contacts[contacts[:,0]==t]
        elif t < told and told >= 0:
            raise ValueError("contacts unsorted in time")
            #print("Time ",t)
        
        assert u == int(tup[1])
        v = int(tup[2])
        if state[v,t] == 0:

            ## count the contribution from the 
            ## number of infected 
            num_inf_v, nsk = _num_neigh_inf(v, t, conts_cache, state)
            
            if state[u,t] == 1:
                # we have counted one more
                 num_inf_v -= 1

            f_uinf = (1-p0)*np.exp((num_inf_v+1)*np.log(1-lamda))
            f_unoinf = (1-p0)*np.exp(num_inf_v*np.log(1-lamda))

            if verbose:
                #if np.abs(np.log(1-f_uinf)) > 20 or \
                #    np.abs(np.log(1-f_unoinf)) > 20:
                #    print(u,v,t,f" f {v} inf: {f_uinf}, no inf: {f_unoinf}")
                print(t,"NC: ", len(conts_cache))
            totnskip += nsk
            #print(f_uinf,f_unoinf)
            if state[v,t+1] == 0:
                ## not infected in next time instant
                logp_b[t] += np.log(f_uinf) - np.log(f_unoinf)
            elif state[v,t+1] ==1:
                ## infected this time interval
                logp_b[t] += np.log(1-f_uinf)-np.log(1-f_unoinf)

    return logp_b, totnskip


@njit()
def calc_prob_times(T_crisp, gamma, p0inf, Ap1, Linf, prec, logp_b, psus, logC_term=None):
    #val_times = np.zeros((numtstates,2),dtype=np.int_)
    #p_states = np.zeros(numtstates)
    probs = np.zeros((T_crisp,T_crisp+1))
#    idx = 0
    #sump = 0
    #idx_logC = logC_term[0]
    #vals_logC = logC_term[1]
    
    for t0 in range(0,T_crisp):
        max_dinf = T_crisp -t0
        
        for dinf in range(1, min(T_crisp+1-t0, T_crisp+1)):
            #print(t0,dinf)
            # A1 = gamma if source
            #    = Ap(t0)*l0(t0)*Linf(t0)*(1-gamma) else
            if t0 == 0:
                logA = np.log(gamma)
            else:
                logA = np.log(p0inf[t0])
                logA += Ap1[t0]+np.log(Linf[t0])
                logA += np.log(1-gamma)
                if t0 == T_crisp-1:
                    logA += np.log(psus)
                else:
                    logA += np.log(1-gamma-psus)
            

            #if t0 < T_crisp-1:
            if dinf < max_dinf:
                logA += np.log(prec[dinf])
            else:
                logA += np.log(prec[dinf:T_crisp+1].sum())
            

            logB = 0            
            for ti in range(t0,t0+dinf):

                logB+= logp_b[ti]

            if logC_term is not None:
                logC = logC_term[t0,dinf]
            else:
                logC = 0

            pst = np.exp(logA + logB + logC)
            probs[t0,dinf] = pst
            
            if np.isnan(pst):
                raise ValueError("Nan value for probability")
            if pst >= np.inf:
                raise ValueError("Infinite probability")

    return probs
@njit()
def _calc_a_terms(T_crisp, p0, logp_ut):
    Linf = np.ones(T_crisp)
    Ap1 = np.zeros(T_crisp)
    #Ap1[2:] = logp_ut.cumsum()[:-1]
    #Ap1[2:] = logp_ut.cumsum()[:-1]
    for t in range(1,T_crisp):
        Linf[t] = 1+(1-p0)*(1-np.exp(logp_ut[t-1]))/p0
        if t>1: Ap1[t] = Ap1[t-1]+ logp_ut[t-2]

    return Linf, Ap1

def _calc_crisp_step(contacts, state, prec, p0inf, logC_term, u, pars, T_crisp,):
    
    logp_ut = calc_logp_ut(contacts, u, state, pars.lamda)
    ## precompute l_infected and l_infected'
    Linf, Ap1 = _calc_a_terms(T_crisp, pars.pautoinf, logp_ut=logp_ut)

    logp_b,ns = calc_B_ratio(contacts, u,
                                    pars.lamda,
                                    p0=pars.pautoinf,
                                    T=T_crisp,
                                    state=state,
                                verbose=False)
    #nctsskip[st_idx] = ns
    #nctsskip#logp_b[-1]=logp_b[-2]
    if sum(logp_b >= np.inf) > 0 or np.isnan(logp_b).sum() > 0 :
        raise ValueError("Have infs")
        
    lC_term = logC_term[u] if u in logC_term else None

    probs = calc_prob_times(T_crisp, pars.p_seed,
                            p0inf=p0inf,
                            Ap1=Ap1,
                            Linf=Linf,
                            prec=prec,
                            logp_b=logp_b,
                            psus=pars.p_sus,
                            logC_term=lC_term)

    #if sum(p_ex >= np.inf) > 0 or np.isnan(p_ex).sum() > 0 :
    #    raise ValueError("Have infs")
    probs_idx = probs.nonzero()
    probs_v = probs[probs_idx]
    probs_v /= probs_v.sum()
    ## extract idcs
    choice = sample(probs_v)
    value_choice = (probs_idx[0][choice], probs_idx[1][choice])
    change = (u,*value_choice, probs_v[choice]/np.max(probs_v))

    return change

@njit()
def _write_state(u, t0, dinf, T, state):
    for t in range(T+1):
        state[u][t] = get_state_time(t, t0, dinf)

def run_crisp(pars, observ, contacts, num_samples, mat_obs, burn_in=0,
     seed=None, state=None,
     start_inf=False):
    """
    Run the whole CRISP method
    on the SIR problem
    """
    
    N = pars.N
    T = pars.T
    T_crisp = pars.T+1
    p0inf = geom_prob(pars.pautoinf,T+1,nosum=False)
    #p0 = pars.pautoinf
    prec = geom_prob(pars.mu,T+2,nosum=False)

    logC_term = make_observat_term(observ,pars.N,T=pars.T,
                                     mat_obs=mat_obs)
    
    if seed is not None:
        set_numba_seed(seed)
        randgen = np.random.RandomState(seed)
    else:
        randgen = np.random
    #ninf_start = max(1, int(pars.p_seed*N))
    tinf = randgen.randint(0,T) 
    ul = randgen.randint(0,N)
    if state is None:
        state = np.zeros((N,T+1),dtype=np.int8)
        ## starting condition is not really important
        if(start_inf):
            state[ul,tinf:] = 1

    stats = np.zeros((N,T+1,3),dtype=np.int_)
    changes = []
    contacts_CRISP = contacts[:,:3].astype(int)
    #contacts = contacts_CRISP
    #nctsskip = np.zeros(NUM_STEPS, dtype=np.int_)
    NUM_STEPS = num_samples
    tim_l = time.time()
    r = np.eye(3,dtype=np.int_)
    for st_idx in range(NUM_STEPS):
        u = int(randgen.rand()*N)
        
        change = _calc_crisp_step(contacts=contacts_CRISP,
                state=state,
                prec=prec,
                p0inf=p0inf,
                logC_term=logC_term,
                u=u,
                pars=pars,
                T_crisp=T_crisp)
        
       
        ## write state 
        _write_state(u, t0=change[1], dinf=change[2], T=pars.T, state=state)

        changes.append(change)
        
        #st_1h = r[state]
        #trace_states[st_idx]=st_1h.sum(0)
        
        if st_idx >= burn_in:
            #save_state(stats=stats, state=state)
            stats[u]+=r[state[u]]
        if (st_idx%400)==0:
            print(f"{st_idx:6d} /{NUM_STEPS:6d}", end="\r")

    print("Finished   ")
    return state, stats, changes