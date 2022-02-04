import numpy as np
from numba import njit, float64, int_
from numba.experimental import jitclass

pars_spec = [(v,float64) for v in ("pautoinf","p_seed", "lamda", "mu", "p_sus",)] + \
    [("N", int_),("T", int_)]
@jitclass(pars_spec)
class Parameters:
    """
    N, T, pautoinf, p_seed, lambda, mu, p_sus
    """
    def __init__(self, N, T, pautoinf, p_seed, lamda, mu, p_sus) -> None:

        self.pautoinf = pautoinf
        self.p_seed = p_seed
        self.lamda = lamda
        self.mu = mu
        self.p_sus = p_sus
        self.N = N
        self.T = T

def make_params(N,T,pautoinf, p_source, lamda, mu,p_sus=0.5):

    if p_source < 0:
        p_source = 1/N

    prob_seed = p_source / (2 - p_source)
    p_sus = p_sus * (1-prob_seed)

    return Parameters(N,T, pautoinf, prob_seed, lamda, mu, p_sus)

@njit()
def get_state_time(time, t0, d_inf):
    """ 
    convention: first time is 0, last time is T-1
    stays for t0 times in state S -> 0 ... t0-1
    - at time t0 is I
    - at time t0+d_inf is R
    """
    if time < t0:
        return 0
    if time < t0+d_inf:
        return 1
    return 2


def geometric_p_cut(p, maxT):
    probs = np.zeros(maxT)
    probs[1:] = p*((1-p)**np.arange(maxT-1))

    probs[-1] = 1 - probs[:-1].sum()

    return probs