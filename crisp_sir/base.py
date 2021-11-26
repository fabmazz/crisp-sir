from numba import njit

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