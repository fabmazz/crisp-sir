# %%
import numpy as np
import pandas as pd

# %%
import crisp_sir

# %%
pwrong=1e-6
mat_obs =crisp_sir.make_mat_obs(p_wrong_obs=pwrong)

# %%
obs_df = pd.read_csv("obs_test.csv")

# %%
contacts_df = pd.read_csv("contacts_test.csv")

# %%
t_limit = contacts_df.t.max()+1

# %%
N = contacts_df[["i","j"]].to_numpy().max()+1

# %%
obs_list = obs_df[["i","obs","t"]].to_records(index=False)

# %%
logC = crisp_sir.make_observat_term(obs_list, N, T=t_limit+1, mat_obs=mat_obs)

# %%
pars = crisp_sir.make_params(N, T=t_limit+1, p_source=1/N, 
    pautoinf=1e-6,
    lamda =contacts_df.lam.mean(), mu=0.,
    p_sus=0.5)

# %%
pars

# %%
cts_np = contacts_df[["t","i","j","lam"]].to_numpy()

# %%
crisp_sir.crisp_sir.set_numba_seed(201)
state2, stats2, changes2 = crisp_sir.run_crisp(pars, obs_list,cts_np,
                                  seed=201,
                                   num_samples=10000,
                                   burn_in=200,
                                   mat_obs=mat_obs,
                                )

# %%
margs_crisp_m = stats2 / stats2.sum(-1)[...,np.newaxis]


# %%
margs_crisp_m.shape

# %%
with np.load("margs_compare.npz") as f:
    margs_compare = f["marginals"]

# %%

assert np.abs(margs_crisp_m - margs_compare).max() <= np.finfo(float).eps

