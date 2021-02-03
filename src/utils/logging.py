"""
Utils for logging
"""
import numpy as np

def parse_logs(logs):
    """
    Take in logs and return params
    """
    # Init params
    n_samples_after_warmup = len(logs)
    n_grid = logs[0]['u'].shape[-1] 
    u = np.zeros((n_samples_after_warmup, n_grid))
    Y = np.zeros((n_samples_after_warmup, n_grid))
    k = np.zeros((n_samples_after_warmup, n_grid))
    kl_trunc_errs = np.empty((n_samples_after_warmup,1))
    n_stoch_disc = logs[0]['coefs'].shape[-1] # e.g., n_alpha_indices for PCE, or kl_dim for KL-E
    coefs = np.empty((n_samples_after_warmup, n_grid, n_stoch_disc))
    stoch_dim = logs[0]['rand_insts'].shape[-1]
    rand_insts = np.empty((n_samples_after_warmup, stoch_dim))
    # Copy logs into params
    for n, log in enumerate(logs):
        k[n,:] = log['k']
        Y[n,:] = log['Y']
        u[n,:] = log['u']
        kl_trunc_errs[n,0] = log['kl_trunc_err']
        coefs[n,:,:] = log['coefs']
        rand_insts[n,:] = log['rand_insts']
    return k, Y, u, kl_trunc_errs, coefs, rand_insts

