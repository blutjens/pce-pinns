"""
Density functions
"""
import numpy as np
import scipy.stats as st


def power_kernel(xgrid, var_y=0.3, lengthscale=0.3, p=1.):
    '''
    Computes the power kernel (TODO: find out actual name)
    
    Args:
        xgrid np.array(n_grid): Array of grid points
        var_y (int): Variance of the output stochastic process
        lengthscale (int): Lengthscale of the kernel
        p (int): Order of the kernel
    
    Returns:
        cov (np.array((n_grid,n_grid))): Covariance matrix
    '''
    n_grid = xgrid.shape[0]
    x1 = np.repeat(xgrid[:,np.newaxis], n_grid, axis=1)
    x2 = np.repeat(xgrid[np.newaxis,:], n_grid, axis=0)
    cov = var_y * np.exp(-1./p * np.power(np.absolute(x1 - x2) / lengthscale, p))

    return cov

def init_gaussian_process(xgrid, y_mean, y_var, lengthscale=0.3, order=1.):
    """
    Initializes parameters of a gaussian process

    Args:
        xgrid np.array(n_grid): 1D grid points
        y_mean float: Constant mean
        y_var float: Constant

    Returns:
        mu_y np.array(n_grid): Mean over x
        cov_y np.array((n_grid,n_grid)): Covariance matrix
    """
    #import GPy
    #vis_kernel1 = GPy.kern.RBF(input_dim=1, variance=3., lengthscale=1)
    n_grid = xgrid.shape[0]
    mu_y = np.repeat(y_mean, n_grid)

    # Compute covariance matrix
    cov = power_kernel(xgrid, var_y=y_var, lengthscale=lengthscale, p=order)

    return mu_y, cov

def calc_stats(u):
    """
    Calculates stats of set of samples u
    
    Args:
        u np.array(n_samples): Function of x
    
    Returns:
        u_stats dict(): Statistics of u
    """
    n_samples = u.shape[0]
    
    # Calculate statistics
    conf_bnds = 0.95
    u_stats = {
        'ux_cum_mean': np.zeros((n_samples)),
        'ux_cum_std' : np.zeros((n_samples)),
        'ux_cum_sem' : np.zeros((n_samples)),
        'ux_cum_var' : np.zeros((n_samples)),
        'ux_conf_int' : np.zeros((n_samples, 2)),
    }

    for n in range(n_samples):
        u_stats['ux_cum_mean'][n] = np.mean(u[:n+1])
        u_stats['ux_cum_std'][n] = np.std(u[:n+1], ddof=1) # Use n-1 in denominator for unbiased estimate. 
        u_stats['ux_cum_sem'][n] = st.sem(u[:n+1])
        u_stats['ux_cum_var'][n] = np.var(u[:n+1])
        if n>0:
            u_stats['ux_conf_int'][n,:] = st.t.interval(conf_bnds, n-1, loc=u_stats['ux_cum_mean'][n], scale=st.sem(u[:n+1]))

    """ Compute confidence interval                        
    for n in range(n_samples):
        est_n = u_stats['ux_cum_mean'][:n+1]
        #if n==0:
        #    ux_conf_int[n,:] = np.array([u_stats['ux_cum_mean'][:n+1], u_stats['ux_cum_mean'][:n+1]])[0]
        #else:
        ux_conf_int[n,:] = st.t.interval(conf_bnds, est_n.shape[0]-1, loc=np.mean(est_n), scale=st.sem(est_n))
    """
    return u_stats
