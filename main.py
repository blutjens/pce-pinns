import numpy as np 
import matplotlib.pyplot as plt
import pickle # For data storage
import argparse

import src.utils.plotting as plotting
import src.utils.logging as logging
import src.utils.densities as densities
import src.utils.utils as utils

import src.rom.kl_expansion as kl # Reduced order models
import src.rom.pce as pce
import src.diffeq as diffeq 
import src.mcmc as mcmc

def sample_approx_gaussian_process(xgrid, 
    mu_y, cov, expansion='KL', 
    kl_dim=3, z_candidate=None, 
    pce_dim=5, pce_coefs=None, 
    plot=True):
    '''Samples from a gaussian process

    Computes a function that samples a gaussian process, given by mu_y and cov,
    via a reduced order model, e.g., a Karhunen-Loeve expansion

    Args:
        xgrid np.array(n_grid): 1D grid points
        mu_y np.array(n_grid): Mean over x
        cov_y np.array((n_grid,n_grid)): Covariance matrix of gaussian process
        expansion string: Chosen reduced order model
        kl_dim int: Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        z_candidate np.array(kl_dim): If not None, these are candidates for the new random variables of the KL-exp
        pce_dim int: Maximum polynomial degree of polynomial chaos expansion, typically equal to kl_dim
        pce_coefs np.array(n_grid, pce_dim: If PCE coefficients are given, they will not be computed

    Returns:
        Y np.array(n_grid): Sample of low dim gaussian process
        exp_Y np.array(n_grid): Sample of low dim exp-gaussian process
        kl_trunc_err float: KL truncation error 
        coefs np.array((n_grid, pce_dim)): Polynomial chaos expansion coefficients 
        rand_insts np.array(kl_dim or pce_dim): Random instances used in expansion 
    '''
    n_grid = xgrid.shape[0]
    # Compute expansion
    Y, exp_Y, kl_trunc_err, kl_eigvals, kl_eigvecs, rand_insts, sampling_fn = kl.kl_expansion(xgrid, mu_y, cov, kl_dim, z_candidate=z_candidate)
    if expansion == 'KL':
        kl_trunc_err = kl_trunc_err
    elif expansion == 'polynomial_chaos':
        Y, exp_Y, kl_trunc_err, coefs, rand_insts, sampling_fn = pce.polynomial_chaos_expansion(xgrid, 
            kl_dim=kl_dim, kl_mu_y=mu_y, kl_eigvals=kl_eigvals, kl_eigvecs=kl_eigvecs,
            pce_dim=kl_dim, plot=plot, c_alphas=pce_coefs)
    return Y, exp_Y, kl_trunc_err, coefs, rand_insts, sampling_fn

def sample_diffeq(diffeq, xgrid, 
    y_gp_mean, y_gp_cov, 
    kl_dim, expansion, z_candidate, pce_dim, 
    x_obs=None,
    sample_y=None,
    pce_coefs=None):
    """Samples the differential equation

    Computes a sample of the stochastic elliptic equation. The solution depends on the 
    stochastic, log-permeability, Y, given by sample_y. If sample_y is not given, we 
    assume it's modeled by a Gaussian process.

    Args: 
        diffeq src.diffeq.StochDiffEq: Differential equation that shall be sampled, e.g., stochastic diffusion equation
        xgrid np.array(()): Evaluation grid
        y_gp_mean np.array(n_grid): Gaussian process mean as fn of x
        y_gp_cov np.array((n_grid,n_grid)): Gaussian process covariance matrix as fn of x
        kl_dim int: Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        expansion
        z_candidate np.array(kl_dim): KL random variables
        pce_dim int: Maximum polynomial degree of PCE polynomial bases        
        x_obs np.array(n_msmts): Grid of observation points. If not None, only values at x_obs are returned.
        sample_y function()->y,exp_y,kl_trunc_err,coefs, rand_insts: 
            Function that draws samples of log-permeability, Y
        pce_coefs np.array(n_grid, n_alpha_indices): PCE coefficients
 
    Returns:
        u_obs np.array(n_msmts): Solution, u, at measurement locations
        vals dict(): Dictionary of model results, e.g., Y, exp_Y, u, kl_trunc_err, coefs, rand_insts
        sample_y fn(): Function that samples log-permeability, y
    """
    vals = {}
    # Sample log-permeability, Y
    if sample_y is not None:
        vals['Y'], vals['exp_Y'], vals['kl_trunc_err'], vals['coefs'], vals['rand_insts'] = sample_y()
    else:
        # Assume log-permeability is a gaussian process
        vals['Y'], vals['exp_Y'], vals['kl_trunc_err'], vals['coefs'], vals['rand_insts'], sample_y = sample_approx_gaussian_process(xgrid, 
            mu_y=y_gp_mean, cov=y_gp_cov,
            kl_dim=kl_dim, expansion=expansion, z_candidate=z_candidate, 
            pce_dim=pce_dim, pce_coefs=pce_coefs)

    # Compute permeability
    vals['k'] = vals['exp_Y']

    # Compute solution
    vals['u'] = stochDiffEq.diffusioneqn(xgrid, k=vals['k'])[:,0]

    # Reduce solution to observed values
    if x_obs is None:
        u_obs = vals['u']
    else:
        # TODO: use src.utils.utils.get_fn_at_x()
        obs_idxs = np.zeros((x_obs.shape[0]))
        for i, x_o in enumerate(x_obs):
            obs_idxs[i] = np.abs(xgrid - x_o).argmin().astype(int)
        obs_idxs = obs_idxs.astype(int)
        u_obs = vals['u'][obs_idxs]

    return u_obs, vals, sample_y
 
def get_n_model_samples(model=sample_diffeq, model_args={}, 
    n_samples=2, 
    parallel=False,
    path_load_simdata=None, path_store_simdata='pce.pickle'):
    """
    Returns n samples of the model
    
    Args:
        model fn:**model_args->u_obs, logs, sample_y: Continuous function returning model evaluations, u.
        model_args dict(): Dictionary of arguments to model() 
        parallel bool: If true, sample the model in parallel; TODO: implement and test
        path_load_simdata string: Path to n model samples
        path_store_simdata string: Path to store model samples
    
    Returns:
        logs n_samples*[]: n samples of the model
    """
    # Load simulation data instead of 
    if path_load_simdata is not None:
        with open(path_load_simdata, 'rb') as handle:
            logs = pickle.load(handle)            
    else:
        # if args.parallel: 
        #    model_r, model_tasks = utils.parallel.init_preprocessing(fn=model, parallel=True)
        logs = n_samples*[None]
        # Compute approximation, e.g., PCE, to sample_y
        _, _, model_args['sample_y'] = model(**model_args)
        # Generate samples
        for n in range(n_samples):
            if n%10==0: print('n', n)
            # Sample solution
            _, logs[n], _ = model(**model_args)

        # Parse parallel tasks
        # model_tasks = utils.parallel.get_parallel_fn(model_tasks)
        # for n in range(n_samples):
        #    _, logs[n] = model_tasks[n]
        
        # Store data
        with open(path_store_simdata, 'wb') as handle:
            pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return logs 

def approx_pce_coefs_w_nn(xgrid, model, model_args, 
    coefs_target, y_gp_mean, y_gp_cov, kl_dim, pce_dim,
    n_samples, plot=True):
    """
    Approximate PCE coefficients with neural network
    """

    coefs_nn, _ = get_param_nn(xgrid=xgrid, y=coefs_target[0,:,:], n_epochs=20, 
        target_name="pce_coefs", plot=plot)

    # Get PCE sampling function with given neural net coefs
    _, _, _, _, _, model_args['sample_y'] = sample_approx_gaussian_process(xgrid, 
        mu_y=y_gp_mean, cov=y_gp_cov,
        kl_dim=kl_dim, expansion='polynomial_chaos', z_candidate=None, 
        pce_dim=pce_dim, pce_coefs=coefs_nn)

    # Draw samples from PCE with neural net-based PCE coefficients 
    logs = n_samples * [None]
    for n in range(n_samples):
        u_ests, logs[n], _ = sample_diffeq(**model_args)
        
    k, Y, u, _, _, _ = logging.parse_logs(logs)

    return k, Y, u

def approx_model_param_w_nn(xgrid, diffeq, k_target, k_true, 
    est_param_nn, kl_dim, pce_dim, rand_insts, n_samples, plot=True):
    """Solves diffeq with NN-based parameters 

    Approximates a model parameter in diffeq, e.g., permeability, k, with a 
    neural network (NN). The NN is trained on observations of k, k_target 
    or k_true. After the NN estimates the parameter multiple samples of diffeq are created.
    
    Args:
    Returns:
    """
    # TODO: try to move the query of multi_indices into pce.py
    alpha_indices = pce.multi_indices(ndim=kl_dim, pce_dim=pce_dim, order='total_degree')
    #n_samples = 150#k.shape[0]#20
    x_in = np.repeat(xgrid[np.newaxis,:], repeats=n_samples, axis=0)

    # Approximate k measurements
    if est_param_nn=='k_true':
        k_target = np.repeat(k_true[np.newaxis,:],repeats=n_samples, axis=0) # Observations
    #alpha_indices = alpha_indices[:,4:] # Omit constant pce coefficient
    # TODO: TEST IF BATCH SIZE CAN BE > n_grid
    pce_coefs, ks_nn = get_param_nn(xgrid=x_in, y=k_target, target_name=est_param_nn, 
        alpha_indices=alpha_indices, rand_insts=rand_insts,
        batch_size=151, n_epochs=100, n_test_samples=n_samples, plot=plot)
    if True:
        plotting.plot_k_vs_ks_nn(xgrid, k_target, ks_nn)
    # Compute log-permeability
    ys_nn = np.log(np.where(ks_nn>0, ks_nn, 1.))

    # Compute solution
    us_nn = np.zeros((n_samples, n_grid))
    for n in range(n_samples):
        us_nn[n,:] = diffeq.diffusioneqn(xgrid, k=ks_nn[n,:])[:,0]

    return ks_nn, ys_nn, us_nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='project 1-2')
    # Differential equation
    parser.add_argument('--rand_flux_bc', action = "store_true",
            help='Use random flux for van neumann condition on left boundary.')
    # Reduced order model
    parser.add_argument('--pce', action = "store_true",
            help='Polynomial chaos expansion.')
    parser.add_argument('--pce_dim', default=4, type=int,
            help='Maximum degree of polynomials in PCE or eigenvectors in KL.')
    parser.add_argument('--eval_kl_trunc_err', action = "store_true",
            help='Evaluation Karhunen-Loeve truncation error.')
    # MCMC
    parser.add_argument('--mcmc', action="store_true",
            help='Uses MCMC to estimate the permeability parameter from observations of the solution.')
    parser.add_argument('--mh_sampler', action="store_true",
            help='Solve inverse problem with MCMC metropolis-hastings.')
    parser.add_argument('--ensemble_kf', action="store_true",
            help='Solve inverse problem with ensemble kalman filter.')
    parser.add_argument('--nn_pce', action = "store_true",
            help='Use a neural net to approximate the polynomial chaos coefficients.')
    parser.add_argument('--n_samples', default=1000, type=int,
            help='Number of samples in forward and inverse solution.')
    parser.add_argument('--warmup_steps', default=1000, type=int,
            help='Number of warmup steps for MCMC.')
    # NN approximation
    parser.add_argument('--est_param_nn', default='pce_coefs', type=str,
            help='Name of parameter than shall be estimated by neural net, e.g., "pce_coefs", "k", "k_true".')
    parser.add_argument('--path_load_simdata', default=None, type=str,
            help='Path to logged simulation data, e.g., pce.pickle.')
    # General
    parser.add_argument('--parallel', action = "store_true",
            help='Enable parallel processing.')

    args = parser.parse_args()

    ## INIT

    # Get xgrid from ground-truth msmts
    x_obs, u_obs, k_true, xgrid = diffeq.get_msmts(plot=True)
    n_grid = xgrid.shape[0]
    n_obs = x_obs.shape[0] 

    np.random.seed(0)

    # Init stochastic differential equation
    stochDiffEq = diffeq.StochDiffEq(xgrid=xgrid, rand_flux_bc=args.rand_flux_bc, 
        injection_wells=args.mcmc)

    # Set prior for log-permeability, Y
    y_gp_mean, y_gp_cov = densities.init_gaussian_process(xgrid, y_mean=1., y_var=0.3, lengthscale=0.3, order=1.)

    # Init reduced order model of log-permeability, Y 
    kl_dim = args.pce_dim
    kl_dims = np.array((kl_dim,))
    if args.pce:
        expansion = 'polynomial_chaos'
        pce_dim = args.pce_dim
        z_init = None
    else:
        expansion = 'KL'
        pce_dim = 0
        z_init = np.random.normal(0., 1., (kl_dim))
        if args.eval_kl_trunc_err:
            kl_dims = 10*np.arange(10)+9

    # Init surrogate model of differential equation
    model_args = {'diffeq':stochDiffEq, 'xgrid':xgrid, 
        'y_gp_mean':y_gp_mean, 'y_gp_cov':y_gp_cov, # Log-permeability
        'kl_dim':kl_dim, 'expansion':expansion, 'z_candidate':z_init, 'pce_dim':pce_dim, # Expansion
        'x_obs':x_obs}

    ## RUN

    # Estimate various parameters with a neural network
    if args.nn_pce:
        import torch
        from src.neural_net import get_param_nn
        torch.manual_seed(0)

        # Get dataset
        logs = get_n_model_samples(model=sample_diffeq, model_args=model_args, 
            n_samples=args.n_samples, path_load_simdata=args.path_load_simdata)
        k, _, u, kl_trunc_errs, coefs, rand_insts = logging.parse_logs(logs)

        # Use NN
        if args.est_param_nn=='pce_coefs':
            k, _, u = approx_pce_coefs_w_nn(xgrid=xgrid, model=sample_diffeq, model_args=model_args, 
                coefs_target=coefs, y_gp_mean=yp_gp_mean, y_gp_cov=y_gp_cov, 
                kl_dim=kl_dim, pce_dim=pce_dim, n_samples=args.n_samples, plot=True)

        elif args.est_param_nn=='k' or args.est_param_nn=='k_true':
            k, _, u = approx_model_param_w_nn(xgrid=xgrid, diffeq=stochDiffEq, k_target=k, 
                k_true=k_true, est_param_nn=args.est_param_nn, kl_dim=kl_dim, pce_dim=pce_dim,
                rand_insts=rand_insts, n_samples=args.n_samples, plot=True)

    # Estimate model parameters with MCMC
    elif args.mcmc:
        logs = mcmc.infer_model_param_w_mcmc(kl_dim=kl_dim, u_obs=u_obs, model=sample_diffeq, 
            model_args=model_args, use_mh_sampler=args.mh_sampler, use_ensemble_kf=args.ensemble_kf, 
            n_samples=args.n_samples, warmup_steps=args.warmup_steps, plot=True)
        k, _, u, _, _, _ = logging.parse_logs(logs)

    ## PLOT

    if args.eval_kl_trunc_err: 
        plotting.plot_kl_trunc_err(kl_dims, kl_trunc_errs)

    plotting.plot_k(xgrid=xgrid, k=k, k_true=k_true, y_gp_mean=y_gp_mean, y_gp_cov=y_gp_cov)

    # Plot solution of differential equation with given, k
    u_true = stochDiffEq.diffusioneqn(xgrid, k=k_true)[:,0]
    at_x = 0.0
    u_at_x = utils.get_fn_at_x(xgrid=x_obs, fn=u, at_x=at_x)
    u_stats_at_x = densities.calc_stats(u=u_at_x)
    plotting.plot_sol(xgrid, u, u_at_x, u_stats_at_x, at_x, xgrid, u_true)