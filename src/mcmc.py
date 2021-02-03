"""
MCMC samplers
"""
import numpy as np

import src.utils.plotting as plotting
from main import sample_diffeq

def gaussian_proposal(z, std=1.):
    """
    Gaussian proposal distribution.

    Draws new parameters from (multi-)variate Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    Args:
        z (np.array(n_params)): Input parameter / current solution
        sigma (float): Standard deviation of Gaussian distribution. Can be 
            scalar or vector of length(z)
    
    Returns: 
        z_candidate (np.array(n_params)): Candidate solution to parameter
        trans_prob_ratio (float): Ratio of transition probabilities 
    """
    n_params = z.shape[0]
    # Draw candidate solution
    z_candidate = np.random.normal(z, std, (n_params))

    # Calculate ratio of transition probabilities
    # p(z|z_candidate) / p(z_candidate|z)
    trans_ratio = 1.

    return z_candidate, trans_ratio 

def ln_gaussian(y, mean=np.zeros(2), std=np.ones(2)):
    """
    Computes log-likelihood of multiple 1D uncorrelated Gaussian prob. density fn at each grid point
    Source: https://en.wikipedia.org/wiki/Gaussian_function
    
    Args: 
        y np.array(n_grid): Grid points
        mean np.array(n_grid): Mean of gaussian at each grid point
        std np.array(n_grid): Std deviation at each grid point
    
    Returns:
        g np.array(n_grid): Likelihood of y
    """
    g = np.log(1.) - np.log(std * np.sqrt(2*np.pi)) + (-0.5 * np.power((y-mean),2) / np.power(std,2))
    return g

def gaussian_lnpost(u, u_obs, obs_noise=1e-4):
    """
    Computes likelihood of posterior, u, given measurements, u_obs, assuming Gaussian msmt noise
    
    Args:
        u (np.array(n_grid)): Solution, given candidate parameter
        u_obs (np.array(n_grid)): Measurements of solution 
        obs_noise (float): Variance of gaussian measurement noise
    
    Returns:
        lnpost (np.array(n_grid))    
    """
    n_grid = u.shape[0]
    std = np.sqrt(obs_noise * np.ones(n_grid))
    lnpost_i = ln_gaussian(y=u, mean=u_obs, std=std)
    lnpost = np.sum(lnpost_i) 
    return lnpost
    
def mh_sampler(z_init,
    proposal_fn=gaussian_proposal, proposal_fn_args={'std':1.},
    model=sample_diffeq, model_args={},
    lnpost_fn=gaussian_lnpost, lnpost_fn_args={'u_obs':None, 'obs_noise':1e-4}, 
    n_samples=1e4, warmup_steps=0, plot=True):
    """
    Uses MCMC Metropolis-Hastings to estimate, e.g., the KL random coefficients, z_init,
    given a prior, proposal_fn, and posterior, lnpost_fn, likelihood. The posterior
    is based on observed solutions of the model.   
    Based on: https://jellis18.github.io/post/2018-01-02-mcmc-part1/ 

    Args:
        z_init np.array(ndim): Start point for sampling with stochastic dimension, ndim
        proposal_fn fn: Computes proposal 
        proposal_fn_args dict(): Arguments to proposal function 
        model fn: Runs model
        model_args dict(): Arguments to model 
        lnpost_fn fn: Computes log-posterior
        lnpost_fn_args dict(): Computes log-posterior
        n_samples int: Number of samples
        warmup_steps int: Number of warmup steps

    Returns:
        chain np.array((n_samples, ndim)): Chain of samples of the approximate posterior of z, after warmup
        lnprobs np.array(n_samples): Log-probabilities to each sample in the chain
        accept_rates np.array(n_samples): Acceptance rates of each sample in the chain
        logs n_samples*[]
    """
    # initialize chain, acceptance rate and lnprob
    ndim = z_init.shape[0]
    chain = np.zeros((n_samples, ndim))
    lnprobs = np.zeros(n_samples)
    accept_rates = np.zeros(n_samples)
    logs = n_samples*[None]

    # Initialize prior and posterior    
    z_init = z_init
    model_args['z_candidate'] = z_init 
    u_init, logs[0], _ = model(**model_args)
    lnprob0 = lnpost_fn(u_init, **lnpost_fn_args)

    naccept = 0.
    for n in range(n_samples):
        if n%10==0: print('n', n)

        # Draw new candidate solution  
        z_candidate, trans_ratio = proposal_fn(z_init, **proposal_fn_args)

        # calculate logs-posterior, given model and data 
        model_args['z_candidate'] = z_candidate 
        u, logs[n], _ = model(**model_args)
        lnprob_candidate = lnpost_fn(u, **lnpost_fn_args)

        # Compute hastings ratio
        H = np.exp(lnprob_candidate - lnprob0) * trans_ratio

        # accept/reject step (update acceptance counter)
        uni_sample = np.random.uniform(0, 1)
        if uni_sample < H: # Accept
            z_init = z_candidate
            lnprob0 = lnprob_candidate
            naccept += 1.
        else:
            pass
        
        # update chain
        chain[n] = z_init
        lnprobs[n] = lnprob0
        accept_rates[n] = np.divide(naccept, float(n), out=np.zeros_like(naccept), where=float(n)!=0)

    # Prune warmup steps
    chain = chain[warmup_steps:]
    lnprobs = lnprobs[warmup_steps:]
    accept_rates = accept_rates[warmup_steps:]
    logs = logs[warmup_steps:]
    
    if plot:
        plotting.plot_mh_sampler(chain, lnprobs)
        plotting.plot_accept_rates(accept_rates)

    return chain, lnprobs, accept_rates, logs

def ensemble_kalman_filter(z_init,
    proposal_fn=gaussian_proposal, proposal_fn_args={'std':1.},
    model=sample_diffeq, model_args={},
    u_obs=None, obs_noise=1e-4,
    n_samples=1e4, n_batch=1, plot=True):
    """
    Uses to Ensemble Kalman Filter to estimate, e.g., the KL-expansion's random 
    variables, z. The model uses the starting point, z_init, proposal likelihood, 
    proposal_fn, and observations of the model, u_obs.

    Based on: https://piazza.com/class/kecbx4m6z2f46j?cid=105

    Args:
        z_init np.array(ndim): Start point for sampling with stochastic dimension, ndim
        proposal_fn fn: Computes proposal 
        proposal_fn_args dict(): Arguments to proposal function 
        model fn: Runs model
        model_args dict(): Arguments to model 
        u_obs np.array(n_obs, n_grid_obs): Observations of the model solution
        obs_noise float: Measurement noise
        n_samples int: Number of samples

    Returns:
        z_post_samples
        logs
    """
    # initialize chain, acceptance rate and lnprob
    ndim = z_init.shape[0]
    n_grid_obs = u_obs.shape[-1]

    z_samples = np.empty((n_samples, ndim))
    u_ests = np.empty((n_samples, n_grid_obs)) 
    logs = n_samples*[None]
    # Draw samples from joint p(model, parameter)
    for n in range(n_samples):
        if n%10==0: print('n', n)

        # Draw sample parameters  
        z_samples[n,:], _ = proposal_fn(z_init, **proposal_fn_args)

        # Propagate sample through model 
        model_args['z_candidate'] = z_samples[n,:]
        u_ests[n,:], logs[n], _ = model(**model_args)
        # Add measurement noise
        u_ests[n,:] += np.random.normal(loc=0.,scale=obs_noise,size=n_grid_obs)

    # Compute sample covariance matrix 
    ## Sanity check: cov_y * cov_y_inv does return the identity; cov is symmetric
    cov = np.cov(z_samples, u_ests, rowvar=False)
    cov_zy = cov[:ndim,ndim:]
    cov_y_inv = np.linalg.inv(cov[ndim:, ndim:]) 
    
    # Compute Kalman Gain
    G = np.matmul(cov_zy, cov_y_inv)

    # Update each parameter sample 
    z_post_samples = np.empty(z_samples.shape) 
    for n in range(n_samples):
        z_post_samples[n,:] = z_samples[n,:] + np.matmul(G, u_obs - u_ests[n,:])

    # Generate posterior predictive samples
    for n in range(n_samples):
        if n%10==0: print('n', n)
        model_args['z_candidate'] = z_post_samples[n,:]
        _, logs[n], _= model(**model_args)

    if plot:
        plotting.plot_ensemble_kf(z_post_samples)

    return z_post_samples, logs

def infer_model_param_w_mcmc(kl_dim, u_obs, model=sample_diffeq, model_args=dict(), 
    use_mh_sampler=True, use_ensemble_kf=False, n_samples=1000, warmup_steps=1000,
    plot=True):
    """
    Use an MCMC method to infer a random parameter, e.g., the KL-expansion's random 
    variable, z, that best explains the model observations, u_obs. 

    Args:
        kl_dim int: Dimension of inference parameter
        u_obs np.array(n_msmts): Solution, u, at measurement locations
        model fn:**model_args->u_obs, logs, sample_y: Continuous function returning model evaluations, u.
        model_args dict(): Dictionary of arguments to model() 
        use_mh_sampler bool: Use Metropolis-Hastings as MCMC method
        use_ensemble_kf bool: Use Ensemble Kalman Filter as MCMC method
        n_samples int: Number of samples of MCMC method. 
        plot bool: If true, creates plot

    Returns:
        logs n_samples*[]: n samples of the model
    """
    obs_noise = 0.0001 # 1e-4
    model_args['z_candidate'] = z_init

    if use_mh_sampler:
        # Infer KL-expansion's random variable, z, with MH-MCMC sampler
        z_init = np.random.normal(0., 1., (kl_dim))
        prop_std = np.sqrt(0.01)
        _, _, _, logs = mh_sampler(z_init,
            proposal_fn=mcmc.gaussian_proposal, proposal_fn_args={'std':prop_std},
            model=sample_diffeq, model_args=model_args,
            lnpost_fn=gaussian_lnpost, lnpost_fn_args={'u_obs':u_obs, 'obs_noise':obs_noise}, 
            n_samples=n_samples, warmup_steps=warmup_steps, plot=plot)

    elif use_ensemble_kf:
        # Infer KL-expansion's random variable, z, with Ensemble Kalman Filter
        z_init = np.zeros(kl_dim)
        prop_std = 0.1 #np.sqrt(1.)#np.sqrt(0.01)
        _, logs = ensemble_kalman_filter(z_init,
            proposal_fn=gaussian_proposal, proposal_fn_args={'std':prop_std},
            model=sample_diffeq, model_args=model_args,
            u_obs=u_obs, obs_noise=obs_noise, 
            n_samples=n_samples, plot=plot)

    return logs
