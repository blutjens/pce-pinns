""" 
Polynomial Chaos Expansion
Author: Björn Lütjens (lutjens@mit.edu)
"""

import numpy as np
import numpy.matlib
import torch

import numpy.polynomial.hermite_e as H
from scipy.stats import norm
from scipy.special import factorial # Computes element-wise factorial function

import src.utils.plotting as plotting

def multi_indices(ndim=2, pce_dim=3, order='total_degree'):
    """
    Creates lattice of multi indices    

    Args:
        ndim int: Input dimension, e.g., number of parameters
        pce_dim int: Maximum polynomial degree of PCE polynomial bases
        order string: Total_order (l_1 norm(multi_indices)<pce_dim), full_order

    Returns:
        multi_indices np.array(ndim,n_multi_indices): Lattice of alpha vectors, that represent the order of polynomial basis 
        For example:
            ndim=3, pce_dim=2, order='total_degree' 
            multi_indices = np.array(   [0, 0, 0], 
                                        [1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1], 
                                        [2, 0, 0], 
                                        [1, 1, 0], 
                                        [1, 0, 1], 
                                        [0, 2, 0], 
                                        [0, 1, 1], 
                                        [0, 0, 2])
    """
    if order=='full_order':
        raise NotImplementedError('Computation of full-order multi_indices() is not implemented.')
    Mk = np.zeros((1,ndim))
    M = Mk.copy()
    for k in range(pce_dim-1):
        kronecker = np.kron(np.eye(ndim), np.ones((Mk.shape[0], 1)))
        Mk = numpy.matlib.repmat(Mk,ndim,1) + kronecker
        Mk = np.unique(Mk, axis=0)
        M = np.vstack((M, Mk))# M = [M; Mk];
    multi_indices = M.astype(int) # dim: ( n_multi_indices ), ndim
    
    return multi_indices

def Herm(p):
    """
    Return Hermite coefficients. Here, these are unit-vectors of degree p. 
    """
    coefs = [0] * (p+1)
    coefs[p] = 1
    return coefs

def herm_inner_product_at_x(h1, h2):
    """
    Return a function that evaluates a product of two hermite polynomials at x
    """
    return lambda x: H.hermeval(x, H.hermemul(h1, h2))

def gauss_hermite(x, deg, verbose=False):
    """
    Computes the Gaussian-Hermite polynomial of degree deg for all points in x 
    Source: https://numpy.org/doc/stable/reference/routines.polynomials.classes.html

    Args:
        x np.array(n_grid): Input array
    """
    domain = [x.min(), x.max()]
    # Get basis coefficients of "probabilists" hermite polynomial
    herm = np.polynomial.hermite_e.HermiteE.basis(deg, 
        domain=domain, window=domain)
    coefs = herm.coef

    y = herm(x)

    return y

def get_pce_coef(alpha_vec, mu_y, eigvals, eigvecs, verbose=True):
    """
    Calculates the PCE coefficient for a specific combination of polynomial degrees, alpha  

    Args:
        alpha_vec np.array(ndim,dtype=int): Realization of one multi-index, indicating the degree of each polynomial, herm_alpha_i;
            ndim is the number of stochastic dimensions, i.e., number of random variables. 
            Elements of alpha_vec, alpha_i, can bigger than ndim
        mu_y np.array(n_grid): Mean of target function
        eigvals np.array(ndim): Eigenvalues of covariance of target function
        eigvecs np.array((n_grid, ndim)): Eigenvectors of covariance of target function

    Returns:
        c_alpha np.array(n_grid): PCE coefficient of alpha_vec
    """   
    if verbose: print('alpha_vec', alpha_vec, alpha_vec.shape)

    ndim = alpha_vec.shape[0]
    n_grid = mu_y.shape[0]
    # Set up Gauss-Hermite quadrature for integration in nominator
    n_quad_pts = ndim**2#pce_dim**2 # number of sample points. Shouldn't this be \sqrt(pce_dim-1)?
    xi_quad, w_quad = H.hermegauss(n_quad_pts) # Get points and weights of Gauss-Hermite quadrature.

    # Calculate nominator
    y_herm = np.ones(n_grid)
    for a_idx, alpha_i in enumerate(alpha_vec):
        # Use KL-expansion with given mu_y, eig(cov_y), to approximate the exponential of the target function, exp(Y)
        exp_y = np.exp(np.sqrt(eigvals[a_idx]) * np.matmul(eigvecs[:,a_idx,np.newaxis],xi_quad[np.newaxis,:])) # dim: n_grid x n_quad_pts
        
        herm_alpha_i = H.hermeval(xi_quad, Herm(alpha_i)) # dim: n_quad_pts
        herm_w = np.multiply(herm_alpha_i, w_quad) # dim: n_quad_pts
        y_herm_alpha_i = np.matmul(exp_y, herm_w) # dim: n_grid
        
        y_herm = np.multiply(y_herm,y_herm_alpha_i) # dim: n_grid

    exp_mu_y = np.exp(mu_y) # dim: n_grid
    nominator = np.multiply(exp_mu_y, y_herm) # dim: n_grid

    # Calculate denominator
    comp_denom_analytically = True
    if comp_denom_analytically:
        denominator = np.sqrt(2*np.pi)*factorial(alpha_vec) # dim: ndim
        denominator = np.prod(denominator) 
    else:
        denominator = 1.
        for a_idx, alpha_i in enumerate(alpha_vec):
            herm_norm = 0
            for idx in range(n_quad_pts):
                herm_norm += herm_inner_product_at_x(Herm(alpha_i), Herm(alpha_i))(xi_quad[idx]) * w_quad[idx]
            denominator = denominator * herm_norm

    c_alpha = np.divide(nominator, denominator)
    
    return c_alpha

def sample_pce_torch(pce_coefs, alpha_indices, rand_inst=None,verbose=False):
    """
    Draw a sample from the PCE when the PCE coefficients are given as autodifferentiable torch tensor
    Note: batch_size can be replace with n_grid=1
    todo: move this function to pce.py  

    Args:
        pce_coefs torch.Tensor(batch_size, ndim): PCE coefficients
        alpha_indices np.array(ndim,pce_dim): Set of alpha indices
        rand_inst np.array(batch_size, ndim): Random instances used to reproduce same function between random instance and k (opt) 

    Returns:
        exp_y_pce torch.Tensor(batch_size, 1): Samples of the stochastic process, k=exp(y)=sum(pce_coefs*herm)
    """

    batch_size, n_alpha_indices = pce_coefs.size()
    exp_y_pce = torch.zeros(batch_size) # np.zeros(batch_size)
    ndim = alpha_indices.shape[1]
    # Sample one random variable per stochastic dimension
    # !! this currently assumes that all samples in batch are dependent, but the batches are independet!!
    if rand_inst is None:
        xi = np.random.normal(0,1,ndim) # one random variable per stochastic dimension
        xi = np.repeat(xi[np.newaxis,:], repeats=batch_size, axis=0)
    else:
        xi = rand_inst
        if verbose:
            _, uniq_ids = np.unique(xi,axis=0,return_index=True)
            print('PCE NN rand smpl xi', xi[np.sort(uniq_ids),:])
        
    herm_alpha_vec = np.zeros((batch_size, alpha_indices.shape[0]))
    # TODO: do all this in matrix multiplication!
    for b in range(batch_size):
        for a, alpha_vec in enumerate(alpha_indices):
            # Evaluate Gauss-Hermite basis polynomials of degree alpha_i at sampled position
            herm_alpha = np.zeros(ndim)#torch.zeros(ndim)#  
            for idx, alpha_i in enumerate(alpha_vec):# in range(pce_dim):
                herm_alpha[idx] = H.hermeval(xi[b, idx], Herm(alpha_i)) #dim: 1
            herm_alpha_vec[b,a] = np.prod(herm_alpha[:])
    # Linear combination of pce coefs gauss-hermite polynomial; in torch to enable autodiff
    herm_alpha_vec = torch.from_numpy(herm_alpha_vec).type(torch.float32)
    exp_y_pce = torch.diag(torch.matmul(pce_coefs[:,:],torch.transpose(herm_alpha_vec,0,1))) # dim: diag((batch_size x ndim) * (batch_size x ndim)^T)=(batch_size)

    return exp_y_pce.unsqueeze(1)

def polynomial_chaos_expansion(xgrids, 
    kl_dim=0, kl_mu_y=None, kl_eigvals=None, kl_eigvecs=None,
    poly_f=gauss_hermite, pce_dim=5, c_alphas=None,
    plot=False, verbose=False):
    """
    Computes the polynomial chaos expansion (PCE) of an n-dimensional process, assumed to be Gaussian   

    Args:
        xgrids: np.array(n_dim, n_grid): Gaussian samples of stochastic dimensions, n_dim, with equidistant grid spacing, n_grid
        poly_f function(): Polynomial basis function, e.g., Gauss Hermite
        pce_dim int: Maximum polynomial degree of PCE polynomial bases
        c_alphas np.array(n_grid, n_alpha_indices): If c_alphas are supplied, they will not be computed.
        plot boolean: If true, plots various quantities of interest

    Returns:
        exp_Y np.array(n1_grid,n2_grid): Exponential of approximated stochastic process, exp(Y)
        c_alphas np.array((n_grid, n_alpha_indices)): PCE coefficients
        xi np.array(ndim): Random instances used to sample the PCE  
        sample_pce function()->y,exp_y,trunc_err,coefs,xi: 
            Function that draws samples of log-permeability, Y, as approximated by PCE
    """
    log = {}
    if len(xgrids.shape)==1:
        xgrids = xgrids[np.newaxis, :]
    n_grid = xgrids.shape[1]

    # Calculate Multi Indices
    ndim = kl_dim # pce_dim
    pce_dim = kl_dim
    alpha_indices = multi_indices(ndim=ndim, pce_dim=pce_dim, order='total_degree')
    n_alpha_indices = alpha_indices.shape[0]

    # Calculate PCE coefficients
    if c_alphas is None:
        c_alphas = np.zeros((n_grid, n_alpha_indices))
        for a, alpha_vec in enumerate(alpha_indices):
            c_alphas[:,a] = get_pce_coef(alpha_vec=alpha_vec, mu_y=kl_mu_y, eigvals=kl_eigvals, eigvecs=kl_eigvecs)

    # Compute truncation error
    trunc_err = 0.

    # Draw a sample from the PCE with given PCE coefficients
    def sample_pce():
        """
        Function that samples the PCE with given PCE coefficents 
        todo: I think, it'd be cleaner to write this as PCE_sampler object and combine the code with sample_pce
        """
        exp_y_pce = np.zeros(n_grid)

        # Sample one random variable per stochastic dimension
        xi = np.random.normal(0,1,ndim)
        for a, alpha_vec in enumerate(alpha_indices):
            herm_alpha = np.zeros(ndim)
            # Evaluate Gaussian-Hermite basis polynomials of degree alpha_i at sampled position
            for idx, alpha_i in enumerate(alpha_vec):
                herm_alpha[idx] = H.hermeval(xi[idx], Herm(alpha_i)) # dim: ndim
            # Linear combination of PCE coefs, c, with product of basis polynomials
            exp_y_pce += c_alphas[:,a] * np.prod(herm_alpha) # dim: n_grid
        y_pce = np.log(np.where(exp_y_pce>0, exp_y_pce, 1.))
        #print('PCE rand smpl xi, k', xi, exp_y_pce[0])

        return y_pce, exp_y_pce, trunc_err, c_alphas, xi # TODO return XI
    y_pce, exp_y_pce, trunc_err, c_alphas, xi = sample_pce()


    if plot:
        plotting.plot_pce(xgrids, exp_y_pce, sample_pce, pce_dim, alpha_indices, c_alphas)

    return y_pce, exp_y_pce, trunc_err, c_alphas, xi, sample_pce

