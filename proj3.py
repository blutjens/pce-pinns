import numpy as np 
import matplotlib.pyplot as plt
import pickle # For data storage
import argparse
import scipy.stats as st
from scipy.special import factorial


class StochDiffEq():
    def __init__(self):
        pass

    def diffusioneqn(self, xgrid, F, k, source, rightbc):
        """
        Solve 1-D diffusion equation with given diffusivity field k
        and left-hand flux F. Domain is given by xgrid (should be [0,1])
        
        Inputs:
        xgrid (np.array(n_grid)): grid points
        F (float): flux at left-hand boundary, k*du/dx = -F 
        source (np.array(n_grid)): source term, either a vector of values at points in xgrid or a constant
        rightbc (float): Dirichlet BC on right-hand boundary
        Outputs:
        usolution (np.array(xgrid)): solution
        """
        N = xgrid.shape[0] # Number of grid points
        h = xgrid[N-1]-xgrid[N-2] # step size; assuming uniform grid

        # Set up discrete system f = Au + b using second-order finite difference scheme
        A = np.zeros((N-1, N-1)) 
        b = np.zeros((N-1,1)) 
        if np.isscalar(source): 
            f = -source * np.ones((N-1,1))
        else:
            f = -source[:N-1,np.newaxis] # [:N] 

        # diagonal entries
        A = A - 2.*np.diag(k[:N-1]) - np.diag(k[1:N]) - np.diag(np.hstack((k[0], k[:N-2]))) 

        # superdiagonal
        A = A + np.diag(k[:N-2], 1) + np.diag(k[1:N-1], 1) 

        # subdiagonal
        A = A + np.diag(k[:N-2], -1) + np.diag(k[1:N-1], -1)

        A = A / (2. * np.power(h,2))

        # Treat Neumann BC on left side
        A[0,1] = A[0,1] + k[0] / np.power(h,2)
        b[0] = 2.*F / h # b(1) = 2*F/h;

        # Treat Dirichlet BC on right side
        b[N-2] = rightbc * (k[N-1] + k[N-2]) / (2.*np.power(h,2)) 

        # Solve it: Au = f-b
        uinternal = np.linalg.solve(A, (f-b))

        usolution = np.vstack((uinternal, rightbc)) 

        return usolution

    def ridge_plot_prob_dist_at_sample_x(self, xs, uxs):
        '''
        Creates ridge plot, plotting a probability distribution over all values in uxs 
        Source: https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
        Input
        xs np.array(n_curves)
        uxs np.array(n_curves, n_samples)
        '''
        import matplotlib.gridspec as grid_spec

        gs = grid_spec.GridSpec(len(xs),1)
        fig = plt.figure(figsize=(16,9))

        axs = []
        for i, x in enumerate(xs):
            x = xs[i]

            # creating new axes object
            axs.append(fig.add_subplot(gs[i:i+1, 0:]))

            # plotting the distribution
            histogram = axs[-1].hist(uxs[i,:],  density=True, color="blue", alpha=0.6)#bins='auto',#, lw=1) bins='auto', density=True, 
            #axs[-1].fill_between(range(uxs[i,:].shape[0]), uxs[i,:] , alpha=1,color=colors[i])

            # setting uniform x and y lims
            axs[-1].set_ylim(0,20)
            u_min = np.min(uxs[:,:])#.mean(axis=0))
            u_max = np.max(uxs[:,:])#.mean(axis=0))
            axs[-1].set_xlim(u_min, u_max)

            # make background transparent
            rect = axs[-1].patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
            axs[-1].set_yticklabels([])

            # Label x-axis
            if i == len(xs)-1:
                axs[-1].set_xlabel(r"$u(x)$", fontsize=16)
            else:
                axs[-1].set_xticklabels([])

            spines = ["top","right","left","bottom"]
            for s in spines:
                axs[-1].spines[s].set_visible(False)

            # Label y-axis
            adj_x = x#.replace(" ","\n")
            axs[-1].set_ylabel(r'$x=$'+str(x))
            #text(-0.02,0,adj_x,fontweight="bold",fontsize=14,ha="right")


        gs.update(hspace=-0.0)

        fig.text(0.07,0.85,"prob dist at sample x",fontsize=20)
        #plt.title("prob dist at sample x",fontsize=20)

        plt.tight_layout()
        plt.savefig('figures/plot_prob_dist_at_sample_x.png')

        plt.show()

    def plot_sol(self, xgrid, u, ux, ux_cum_mean, ux_cum_std, ux_cum_sem, ux_conf_int, ux_cum_var, x, xgrid_true, u_true=None):
        """
        Plot solution and statistics
        Inputs:
        xgrid_true (np.array(n_grid)): grid of true solution
        u_true (np.array(n_grid)): true solution
        """
        fig, axs = plt.subplots(2,3, figsize=(15,10), dpi=150)
        # Plot sample solutions
        n_plotted_samples = 4
        for i in range(n_plotted_samples):
            axs[0,0].plot(xgrid, u[-i,:])
        axs[0,0].set_xlabel(r'x')
        axs[0,0].set_ylabel(r'$u_{n=0}(x,\omega)$')
        axs[0,0].set_title('sample solutions')

        # Plot mean and std dev of solution
        print('u', u.shape)
        if u_true is not None:
            axs[0,1].plot(xgrid_true, u_true, color='black', label=r'$u(x, k_{true})$')
        u_mean = u.mean(axis=0)
        u_std = u.std(axis=0)
        axs[0,1].plot(xgrid, u_mean[:], label=r'$\mathbf{E}_w[u(x,w)]$')
        axs[0,1].set_xlabel('x')
        #axs[0,1].set_ylabel(r'$u(x{=}'+str(x)+', \omega)$')
        axs[0,1].fill_between(xgrid,#range(u_mean.shape[0]),
            u_mean + u_std, u_mean - u_std, 
            alpha=0.3, color='blue',
            label=r'$\mathbf{E}_w[u(x,w)] \pm \sigma_n$')
        axs[0,1].set_title('mean and std dev of solution')
        axs[0,1].legend()


        axs[1,0].plot(ux[:])
        axs[1,0].set_xlabel('sample id, n')
        axs[1,0].set_ylabel(r'$u_n(x='+str(x)+', \omega)$')
        axs[1,0].set_title('sol at x over iterations')

        axs[1,1].plot(ux_cum_mean[:], label=r'$\bar u_n = \mathbf{E}_{n^* \in \{0,..., n\}}[u_{n^*}(x{=}'+str(x)+',\omega)]$')
        axs[1,1].fill_between(range(ux_cum_mean.shape[0]), 
            ux_cum_mean[:] + ux_cum_std[:], 
            ux_cum_mean[:] - ux_cum_std[:], 
            alpha=0.4, color='blue',
            label=r'$\bar u_n \pm \sigma_n$')
        axs[1,1].fill_between(range(ux_cum_mean.shape[0]), 
            ux_cum_mean[:] + ux_cum_sem[:], 
            ux_cum_mean[:] - ux_cum_sem[:], 
            alpha=0.4, color='black',
            label=r'$\bar u_n \pm$ std err$_n$')
        axs[1,1].fill_between(range(ux_cum_mean.shape[0]), 
            ux_conf_int[:,0], ux_conf_int[:,1], 
            alpha=0.4, color='orange',
            label=r'$95\% conf. interval; P $')
        axs[1,1].set_xlabel('sample id, n')
        axs[1,1].set_ylabel(r'$u(x{=}'+str(x)+', \omega)$')
        axs[1,1].set_ylim((ux_cum_mean.min()-np.nanmax(ux_cum_std), ux_cum_mean.max()+np.nanmax(ux_cum_std)))
        axs[1,1].set_title('sample mean over iterations')
        axs[1,1].legend()

        axs[1,2].plot(ux_cum_var[:], label=r'$\sigma^2_n$')
        axs[1,2].set_xlabel('sample id, n')
        axs[1,2].set_ylabel(r'$\sigma^2(u(x{=}'+str(x)+', \omega)$')
        axs[1,2].fill_between(range(ux_cum_var.shape[0]), 
            ux_cum_var[:] + ux_cum_sem[:], 
            ux_cum_var[:] - ux_cum_sem[:], 
            alpha=0.4, color='black',
            label=r'$\sigma^2_n \pm$ std err$_n$')
        axs[1,2].set_title('sample std dev. at x over iterations')
        axs[1,2].legend()

        #axs[1,1].plot(ux_conf_int[:])
        #axs[1,1].set_xlabel('sample id, n')
        #axs[1,1].set_ylabel(r'$V[E[u_{n\prime}(x='+str(x)+',w)]_0^n]$')
        #axs[1,1].set_title('95\% conf. int.')


        fig.tight_layout()
        plt.savefig('figures/proj1_3.png')

        # Plot prob. dist. at sample locations
        xs = np.linspace(0,1, 5)#[0., 0.25, 0.5, 0.75]
        ux_samples = np.empty((len(xs), u.shape[0]))
        for i, x_sample in enumerate(xs):
            x_id = (np.abs(xgrid - x_sample)).argmin()
            ux_samples[i,:] = u[:, x_id]

        self.ridge_plot_prob_dist_at_sample_x(xs, ux_samples)
        
def get_k_piecewise_constant(n_grid):
    n_y = 4 # number of different diffusivities
    assert n_grid%n_y == 0
    mu_y = -1.0
    var_y = 1.  
    Y = np.repeat(np.random.normal(mu_y, var_y, n_y), n_grid/n_y)
    k = np.exp(Y) # diffusivity
    return k

def power_kernel(xgrid, var_y=0.3, lengthscale=0.3, p=1.):
    '''
    Computes the power kernel (TODO: find out actual name)
    Input
    xgrid np.array(n_grid): Array of grid points
    var_y (int): variance of the output stochastic process
    lengthscale (int): lengthscale of the kernel
    p (int): order of the kernel
    Output:
    cov (np.array((n_grid,n_grid))): covariance matrix
    '''
    n_grid = xgrid.shape[0]
    x1 = np.repeat(xgrid[:,np.newaxis], n_grid, axis=1)
    x2 = np.repeat(xgrid[np.newaxis,:], n_grid, axis=0)
    cov = var_y * np.exp(-1./p * np.power(np.absolute(x1 - x2) / lengthscale, p))

    return cov

def kl_expansion(xgrid, mu_y, cov, trunc=3, z_candidate=None, plot=True):
    '''
    Applies karhunen loeve expansion to a stochastic process in 1D
    Assumes the stochastic process is a Gaussian Process (GP) with given mean and covariance
    Input:
    xgrid np.array(n_grid): 1D grid points
    mu_y np.array(n_grid): computed mean values of stochastic process on grid
    cov np.array(n_grid, n_grid): computed covariance function of stochastic process on grid
    trunc (int): number of non-truncated eigenvalues for Karhunen-Loeve expansion
    z_candidate (np.array(trunc)): if not None, these are candidates for the new random variables
    Output:
    Y np.array(n_grid): stochastic process 
    trunc_err float: truncation error    
    kl_fn (fn(np.array((n_grid,trunc))): Function to quickly query KL-expansion for new z  
    '''
    # Compute covariance eigenvalues and truncate
    eigvals, eigvecs = np.linalg.eig(cov) # eig s are returned sorted 
    eigvals_major = eigvals[:trunc]
    eigvecs_major = eigvecs[:, :trunc] #eigvecs[:trunc,:].T#
    
    # Function to quickly query KL-expansion for new z
    def sample_kl(kl_trunc=trunc):
        # Sample new random variable
        if z_candidate is None:
            # Assume Y is generated by Gaussian Process
            # Wrong: z = np.random.normal(0., 1., (n_grid, trunc))
            # Wrong: z = np.random.normal(0., 1.) * np.ones((n_grid,trunc)) # , trunc)[np.newaxis,:], repeats=n_grid, axis=0)
            z = np.repeat(np.random.normal(0., 1., trunc)[np.newaxis,:], repeats=n_grid, axis=0)
        else:
            z = np.repeat(z_candidate[np.newaxis,:], repeats=n_grid, axis=0)

        y_sample = mu_y + np.matmul(np.multiply(eigvecs[:,:kl_trunc], z), np.sqrt(eigvals[:trunc]))
        exp_y_sample = np.exp(y_sample)
        return y_sample, exp_y_sample
    Y, exp_Y = sample_kl()
    #kl_fn = lambda z_vec: mu_y + np.matmul(np.multiply(eigvecs_major, z_vec), np.sqrt(eigvals_major))

    # Compute new stochastic process at z
    #Y = mu_y + np.matmul(np.multiply(eigvecs_major, z), np.sqrt(eigvals_major))

    # Compute truncation error
    trunc_err = np.sum(eigvals[trunc:])


    if plot:
        # Plot eigenvalues
        fig = plt.figure(figsize=(9,5))
        plt.plot(range(eigvals.shape[0]), eigvals)
        plt.vlines(trunc,ymin=eigvals.min(),ymax=eigvals.max(), linestyles='--', label=r'truncation, $r=$'+str(trunc))
        plt.xlabel(r'mode id')
        plt.ylabel(r'eigenvalue of prior covariance')
        plt.legend()
        plt.savefig('figures/kl_exp_eigvals.png')
        plt.close()

        # Plot eigenvectors
        fig = plt.figure(figsize=(9,5))
        for t in range(trunc):
            plt.plot(xgrid, eigvecs[:,t], label=r'$\phi_'+str(t)+'$')
        plt.xlabel(r'location, $x$')
        plt.ylabel(r'eigenvector of prior covariance')
        plt.legend()
        plt.savefig('figures/kl_exp_eigvecs.png')
        plt.close()
        
        # Plot KL expansion
        fig = plt.figure(figsize=(9,5))
        plt.plot(xgrid, Y, color='gray', label='first sample')
        n_samples_plt = 1000
        Y_plt = np.zeros((n_samples_plt, n_grid))
        for n in range(n_samples_plt):
            #z_plt = np.repeat(np.random.normal(0., 1., trunc)[np.newaxis,:], repeats=n_grid, axis=0)
            #Y_plt[n,:] = kl_fn(z_plt) 
            Y_plt[n,:],_ = sample_kl()
        Y_plt_mean = Y_plt.mean(axis=0)
        Y_plt_var = Y_plt.var(axis=0)
        plt.plot(xgrid, Y_plt_mean,color='blue')
        plt.fill_between(xgrid, Y_plt_mean+Y_plt_var, Y_plt_mean-Y_plt_var,alpha=0.4,color='blue', label=r'$Y_{KL} \pm \sigma^2$')
        plt.xlabel(r'location, $x$')
        plt.ylabel(r'KL(log-permeability), $KL(Y)$')
        plt.legend()
        plt.savefig('figures/kl_exp_log_perm.png')
        plt.close()
        
    return Y, exp_Y, trunc_err, eigvals_major, eigvecs_major, sample_kl

def init_gaussian_process(xgrid, y_mean, y_var, lengthscale=0.3, order=1.):
    """
    Initializes parameters of a gaussian process
    xgrid (np.array(n_grid)): 1D grid points
    y_mean (float): constant mean
    y_var (float): constant
    Output:
    mu_y (np.array(n_grid)): mean over x
    cov_y (np.array((n_grid,n_grid))): covariance matrix
    """
    #import GPy
    #vis_kernel1 = GPy.kern.RBF(input_dim=1, variance=3., lengthscale=1)
    n_grid = xgrid.shape[0]
    mu_y = np.repeat(y_mean, n_grid)

    # Compute covariance matrix
    cov = power_kernel(xgrid, var_y=y_var, lengthscale=lengthscale, p=order)

    return mu_y, cov

def sample_low_dim_gaussian_process(xgrid, 
    mu_y, cov, expansion='KL', 
    trunc=3, z_candidate=None, 
    poly_deg=5, pce_coefs=None, 
    plot=True):
    '''
    Computates a function that samples from a low dimensional expansion of a gaussian process, given by mu_y and cov.
    Input:
    xgrid np.array(n_grid): 1D grid points
    mu_y (np.array(n_grid)): mean over x
    cov_y (np.array((n_grid,n_grid))): covariance matrix of gaussian process
    expansion (string): chosen expansion
    ## KL-expansion params:
    trunc (int): number of non-truncated eigenvalues for Karhunen-Loeve expansion
    z_candidate (np.array(trunc)): if not None, these are candidates for the new random variables
    ## PCE-expansion params:
    poly_deg (int): maximum polynomial degree of polynomial chaos expansion, typically equal to KL-exp.
    pce_coefs np.array(n_grid, poly_deg): If PCE coefficients are given, they will not be computed
    Output:
    Y np.array(n_grid): sample of low dim gaussian process
    exp_Y np.array(n_grid): sample of low dim exp-gaussian process
    trunc_err (float): truncation error 
    coefs np.array((n_grid, poly_deg)):  polynomial chaos expansion coefficients 
    '''
    n_grid = xgrid.shape[0]
    # Compute expansion
    Y, exp_Y, trunc_err_kl, kl_eigvals, kl_eigvecs, sampling_fn = kl_expansion(xgrid, mu_y, cov, trunc, z_candidate=z_candidate)
    if expansion == 'KL':
        trunc_err = trunc_err_kl
    elif expansion == 'polynomial_chaos':
        Y, exp_Y, trunc_err, coefs, sampling_fn = polynomial_chaos_expansion(xgrid, 
            kl_trunc=trunc, kl_mu_y=mu_y, kl_eigvals=kl_eigvals, kl_eigvecs=kl_eigvecs,
            poly_deg=trunc, plot=plot, c_alphas=pce_coefs)
    return Y, exp_Y, trunc_err, coefs, sampling_fn

def gauss_hermite(x, deg, verbose=False):
    """
    Computes the Gaussian-Hermite polynomial of degree deg for all points in x 
    Source: https://numpy.org/doc/stable/reference/routines.polynomials.classes.html
    x np.array(n_grid): input array
    """
    domain = [x.min(), x.max()]
    # Get basis coefficients of "probabilists" hermite polynomial
    herm = np.polynomial.hermite_e.HermiteE.basis(deg, 
        domain=domain, window=domain)
    coefs = herm.coef

    y = herm(x)

    return y


### 
# Polynomial Chaos Expansion
###
import numpy.polynomial.hermite_e as H
from scipy.stats import norm
import numpy.matlib

def multi_indices(ndim=2, poly_deg=3, order='total_degree'):
    """
    Creates lattice of multi indices
    Input:
    ndim int: input dimension, e.g., number of parameters
    poly_deg int: polynomial degree
    order string: total_order (l_1 norm(multi_indices)<poly_deg), full_order
    Output:
    multi_indices np.array(ndim,poly_deg)
    For example:
        ndim=3, poly_deg=2, order='total_degree' 
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
    for k in range(poly_deg-1):
        kronecker = np.kron(np.eye(ndim), np.ones((Mk.shape[0], 1)))
        Mk = numpy.matlib.repmat(Mk,ndim,1) + kronecker
        Mk = np.unique(Mk, axis=0)
        M = np.vstack((M, Mk))# M = [M; Mk];
    multi_indices = M.astype(int) # dim: ( n_multi_indices ), ndim
    
    return multi_indices

    """
    TODO: use this function instead
    function M = totalorderindices(n,p)
    Mk = zeros(1,n);
    M = Mk;
    for k = 1:p
        Mk = repmat(Mk,n,1)+kron(eye(n),ones(size(Mk,1),1));
        Mk = unique(Mk,'rows');
        M = [M; Mk];
    end
    end
    """

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

def trapezoid_int(f, a, b, n=100):
    """
    Calculate integral via trapezoidal rule
    """
    P = [a + i * (b-a) / n for i in range(0, n+1)] # Grid points
    F = [1. / 2. * np.abs(P[i+1] - P[i]) * (f(P[i+1]) + f(P[i])) for i in range(0, n+1)] # Function evals at grid points
    return sum(F)

def one_d_gauss_herm_pce_coef(alpha):
    mu = 0.
    var = 1.
    return np.exp(mu + var/2.) * np.power(var, alpha) / np.math.factorial(alpha)

def one_d_gauss_herm_pce_nominator(h, alpha):
    def gauss_herm(y):
        inv_cdf_gaussian = norm.ppf(y, loc=0, scale=1)
        H.hermeval(inv_cdf_gaussian, Herm(alpha))
    integrand = lambda y: h(y) * gauss_herm(y)
    nominator = trapezoid_int(integrand, 0.001, 1-0.001, 1000)
    return nominator * np.sqrt(2*np.pi)

def get_pce_coefs(poly_deg, xgrid=None, 
    Y_hat=None, kl_trunc=10, kl_mu_y=None, kl_eigvals=None, kl_eigvecs=None,
    Y_inv_cdf=None, plot=True, mc_integration=False):
    """
    WRONG! Computes the full tensor PCE coefficients with Gaussian-Hermite polynomials 
    Adapted from: Emily Gorcenski - Polynomial Chaos: A technique for modeling uncertainty; url=https://youtu.be/Z-Qio-n6yPc; 
    Input:
    poly_deg (int): Target polynomial degree truncation
    xgrid (np.array((n_grid))): Grid points of Y_hat in spatial domain 
    Y_hat (fn(np.array(n_grid, kl_trunc))): Function that approximates the target function at all x, as function of Gaussian noise, e.g., the KL-expansion
    Y_inv_cdf (fn): Inverse CDF of target distribution. This is an alternative input to Y_hat, that only works with constant Y.
    mc_integration (bool): Use Monte Carlo Integration (don't use!)
    Output:
    c_alpha (np.array(n_grid, poly_deg)): PCE coefficients
    """
    test = False # Use this to make code look like Emily's example
    test2 = True # Use this to make code look like plots reported in proj2

    # Initialize output coefficients
    n_grid = xgrid.shape[0]
    c_alphas = np.zeros((n_grid, poly_deg))

    # Set up Gauss-Hermite quadrature for integration in denominatory
    n_quad_pts = poly_deg**2#poly_deg**2 # number of sample points. Shouldn't this be \sqrt(poly_deg-1)?
    xi, w = H.hermegauss(n_quad_pts) # Get points and weights of Gauss-Hermite quadrature.

    if kl_eigvals is not None:
        pass
    else:
        for alpha in range(0,poly_deg): # Change this to do sth else than full-tensor truncation
            # Compute the nominator/integral
            if mc_integration:
                n_mc = 1000
                np.random.sample             
                print('poly deg', poly_deg)
                print('kl trunc', kl_trunc)
                H_alpha_mc = np.zeros((n_mc))
                Y_alpha_mc = np.zeros((n_mc, n_grid))
                Y_alpha_mc_int = np.zeros(n_grid)
                for n in range(n_mc):
                    xi_mc = np.random.normal(0,1)
                    H_alpha_mc[n] = H.hermeval(xi_mc, Herm(alpha))
                    z_kl = np.repeat(np.random.normal(0,1, kl_trunc)[np.newaxis,:],repeats=n_grid,axis=0)
                    Y_alpha_mc[n,:] = Y_hat(z_kl) * H_alpha_mc[n]
                    Y_alpha_mc_int[:] += Y_alpha_mc[n,:]
                Y_alpha_mc_int = 1./float(n_mc) * Y_alpha_mc_int
                nominator = Y_alpha_mc_int
                print('nominator', alpha, nominator)
                import pdb;pdb.set_trace
            elif Y_hat is not None:
                # Integrate target fn with gauss-hermite quadrature rule
                # TODO: Feed in distribution of Y_hat as it's stochastic in the case of the KL expansion.
                # TODO: check if i need scaling term np.sqrt(2*np.pi)
                # TODO: should Y_hat be independent of xi
                herm_sum = np.zeros((n_grid)) # []
                for idx in range(n_quad_pts):
                    #z_kl = xi[idx] * np.ones((n_grid, kl_trunc))
                    #! Gaussian quadrature integration with KL-expansion doesn't make sense, bc KL has more than 
                    #   one stochastic dimension!
                    z_kl = np.repeat(np.random.normal(0,1, kl_trunc)[np.newaxis,:],repeats=n_grid,axis=0)
                    #z_kl = np.repeat(np.random.normal(xi[idx],1, kl_trunc)[np.newaxis,:],repeats=n_grid,axis=0)
                    #print('shape ykl, h, w', Y_hat(z_kl).shape, H.hermeval(xi[idx], Herm(alpha)).shape, w[idx].shape)
                    #print('shape ykl, h, w', Y_hat(z_kl).shape, H.hermeval(xi[idx], Herm(alpha)), w[idx])
                    #import pdb;pdb.set_trace()
                    herm_sum += Y_hat(z_kl) * H.hermeval(xi[idx], Herm(alpha)) * w[idx]
                #nominator = Y_hat * herm_sum 
                nominator = herm_sum
                #print('nominator', alpha, nominator)
                #nominator = Y_hat[z_init = xi] * Herm(xi) * w
                #nominator = Y_hat * sum([H.hermeval(xi[idx], Herm(alpha)) * w[idx] for idx in range(n_quad_pts)])
            elif Y_inv_cdf is not None:
                nominator = one_d_gauss_herm_pce_nominator(Y_inv_cdf)
            else:
                return NotImplementedError()
            
            # Compute denominator with Gauss-Hermite quadrature rule for integration
            if test2: 
                herm_norm = 0
                for idx in range(n_quad_pts):
                    herm_norm += herm_inner_product_at_x(Herm(alpha), Herm(alpha))(xi[idx]) * w[idx]
                denominator = herm_norm
            else:
                # TODO: this is only valid for 1-dimensional hermite polynomials, i.e., if we're estimating
                #  1 random variable. Derive formula for the multivariate case.
                denominator = np.math.factorial(alpha)
            
            c_alphas[:,alpha] = nominator / denominator

    if plot:
        # Plot hermite basis polynomials
        fig = plt.figure(figsize=(9,5))
        for alpha in range(poly_deg): # Change this to do sth else than full-tensor truncation
            h_x = np.zeros((n_quad_pts))
            for idx in range(n_quad_pts):
                h_x[idx] = H.hermeval(xi[idx], Herm(alpha)) #* w[idx]
            plt.plot(xi, h_x, label=r'$H_{\alpha='+str(alpha)+'}$')
        plt.xlabel(r'location, $x$')
        plt.ylabel(r'polynomial, $\psi_\alpha(x)$')
        plt.xlim((-3,6))
        plt.ylim((-10,20))
        plt.tight_layout()
        plt.legend()
        plt.savefig('figures/hermite_poly_Herm.png')

    return c_alphas

def get_pce_coef(alpha_vec, mu_y, eigvals, eigvecs, verbose=True):
    """
    Calculates the PCE coefficient for a specific combination of polynomial degrees, alpha
    Input:
    alpha_vec np.array(ndim,dtype=int): realization of one multi-index, indicating the degree of each polynomial, herm_alpha_i;
        ndim is the number of stochastic dimensions, i.e., number of random variables. 
        Elements of alpha_vec, alpha_i, can bigger than ndim
    mu_y np.array(n_grid): mean of target function
    eigvals np.array(ndim): eigenvalues of covariance of target function
    eigvecs np.array((n_grid, ndim)): eigenvectors of covariance of target function
    Ouput:
    c_alpha np.array(n_grid): PCE coefficient of alpha_vec
    """   
    if verbose: print('alpha_vec', alpha_vec, alpha_vec.shape)

    ndim = alpha_vec.shape[0]
    n_grid = mu_y.shape[0]
    # Set up Gauss-Hermite quadrature for integration in nominator
    n_quad_pts = ndim**2#poly_deg**2 # number of sample points. Shouldn't this be \sqrt(poly_deg-1)?
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

def polynomial_chaos_expansion(xgrids, 
    kl_trunc=0, kl_mu_y=None, kl_eigvals=None, kl_eigvecs=None,
    poly_f=gauss_hermite, poly_deg=5, c_alphas=None,
    plot=False,verbose=False):
    """
    Computes the polynomial chaos expansion (PCE) of an n-dimensional process, assumed to be Gaussian
    Input:
    xgrids: np.array(n_dim, n_grid): Gaussian samples of stochastic dimensions, n_dim, with equidistant grid spacing, n_grid
    poly_f np.array(p): polynomial degree
    poly_deg function(): polynomial basis function, e.g., Gauss Hermite
    c_alphas (np.array(n_grid, poly_deg)): If c_alphas are supplied, they will not be computed.
    plot (boolean): if true, plots various quantities of interest
    Output:
    exp_Y np.array(n1_grid,n2_grid): exponential of approximated stochastic process, exp(Y)
    c_alphas (np.array(n_grid, poly_deg)): PCE coefficients
    """
    log = {}
    if len(xgrids.shape)==1:
        xgrids = xgrids[np.newaxis, :]
    n_grid = xgrids.shape[1]

    # Calculate Multi Indices
    ndim = kl_trunc # poly_deg
    poly_deg = kl_trunc
    alpha_indices = multi_indices(ndim=ndim, poly_deg=poly_deg, order='total_degree')
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
        exp_y_pce = np.zeros(n_grid)
        xi = np.random.normal(0,1,ndim) # one random variable per stochastic dimension
        for a, alpha_vec in enumerate(alpha_indices):
            herm_alpha = np.zeros(ndim)
            for idx, alpha_i in enumerate(alpha_vec):
                herm_alpha[idx] = H.hermeval(xi[idx], Herm(alpha_i)) # dim: ndim
            exp_y_pce += c_alphas[:,a] * np.prod(herm_alpha) # dim: n_grid
        y_pce = np.log(np.where(exp_y_pce>0, exp_y_pce, 1.))

        return y_pce, exp_y_pce, trunc_err, c_alphas
    y_pce, exp_y_pce, trunc_err, c_alphas = sample_pce()


    if plot:
        # Plot basis polynomials
        fig = plt.figure(figsize=(9,5))
        x_poly = np.linspace(-4,4)
        for alpha in range(poly_deg):
            herm_alpha = H.hermeval(x_poly, Herm(alpha))
            plt.plot(x_poly, herm_alpha, label=r'$\alpha_i=$'+str(alpha))
        plt.xlabel(r'location, $x$')
        plt.ylabel(r'Probabilists Hermite polynomial, $He_{\alpha_i}(x)$')
        plt.ylim((-10,20))
        plt.tight_layout()
        plt.legend()
        plt.savefig('figures/hermite_poly.png')
        plt.close()

        # Plot the stochastic process
        fig = plt.figure(figsize=(9,5))
        #for a, y_alpha in enumerate(y_alphas):
        #    plt.plot(xgrids[dim,:], y_alpha, label=r'$y(\alpha\leq$' + str(a) + ')')
        plt.plot(xgrids[0,:], exp_y_pce, label=r'$y$')
        plt.xlabel(r'location $x$')
        plt.ylabel(r'log-permeability $x$')
        plt.tight_layout()
        plt.legend()
        plt.savefig('figures/polynomial_chaos.png')
        plt.close()

        # Plot mean and variance of PCE-estimated stochastic process 
        fig = plt.figure(figsize=(9,5))
        n_samples_plt = 100
        print('poly deg', poly_deg)
        H_alpha_plt = np.zeros((n_samples_plt, poly_deg))
        Y_alpha_plt = np.zeros((n_samples_plt, n_grid))
        exp_Y_plt = np.zeros((n_samples_plt, n_grid))
        for n in range(100):
            #xi = np.random.normal(0,1,ndim) # one random variable per stochastic dimension
            #for a, alpha_vec in enumerate(alpha_indices):
            #    herm_alpha = np.zeros(ndim)
            #    for idx, alpha_i in enumerate(alpha_vec):
            #        herm_alpha[idx] = H.hermeval(xi[idx], Herm(alpha_i))
            #    exp_Y_plt[n,:] += c_alphas[:,a] * np.prod(herm_alpha)
            _, exp_Y_plt[n,:], _, _ = sample_pce()

        exp_Y_plt_mean = exp_Y_plt.mean(axis=0)
        exp_Y_plt_std = exp_Y_plt.std(axis=0)
        plt.plot(xgrid, exp_Y_plt_mean)
        plt.fill_between(xgrid, exp_Y_plt_mean+exp_Y_plt_std, exp_Y_plt_mean-exp_Y_plt_std,alpha=0.4,color='blue', label=r'$Y_{PCE} \pm \sigma$')
        plt.xlabel(r'location, $x$')
        plt.ylabel(r'PCE of permeability, $PCE(\exp(Y))$')
        plt.savefig('figures/pce_exp_of_y.png')
        plt.close()

        # Plot PCE coefficients
        fig = plt.figure(figsize=(9,5))
        for a, alpha_vec in enumerate(alpha_indices):
            plt.plot(xgrids[0,:], c_alphas[:,a], label=r'$C_{\vec \alpha=' + str(alpha_vec) + '}$)')
        plt.xlabel(r'location $x$')
        plt.ylabel(r'PCE coefficient, $C_{\vec\alpha}(x)$')
        plt.tight_layout()
        plt.legend()
        plt.savefig('figures/pce_coefs.png')
        plt.close()

    return y_pce, exp_y_pce, trunc_err, c_alphas, sample_pce

# https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
def plot_trunc_err(truncs, trunc_err):
    # Input
    # truncs np.array(n_truncations)
    # trunc_err np.array(n_truncations)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(truncs, trunc_err)
    axs.set_xlabel(r'truncation, $r$')
    axs.set_ylabel(r'truncation err.')
    fig.savefig('figures/ai_trunc_err.png')

# PSET 3: inverse problem
def get_msmts(plot=False):
    x_obs = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u_obs = np.array([1.33324463455171, 1.36464171067521, 1.40744323365932, 1.40555567476256, 1.38851427638760, 1.39799450797509, 1.31587775071708, 1.23031611117801, 1.15017066712980, 1.06664471477387])
    k_true = np.array([4.60851932653568, 4.86913697560332, 4.62032247978475, 5.00739558303546, 4.82792526785225, 5.07901564393012, 4.66138567908013, 4.17793583225948, 4.12539240678111, 4.44239649473624, 3.92333780870393, 4.54533170853051, 4.60912071107044, 4.52595709855134, 4.07893866778093, 3.82307832614272, 3.32917917553473, 3.01300181823486, 2.84411210569811, 2.87006579934071, 2.89882059530642, 3.00884548070947, 2.91266269929508, 2.81452942762117, 2.49228002628364, 2.71739595852760, 3.03644913148034, 2.57126956371258, 3.34661706710215, 2.94035041736722, 2.76450418823743, 2.62201506635126, 2.67758311125735, 3.12653883764129, 2.82006727239076, 2.92333827548000, 2.96095777130370, 3.02618524256612, 2.86938905850092, 2.70856278568016, 2.78734680029447, 3.10554300673895, 2.69887845768822, 2.65547556555735, 2.54448709529566, 2.28154078928308, 2.05414576956807, 2.15512302854182, 2.02513576436545, 1.87145499501584, 1.62145222971342, 1.43214562583999, 1.54012048601883, 1.57796945154306, 1.60947965439388, 1.47991657877595, 1.43280284347803, 1.42363837350593, 1.27337927667739, 1.24573074923507, 1.17312198311588, 1.31585192605426, 1.44015136034874, 1.66857903299136, 1.50962174415947, 1.54635945079439, 1.57589884658520, 1.70268496644981, 1.48816588773939, 1.43441178934950, 1.30716586222675, 1.30448229931249, 1.35949239558635, 1.45074973209516, 1.31924083608101, 1.42582943243071, 1.44731165671863, 1.31590290213155, 1.54561861639610, 1.14077717049381, 1.16094734995639, 1.23679487215449, 1.29243734753453, 1.09482574667417, 1.01743297741724, 1.28626862411626, 1.22763470870996, 1.24686001335088, 1.16866417198729, 1.11627212135544, 1.15750020242220, 1.21062567892525, 1.09896520713583, 1.18421265203534, 1.18059404644927, 1.17768529209078, 1.22455198272902, 1.04759883491168, 1.11647043353682, 1.33203747787611, 1.81313704286957, 1.60459248681251, 1.83411925167631, 1.86578565385545, 1.85751416668662, 1.74414424967352, 1.73943691889385, 1.65779696531455, 1.83539329972025, 1.66501410132260, 1.77233391597251, 2.12184254631078, 2.08316076727839, 2.56829457862793, 2.89616947379460, 2.76305598716123, 2.83952732577347, 2.84136601757010, 3.30459311061616, 3.71748320354009, 3.39326458870804, 3.72997251289344, 3.99814053512306, 3.56505218268485, 2.97146782046398, 2.53762825756458, 2.60267064123793, 2.80376324967721, 2.88830670266253, 2.63016047253275, 2.96651061385475, 2.84645722915763, 2.31591604193898, 2.33897006308885, 2.54556736907830, 2.49416021480434, 2.58507404041490, 2.92341952128821, 3.03701111325719, 3.13978838256269, 2.68511723989237, 2.82033378099678, 3.00537274866223, 3.09752033239746, 3.37058892306905, 2.81517596314294, 2.62448795757048, 2.27282371468636, 2.12617721213605, 2.00544200770971, 1.95385075829803])
    xgrid = np.linspace(0.,1.,k_true.shape[0])
    Y_true = np.log(k_true)
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=300)
        axs[0].plot(x_obs, u_obs)
        axs[0].set_xlabel(r'location, $x$')
        axs[0].set_ylabel(r'measurements, $u$')
        axs[1].plot(xgrid, k_true)
        axs[1].set_xlabel(r'location, $x$')
        axs[1].set_ylabel(r'permeability, $k$')
        axs[2].plot(xgrid, Y_true)
        axs[2].set_xlabel(r'location, $x$')
        axs[2].set_ylabel(r'log-permeability, $Y$')
        fig.tight_layout()
        fig.savefig('figures/ai_msmts.png')

    return x_obs, u_obs, k_true, xgrid

def plot_accept_rates(accept_rates):
    """
    Create plot of acceptance rates
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    n_samples = accept_rates.shape[0]
    axs.plot(range(n_samples), accept_rates)
    axs.set_xlabel(r'sample_id, $t$')
    axs.set_ylabel(r'accept rates')
        
    fig.tight_layout()
    fig.savefig('figures/mh_accept_rates.png')

def plot_k(xgrid, k, k_true, y_gp_mean, y_gp_cov):
    """
    Plot k samples vs. ground-truth
    Input:
    xgrid: np.array(n_grid): Grid with equidistant grid spacing, n_grid
    k: np.array(n_samples, n_grid): Samples of permeability, k
    k_true: np.array(n_grid): Ground truth of permeability, k
    """
    kmean = k.mean(axis=0)
    kvar = k.var(axis=0)
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    axs[0].plot(xgrid, k_true, color='black', label=r'true $k$')
    axs[0].plot(xgrid, kmean, color='blue')#, label=r'post. $\bar k_n$')
    axs[0].fill_between(xgrid, kmean+kvar, kmean-kvar,alpha=0.4,color='blue', label=r'post. $\bar k_n \pm \sigma_k^2$')
    axs[0].set_ylim((k_true.min()-5.,k_true.max()+5.))
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'permeability, $k$')
    axs[0].legend()
    
    # log permeability
    Y = np.log(k)
    Y_true = np.log(k_true)
    Ymean = Y.mean(axis=0)
    Yvar = Y.var(axis=0)
    # True
    axs[1].plot(xgrid, Y_true, color='black', label=r'true $Y$')
    # Prior
    axs[1].plot(xgrid, y_gp_mean, color='green')#, label=r'prior $\bar Y_n$')
    axs[1].fill_between(xgrid, y_gp_mean+np.diag(y_gp_cov), y_gp_mean-np.diag(y_gp_cov),alpha=0.4,color='green', label=r'prior $\bar Y_n \pm \bar\sigma_Y^2$')
    # Posterior
    axs[1].plot(xgrid, Ymean, color='blue')#, label=r'post. $\bar Y_n$')
    axs[1].fill_between(xgrid, Ymean+Yvar, Ymean-Yvar,alpha=0.4,color='blue', label=r'post. $\bar Y_n \pm \bar\sigma_Y^2$')
    axs[1].set_xlabel(r'location, $x$')
    axs[1].set_ylabel(r'log-permeability, $Y$')
    axs[1].legend()
        
    fig.tight_layout()
    fig.savefig('figures/k_gp_vs_msmts.png')

def injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05, plot=True):
    """
    Returns a function over xgrid that models injection wells
    Input: 
    xgrid (np.array(n_grid)): grid
    n_wells (int): number of equidistantly placed wells
    strength (float): source strength, theta
    width (float): source width, delta
    Output:
    source (np.array(n_grid)): function that models injection wells
    """
    n_grid=xgrid.shape[0]

    pos_wells = np.linspace(xgrid.min(), xgrid.max(), num=n_wells+2) # equidistant wells
    pos_wells = pos_wells[1:-1] # remove wells at borders
    pos_wells = np.repeat(pos_wells[np.newaxis,:], repeats=n_grid,axis=0) # repeat for matrix math
    xgrid_w = np.repeat(xgrid[:,np.newaxis], repeats=n_wells,axis=1) # repeat for matrix math
    
    amp = strength / (width * np.sqrt(2.*np.pi)) # amplitude
    source = np.sum(amp * np.exp(-(xgrid_w - pos_wells)**2 / (2.*width**2)),axis=1) # sum over wells

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
        axs.plot(xgrid, source)
        axs.set_xlabel(r'location, $x$')
        axs.set_ylabel(r'source, $s$')
        fig.tight_layout()
        fig.savefig('figures/source_injection_wells.png')

    return source

def gaussian_proposal(z, std=1.):
    """
    Gaussian proposal distribution.

    Draws new parameters from (multi-)variate Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    Inputs:
    z (np.array(n_params)): input parameter / current solution
    sigma (float): Standard deviation of Gaussian distribution. Can be 
        scalar or vector of length(z)
    
    Outputs: 
    z_candidate (np.array(n_params)): candidate solution to parameter
    trans_prob_ratio (float): ratio of transition probabilities 
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
    Input: 
    y np.array(n_grid): grid points
    mean np.array(n_grid): mean of gaussian at each grid point
    std np.array(n_grid): std deviation at each grid point
    Output:
    g np.array(n_grid): likelihood of y
    """
    g = np.log(1.) - np.log(std * np.sqrt(2*np.pi)) + (-0.5 * np.power((y-mean),2) / np.power(std,2))
    return g


def gaussian_lnpost(u, u_obs, obs_noise=1e-4):
    """
    Computes likelihood of, u, given measurements, u_obs, assuming Gaussian msmt noise
    u (np.array(n_grid)): solution, given candidate parameter
    u_obs (np.array(n_grid)): measurements of solution 
    obs_noise (float): variance of gaussian measurement noise
    Outputs:
    lnpost (np.array(n_grid))    
    """
    n_grid = u.shape[0]
    std = np.sqrt(obs_noise * np.ones(n_grid))
    lnpost_i = ln_gaussian(y=u, mean=u_obs, std=std)
    lnpost = np.sum(lnpost_i) 
    return lnpost
    
def get_sol_w_gp_prior(xgrid, 
        y_gp_mean, y_gp_cov, 
        trunc, expansion, z_candidate, 
        poly_deg, 
        flux, source, rightbc, x_obs=None,
        sample_y=None,
        pce_coefs=None):
    """
    Computes a solution of stochastic elliptic equation given all parameters
    x_obs (np.array(n_msmts)): if not None, only values at xobs are returned
    """
    vals = {}

    if sample_y is None:
        vals['Y'], vals['exp_Y'], vals['trunc_err'], vals['coefs'], sample_y = sample_low_dim_gaussian_process(xgrid, 
            mu_y=y_gp_mean, cov=y_gp_cov,
            trunc=trunc, expansion=expansion, z_candidate=z_candidate, 
            poly_deg=poly_deg, pce_coefs=pce_coefs)
    else:
        vals['Y'], vals['exp_Y'], vals['trunc_err'], vals['coefs'] = sample_y()
    # Compute permeability
    vals['k'] = vals['exp_Y'] # np.exp(vals['Y']) 
    # Compute solution
    vals['u'] = stochDiffEq.diffusioneqn(xgrid, F=flux, k=vals['k'], source=source, rightbc=rightbc)[:,0]

    # Reduce solution to observed values
    if x_obs is None:
        u_obs = vals['u']
    else:
        obs_idxs = np.zeros((x_obs.shape[0]))
        for i, x_o in enumerate(x_obs):
            obs_idxs[i] = np.abs(xgrid - x_o).argmin().astype(int)
        obs_idxs = obs_idxs.astype(int)
        u_obs = vals['u'][obs_idxs]

    return u_obs, vals, sample_y


def plot_mh_sampler(chain, lnprobs, disable_ticks=False):
    # Plot metropolis hastings chain and log posterios
    sample_ids = np.arange(chain.shape[0], dtype=int)
    ndim = chain.shape[1]

    # Plot mh chain
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        ax.plot(sample_ids, chain[:,i], label=f'{i}')
        if col_id == 0: 
            ax.set_ylabel(r'parameter, $z$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'sample id, $t$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('figures/mh_chain.png')

    # Plot mh lnprobs
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(sample_ids, lnprobs[:])
    axs.set_ylabel(r'log-probability, $p(u^{(t)}|X)$')
    axs.set_xlabel(r'sample id, $t$')
    axs.set_ylim((-5.,lnprobs.max()+1.))
    fig.tight_layout()
    fig.savefig('figures/mh_lnprobs.png')

    # Plot autocorrelation of parameters as fn of lag (only useful after warmup)
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        autocorr_i = np.correlate(chain[:,i], chain[:,i], mode='full')
        autocorr_i = autocorr_i[int(autocorr_i.size/2.):] # take 0 to +inf
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        ax.plot(sample_ids, autocorr_i, label=f'{i}')
        if col_id == 0: 
            ax.set_ylabel(r'autocorrelation')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'sample id, $t$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('figures/mh_autocorr.png')

    # Plot posterior distribution of parameters (only useful after warmup)
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        hist_i = ax.hist(chain[:,i], density=True, color="blue", alpha=0.6, label=f'{i}')#, bins=50)#bins='auto',#, lw=1) bins='auto', density=True, 
        if col_id == 0: 
            ax.set_ylabel(r'posterior $p(z|X)$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'$z$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('figures/mh_param_post.png')

def mh_sampler(z_init,
    proposal_fn=gaussian_proposal, proposal_fn_args={'std':1.},
    model=get_sol_w_gp_prior, model_args={},
    lnpost_fn=gaussian_lnpost, lnpost_fn_args={'u_obs':None, 'obs_noise':1e-4}, 
    n_samples=1e4, warmup_steps=0, plot=True):
    """
    MCMC sampler: Metropolis-Hastings
    Sources: https://jellis18.github.io/post/2018-01-02-mcmc-part1/ 

    Inputs:
    z_init (np.array(ndim)): start point for sampling with stochastic dimension, ndim
    proposal_fn (fn): computes proposal 
    proposal_fn_args (dict()): arguments to proposal function 
    model (fn): runs model
    model_args (dict()): arguments to model 
    lnpost_fn (fn): computes log-posterior
    lnpost_fn_args (dict()): computes log-posterior
    n_samples (int): number of samples
    warmup_steps (int): number of warmup steps
    Outputs:
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
        plot_mh_sampler(chain, lnprobs)

    return chain, lnprobs, accept_rates, logs

def plot_ensemble_kf(z_post_samples,disable_ticks=False):
    ndim = z_post_samples.shape[-1]
    # Plot posterior distribution of parameters 
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        hist_i = ax.hist(z_post_samples[:,i], density=True, color="blue", alpha=0.6, label=f'{i}')#, bins=50)#bins='auto',#, lw=1) bins='auto', density=True, 
        if col_id == 0: 
            ax.set_ylabel(r'posterior $p(z|X)$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'$z$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('figures/enkf_param_post.png')

def ensemble_kalman_filter(z_init,
    proposal_fn=gaussian_proposal, proposal_fn_args={'std':1.},
    model=get_sol_w_gp_prior, model_args={},
    u_obs=None, obs_noise=1e-4,
    n_samples=1e4, n_batch=1, plot=True):
    """
    Ensemble Kalman Filter
    Sources: https://piazza.com/class/kecbx4m6z2f46j?cid=105

    Inputs:
    z_init (np.array(ndim)): start point for sampling with stochastic dimension, ndim
    proposal_fn (fn): computes proposal 
    proposal_fn_args (dict()): arguments to proposal function 
    model (fn): runs model
    model_args (dict()): arguments to model 
    u_obs (np.array(n_obs, n_grid_obs)): observations of the model solution
    obs_noise (float): measurement noise
    n_samples (int): number of samples
    Outputs:
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
    # Del:
    # cov_z = cov[:ndim,:ndim]
    # cov_yz_t = cov[ndim:,:ndim]
    # cov_y = cov[ndim:, ndim:]
    
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
        plot_ensemble_kf(z_post_samples)

    return z_post_samples, logs

# Integrate tqdm
def to_iterator(obj_id):
    # Call this to display tqdm progressbar when using ray parallel processing
    # Source https://github.com/ray-project/ray/issues/5554
    while obj_id:
        done, obj_id = ray.wait(obj_id)
        yield ray.get(done[0])

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
    trunc_errs = np.empty((n_samples_after_warmup,1))
    stoch_dim = logs[0]['coefs'].shape[-1]
    coefs = np.empty((n_samples_after_warmup, n_grid, stoch_dim))

    # Copy logs into params
    for n, log in enumerate(logs):
        k[n,:] = log['k']
        Y[n,:] = log['Y']
        u[n,:] = log['u']
        trunc_errs[n,0] = log['trunc_err']
        coefs[n,:,:] = log['coefs']
    return k, Y, u, trunc_errs, coefs

def init_preprocessing(fn, parallel=False):
    """
    Init parallel processing
    fn (fn): function that's to be parallelized
    """
    if parallel:
        import ray
        from tqdm import tqdm
        import psutil
        # https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8
        num_cpus = psutil.cpu_count(logical=False)
        print('n cpus', num_cpus)
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)            
    if parallel:
        fn_r = ray.remote(fn).remote
    else:
        fn_r = fn
    fn_tasks = []
    return fn, fn_tasks

def get_parallel_fn(model_tasks):
    """
    Waits for parallel model tasks to finish and returns outputs
    Outputs:
    model_outputs (list(tuple)): outputs of model
    """
    for x in tqdm(to_iterator(model_tasks), total=len(model_tasks)):
        pass
    model_outputs = ray.get(model_tasks) # [0, 1, 2, 3]
    return model_outputs
def plot_k_vs_ks_nn(k, k_samples_nn):
    plt.figure(figsize=(15,8))
    n_samples = k_samples_nn.shape[0]
    k_samples_mean = k_samples_nn.mean(axis=0)
    k_samples_std = k_samples_nn.std(axis=0)
    plt.plot(xgrid, k_samples_mean, color='blue')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_samples_mean + k_samples_std, k_samples_mean - k_samples_std, 
        alpha=0.3, color='blue',
        label=r'PCE target: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    k_mean = k.mean(axis=0)
    k_std = k.std(axis=0)
    plt.plot(xgrid, k_mean, color='green')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_mean + k_std, k_mean - k_std, 
        alpha=0.3, color='green',
        label=r'Learned: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('../final_project/figures/nn_k_samples.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='project 1-2')
    parser.add_argument('--pset2', action = "store_true",
            help='diffusion eqn')
    parser.add_argument('--pce', action = "store_true",
            help='polynomial chaos expansion (pset 2aii)')
    parser.add_argument('--poly_deg', default=4, type=int,
            help='Maximum degree of polynomials in PCE or eigenvectors in KL')
    parser.add_argument('--parallel', action = "store_true",
            help='enable parallel processing')
    parser.add_argument('--inverse_problem', action="store_true",
            help='inverse_problem')
    parser.add_argument('--mcmc', action="store_true",
            help='solve inverse problem with MCMC metropolis-hastings')
    parser.add_argument('--ensemble_kf', action="store_true",
            help='solve inverse problem with ensemble kalman filter')
    parser.add_argument('--eval_trunc_err', action = "store_true",
            help='evaluation truncation error for pset2.ai')
    parser.add_argument('--n_samples', default=1000, type=int,
            help='number of samples')
    parser.add_argument('--warmup_steps', default=1000, type=int,
            help='number of warmup steps for MCMC')
    parser.add_argument('--sim_datapath', default=None, type=str,
            help='path to logged simulation data, e.g., pce.pickle')
    parser.add_argument('--nn_pce', action = "store_true",
            help='use a neural net to approximate the polynomial chaos coefficients')
    parser.add_argument('--est_param_nn', default='pce_coefs', type=str,
            help='name of parameter than shall be estimated by neural net, e.g., "pce_coefs", "k"')

    args = parser.parse_args()
    # Set random seeds
    np.random.seed(0)
    if args.nn_pce:
        import torch
        torch.manual_seed(0)


    # Pset 3: Get xgrid from ground-truth msmts
    x_obs, u_obs, k_true, xgrid = get_msmts(plot=True)
    n_grid = xgrid.shape[0]
    n_obs = x_obs.shape[0]
    #MCMC: https://jellis18.github.io/post/2018-01-02-mcmc-part1/

    # Pset 2: Define xgrid
    if args.pset2:
        #n_grid = 100 # number of grid points
        #xgrid = np.linspace(0, 1, n_grid) # vector with grid points
        #x_obs = xgrid
        #n_obs = n_grid
        pass

    if args.inverse_problem:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = 0
    # Initialize forward and inverse solution
    n_samples = args.n_samples

    ## Set up SDE initial conditions and parameters
    if args.pset2:
        mu_f = -1. #-2.
        var_f = .2 #.5
        flux = np.random.normal(mu_f, var_f, 1) # np.ones(n_grid) * 
        rightbc = 1.
        source = 5. # np.ones((n_grid)) 
    elif args.inverse_problem:
        flux = -1. 
        rightbc = 1.
        source = injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05)
    # Initialize stochastic process prior
    y_gp_mean, y_gp_cov = init_gaussian_process(xgrid, y_mean=1., y_var=0.3, lengthscale=0.3, order=1.)

    # Initialize truncation error of dimensionality reduction
    if args.eval_trunc_err:
        n_truncs = 10
        truncs = n_truncs -1+ n_truncs*np.arange(n_truncs,dtype=int) # number of stochastic dimensions until truncation
    else:
        n_truncs = 1
        if args.inverse_problem:
            kl_trunc = args.poly_deg# 9
        else:
            kl_trunc = args.poly_deg# n_grid-11 # Truncation id 
        truncs = np.array((kl_trunc,))
    print('trunc', truncs.shape, truncs, kl_trunc)

    # Initialize dimensionality reduction of gaussian process 
    if args.pce:
        expansion = 'polynomial_chaos'
        poly_deg = args.poly_deg
        kl_trunc = poly_deg
        z_init = None
    else:
        expansion = 'KL'
        poly_deg = 0
        z_init = np.random.normal(0., 1., (kl_trunc))

    # Initialize model
    model_args = {'xgrid':xgrid, 
        'y_gp_mean':y_gp_mean, 'y_gp_cov':y_gp_cov, 
        'trunc':kl_trunc, 'expansion':expansion, 'z_candidate':z_init, 
        'poly_deg':poly_deg, #'pce_coefs': pce_coefs,
        'flux':flux, 'source':source, 'rightbc':rightbc, 'x_obs':x_obs}


    # Initialize stochastic differential equation
    stochDiffEq = StochDiffEq()

    if args.inverse_problem:
        obs_noise = 0.0001 # 1e-4
        prop_std = np.sqrt(0.01)

        model_args['z_candidate'] = z_init

        if args.mcmc:
            # Start for MCMC
            chain, lnprobs, accept_rates, logs = mh_sampler(z_init,
                proposal_fn=gaussian_proposal, proposal_fn_args={'std':prop_std},
                model=get_sol_w_gp_prior, model_args=model_args,
                lnpost_fn=gaussian_lnpost, lnpost_fn_args={'u_obs':u_obs, 'obs_noise':obs_noise}, 
                n_samples=n_samples, warmup_steps=warmup_steps)

        elif args.ensemble_kf:
            # Infer KL-expansion's random variable, z, with Ensemble Kalman Filter
            z_init = np.zeros(kl_trunc)
            prop_std = 0.1 #np.sqrt(1.)#np.sqrt(0.01)
            z_post, logs = ensemble_kalman_filter(z_init,
                proposal_fn=gaussian_proposal, proposal_fn_args={'std':prop_std},
                model=get_sol_w_gp_prior, model_args=model_args,
                u_obs = u_obs, obs_noise=obs_noise, 
                n_samples=n_samples)

    else:
        # Load simulation data instead of 
        if args.sim_datapath is not None:
            with open(args.sim_datapath, 'rb') as handle:
                logs = pickle.load(handle)            
        else:
            if args.parallel: 
                model_r, model_tasks = init_preprocessing(fn=get_sol_w_gp_prior, parallel=True)

            # Sample solution
            sample_y = None
            logs = n_samples * [None]
            for n in range(n_samples):
                if n%10==0: print('n', n)
                for t, trunc in enumerate(truncs):
                    # Sample stochastic parameters
                    if args.pset2:
                        model_args['flux'] = np.random.normal(mu_f, var_f, 1)
                    # Sample solution
                    u_ests, logs[n], sample_y = get_sol_w_gp_prior(**model_args)
                    
                    model_args['sample_y'] = sample_y
            
            # Parse parallel tasks
            # model_tasks = get_parallel_fn(model_tasks)
            # for n in range(n_samples):
            #    _, logs[n] = model_tasks[n]
            
            # Store data
            with open('pce.pickle', 'wb') as handle:
                pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    k, Y, u, trunc_errs, coefs = parse_logs(logs)

    # Estimate various parameters with neural network
    predict_pce_coefs = True
    if args.nn_pce and predict_pce_coefs:
        from neural_net import get_param_nn
        if args.est_param_nn=='pce_coefs':
            # Approximate PCE coefficients with neural network
            coefs_nn,_ = get_param_nn(xgrid, coefs[0,:,:], target_name=args.est_param_nn, plot=True)

            # Get PCE sampling function with given neural net coefs
            _, _, _, _, sample_pce_nn_coefs = sample_low_dim_gaussian_process(xgrid, 
                mu_y=y_gp_mean, cov=y_gp_cov,
                trunc=kl_trunc, expansion='polynomial_chaos', z_candidate=None, 
                poly_deg=poly_deg, pce_coefs=coefs_nn)

            # Set PCE sampling fn as function to sample from model
            model_args['sample_y'] = sample_pce_nn_coefs

            # Draw PCE samples
            logs = n_samples * [None]
            for n in range(n_samples):
                # Sample stochastic parameters
                if args.pset2:
                    model_args['flux'] = np.random.normal(mu_f, var_f, 1)
                u_ests, logs[n], _ = get_sol_w_gp_prior(**model_args)
                
            k, Y, u, _, _ = parse_logs(logs)
        elif args.est_param_nn=='k':
            alpha_indices = multi_indices(ndim=kl_trunc, poly_deg=poly_deg, order='total_degree')
            #n_samples = 150#k.shape[0]#20
            x_in = np.repeat(xgrid[np.newaxis,:], repeats=n_samples, axis=0)
            # TODO: delete (this is just to test if mean k can be estimated)
            #k = np.repeat(k.mean(axis=0)[np.newaxis,:],repeats=n_samples, axis=0) # Samples
            #k = np.repeat(np.exp(np.ones(n_grid))[np.newaxis,:],repeats=n_samples, axis=0) # Ground truth sample mean
            #k = np.repeat(k_true[np.newaxis,:],repeats=n_samples, axis=0) # Observations
            alpha_indices = alpha_indices[:,4:] # Omit constant pce coefficient
            pce_coefs, ks_nn = get_param_nn(xgrid=x_in, y=k, target_name=args.est_param_nn, alpha_indices=alpha_indices, 
                batch_size=1000, n_epochs=20, n_test_samples=n_samples, plot=True)
            if True:
                plot_k_vs_ks_nn(k, ks_nn)
            # Compute log-permeability
            ys_nn = np.log(np.where(ks_nn>0, ks_nn, 1.))
            # Compute solution
            us_nn = np.zeros((n_samples, n_grid))
            for n in range(n_samples):
                if args.pset2:
                    flux_sample = np.random.normal(mu_f, var_f, 1)
                us_nn[n,:] = stochDiffEq.diffusioneqn(xgrid, F=flux_sample, k=ks_nn[n,:], source=source, rightbc=rightbc)[:,0]

            k = ks_nn
            y = ys_nn
            u = us_nn
    # Plot k
    plot_k(xgrid=xgrid, k=k, k_true=k_true, y_gp_mean=y_gp_mean, y_gp_cov=y_gp_cov)
    if args.mcmc:
        plot_accept_rates(accept_rates)

    # Compute ground-truth solution with given, k
    if args.inverse_problem:
        u_true = stochDiffEq.diffusioneqn(xgrid, F=flux, k=k_true, source=source, rightbc=rightbc)[:,0]
    else:
        u_true = None

    if args.eval_trunc_err: 
        plot_trunc_err(truncs, trunc_errs)

    x = 0.0
    x_id = np.abs(x_obs - x).argmin()
    u_at_x = u[:, x_id]

    # Calculate estimated sample mean at each sampling iteration
    #ux_cum_mean = np.divide(np.cumsum(ux)*np.ones(ux.shape[0]), np.arange(ux.shape[0], dtype=float)+1)
    #ux_cum_mean = np.asarray([np.mean(ux[:n]) for n in range(n_samples)])
    # Calculate estimated standard deviation of the estimated mean at each sampling iteration 
    #ux_cum_std = np.asarray([np.std(ux_cum_mean[:n+1]) for n in range(n_samples)]) 

    # Calculate statistics
    ux_cum_mean = np.zeros((n_samples))
    ux_cum_std = np.zeros((n_samples))
    ux_cum_sem = np.zeros((n_samples))
    ux_cum_var = np.zeros((n_samples))
    conf_bnds = 0.95
    ux_conf_int = np.zeros((n_samples, 2))
    for n in range(n_samples):
        ux_cum_mean[n] = np.mean(u_at_x[:n+1])
        ux_cum_std[n] = np.std(u_at_x[:n+1], ddof=1) # Use n-1 in denominator for unbiased estimate. 
        ux_cum_sem[n] = st.sem(u_at_x[:n+1])
        ux_cum_var[n] = np.var(u_at_x[:n+1])
        if n>0:
            ux_conf_int[n,:] = st.t.interval(conf_bnds, n-1, loc=ux_cum_mean[n], scale=st.sem(u_at_x[:n+1]))

    """                        
    for n in range(n_samples):
        est_n = ux_cum_mean[:n+1]
        #if n==0:
        #    ux_conf_int[n,:] = np.array([ux_cum_mean[:n+1], ux_cum_mean[:n+1]])[0]
        #else:
        ux_conf_int[n,:] = st.t.interval(conf_bnds, est_n.shape[0]-1, loc=np.mean(est_n), scale=st.sem(est_n))
    """
    #print('ux_con', ux_conf_int)
    #print(f'u(x={x}, w) = {ux}')
    print('plot')
    stochDiffEq.plot_sol(xgrid, u, u_at_x, ux_cum_mean, ux_cum_std, ux_cum_sem, ux_conf_int, ux_cum_var, x, 
        xgrid, u_true)