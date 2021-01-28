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
        plt.savefig('doc/figures/hermite_poly_Herm.png')

    return c_alphas
