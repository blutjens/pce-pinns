"""
Accumulation of plotting functions
"""
import numpy as np 
import matplotlib.pyplot as plt

def ridge_plot_prob_dist_at_sample_x(xs, uxs):
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
    plt.savefig('doc/figures/plot_prob_dist_at_sample_x.png')

    plt.show()

def plot_sol(xgrid, u, ux, ux_stats, x, xgrid_true, u_true=None):
    """
    Plot solution and statistics
    Inputs:
    xgrid_true np.array(n_grid): grid of true solution
    u_true np.array(n_grid): true solution
    ux np.array(n_samples): Samples of solution, u, at x
    ux_stats {np.array(n_samples)}: Stats about the set of solutions, u(x)
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

    axs[1,1].plot(ux_stats['ux_cum_mean'][:], label=r'$\bar u_n = \mathbf{E}_{n^* \in \{0,..., n\}}[u_{n^*}(x{=}'+str(x)+',\omega)]$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_cum_mean'][:] + ux_stats['ux_cum_std'][:], 
        ux_stats['ux_cum_mean'][:] - ux_stats['ux_cum_std'][:], 
        alpha=0.4, color='blue',
        label=r'$\bar u_n \pm \sigma_n$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_cum_mean'][:] + ux_stats['ux_cum_sem'][:], 
        ux_stats['ux_cum_mean'][:] - ux_stats['ux_cum_sem'][:], 
        alpha=0.4, color='black',
        label=r'$\bar u_n \pm$ std err$_n$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_conf_int'][:,0], ux_stats['ux_conf_int'][:,1], 
        alpha=0.4, color='orange',
        label=r'$95\% conf. interval; P $')
    axs[1,1].set_xlabel('sample id, n')
    axs[1,1].set_ylabel(r'$u(x{=}'+str(x)+', \omega)$')
    axs[1,1].set_ylim((ux_stats['ux_cum_mean'].min()-np.nanmax(ux_stats['ux_cum_std']), ux_stats['ux_cum_mean'].max()+np.nanmax(ux_stats['ux_cum_std'])))
    axs[1,1].set_title('sample mean over iterations')
    axs[1,1].legend()

    axs[1,2].plot(ux_stats['ux_cum_var'][:], label=r'$\sigma^2_n$')
    axs[1,2].set_xlabel('sample id, n')
    axs[1,2].set_ylabel(r'$\sigma^2(u(x{=}'+str(x)+', \omega)$')
    axs[1,2].fill_between(range(ux_stats['ux_cum_var'].shape[0]), 
        ux_stats['ux_cum_var'][:] + ux_stats['ux_cum_sem'][:], 
        ux_stats['ux_cum_var'][:] - ux_stats['ux_cum_sem'][:], 
        alpha=0.4, color='black',
        label=r'$\sigma^2_n \pm$ std err$_n$')
    axs[1,2].set_title('sample std dev. at x over iterations')
    axs[1,2].legend()

    #axs[1,1].plot(ux_stats['ux_conf_int'][:])
    #axs[1,1].set_xlabel('sample id, n')
    #axs[1,1].set_ylabel(r'$V[E[u_{n\prime}(x='+str(x)+',w)]_0^n]$')
    #axs[1,1].set_title('95\% conf. int.')


    fig.tight_layout()
    plt.savefig('doc/figures/proj1_3.png')

    # Plot prob. dist. at sample locations
    xs = np.linspace(0,1, 5)#[0., 0.25, 0.5, 0.75]
    ux_samples = np.empty((len(xs), u.shape[0]))
    for i, x_sample in enumerate(xs):
        x_id = (np.abs(xgrid - x_sample)).argmin()
        ux_samples[i,:] = u[:, x_id]

    ridge_plot_prob_dist_at_sample_x(xs, ux_samples)

def plot_trunc_err(truncs, trunc_err):
    # https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    # Input
    # truncs np.array(n_truncations)
    # trunc_err np.array(n_truncations)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(truncs, trunc_err)
    axs.set_xlabel(r'truncation, $r$')
    axs.set_ylabel(r'truncation err.')
    fig.savefig('doc/figures/ai_trunc_err.png')

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
    fig.savefig('doc/figures/mh_accept_rates.png')

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
    fig.savefig('doc/figures/k_gp_vs_msmts.png')

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
    fig.savefig('doc/figures/mh_chain.png')

    # Plot mh lnprobs
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(sample_ids, lnprobs[:])
    axs.set_ylabel(r'log-probability, $p(u^{(t)}|X)$')
    axs.set_xlabel(r'sample id, $t$')
    axs.set_ylim((-5.,lnprobs.max()+1.))
    fig.tight_layout()
    fig.savefig('doc/figures/mh_lnprobs.png')

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
    fig.savefig('doc/figures/mh_autocorr.png')

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
    fig.savefig('doc/figures/mh_param_post.png')

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
    fig.savefig('doc/figures/enkf_param_post.png')

def plot_k_vs_ks_nn(xgrid, k, k_samples_nn):
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
    plt.savefig('doc/figures/nn_k_samples.png')

"""
Neural net plots
""" 
def plot_nn_k_samples(xgrid, k_samples):
    plt.figure(figsize=(15,8))
    n_samples = k_samples.shape[0]
    k_mean = k_samples.mean(axis=0)
    k_std = k_samples.std(axis=0)
    plt.plot(xgrid, k_mean, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_mean + k_std, k_mean - k_std, 
        alpha=0.3, color='blue',
        label=r'$\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('doc/figures/nn_k_samples.png')

def plot_train_curve(loss_stats):
    #train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    plt.figure(figsize=(15,8))
    plt.plot(np.arange(len(loss_stats['train']))+1, loss_stats['train'])
    plt.yscale('log')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title('Loss/Epoch')
    plt.savefig('doc/figures/train_val_loss.png')

def plot_nn_pred(x, y):
    plt.figure(figsize=(15,8))
    n_out = y.shape[1]
    for i in range(n_out):
        plt.plot(x, y[:,i], label=r'$C_{\alpha}$, '+str(i))
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"$\hat C_\alpha(x)$")
    plt.legend()
    plt.title('PCE coefs')
    plt.savefig('doc/figures/nn_pce_coefs.png')
