"""
Differential equation solver and boundary conditions
"""
import numpy as np
import src.utils.plotting as plotting

class StochDiffEq():
    def __init__(self, xgrid, rand_flux_bc=False, injection_wells=False):
        """
        Sets up 1D stochastic diffusion eqn's initial conditions and parameters
        
        Args:
            xgrid np.array(n_grid): Grid points
            rand_flux_bc bool: If true, use random flux for van neumann condition on left boundary
            inverse_problem bool: If true, sets source term to be a function  

        Attributes:
            flux function()->float: Flux at left-hand boundary, k*du/dx = -F; could be stochastic or deterministic  
            source np.array(n_grid) or float: Source term, either a vector of values at points in xgrid or a constant
            rightbc float: Dirichlet BC on right-hand boundary
        """
        self.rightbc = 1.
        self.flux = lambda: -1. 
        self.source = 5.
        if rand_flux_bc:
            mu_f = -1. #-2.
            var_f = .2 #.5
            self.flux = lambda: np.random.normal(mu_f, var_f, 1) # 
        elif injection_wells:
            self.source = injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05)

    def diffusioneqn(self, xgrid, k):
        """
        Solves 1-D diffusion equation with given diffusivity field k
        and left-hand flux F. Domain is given by xgrid (should be [0,1])
        
        Args:
            xgrid np.array(n_grid): Grid points
        
        Returns:
            usolution np.array(xgrid): Solution
        """
        # Sample stochastic parameters
        F = self.flux() 

        N = xgrid.shape[0] # Number of grid points
        h = xgrid[N-1]-xgrid[N-2] # step size; assuming uniform grid

        # Set up discrete system f = Au + b using second-order finite difference scheme
        A = np.zeros((N-1, N-1)) 
        b = np.zeros((N-1,1)) 
        if np.isscalar(self.source): 
            f = -self.source * np.ones((N-1,1))
        else:
            f = -self.source[:N-1,np.newaxis] # [:N] 

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
        b[N-2] = self.rightbc * (k[N-1] + k[N-2]) / (2.*np.power(h,2)) 

        # Solve it: Au = f-b
        uinternal = np.linalg.solve(A, (f-b))

        usolution = np.vstack((uinternal, self.rightbc)) 

        return usolution

def injection_wells(xgrid, n_wells=4, strength=0.8, width=0.05, plot=True):
    """
    Returns a function over xgrid that models injection wells
    
    Args: 
        xgrid np.array(n_grid): Grid
        n_wells int: Number of equidistantly placed wells
        strength float: Source strength, theta
        width float: Source width, delta
    
    Returns:
        source np.array(n_grid): Function that models injection wells
    """
    n_grid=xgrid.shape[0]

    pos_wells = np.linspace(xgrid.min(), xgrid.max(), num=n_wells+2) # equidistant wells
    pos_wells = pos_wells[1:-1] # remove wells at borders
    pos_wells = np.repeat(pos_wells[np.newaxis,:], repeats=n_grid,axis=0) # repeat for matrix math
    xgrid_w = np.repeat(xgrid[:,np.newaxis], repeats=n_wells,axis=1) # repeat for matrix math
    
    amp = strength / (width * np.sqrt(2.*np.pi)) # amplitude
    source = np.sum(amp * np.exp(-(xgrid_w - pos_wells)**2 / (2.*width**2)),axis=1) # sum over wells

    if plot:
        plotting.plot_source(xgrid, source)

    return source

def get_msmts(plot=False):
    """
    Returns a set of measurements of a the solved differential equation 

    Args:
        plot bool: If true, plots the measurements
    
    Returns:
        x_obs np.array(n_msmts): Grid of observation points
        u_obs np.array(n_msmts): Solution, u, at measurement locations
        k_true np.array(n_msmts): Ground-truth permeability, k, at and between measurement locations; probably noisy
        xgrid np.array(()): Grid corresponding to ground-truth permeability, k
    """
    x_obs = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u_obs = np.array([1.33324463455171, 1.36464171067521, 1.40744323365932, 1.40555567476256, 1.38851427638760, 1.39799450797509, 1.31587775071708, 1.23031611117801, 1.15017066712980, 1.06664471477387])
    k_true = np.array([4.60851932653568, 4.86913697560332, 4.62032247978475, 5.00739558303546, 4.82792526785225, 5.07901564393012, 4.66138567908013, 4.17793583225948, 4.12539240678111, 4.44239649473624, 3.92333780870393, 4.54533170853051, 4.60912071107044, 4.52595709855134, 4.07893866778093, 3.82307832614272, 3.32917917553473, 3.01300181823486, 2.84411210569811, 2.87006579934071, 2.89882059530642, 3.00884548070947, 2.91266269929508, 2.81452942762117, 2.49228002628364, 2.71739595852760, 3.03644913148034, 2.57126956371258, 3.34661706710215, 2.94035041736722, 2.76450418823743, 2.62201506635126, 2.67758311125735, 3.12653883764129, 2.82006727239076, 2.92333827548000, 2.96095777130370, 3.02618524256612, 2.86938905850092, 2.70856278568016, 2.78734680029447, 3.10554300673895, 2.69887845768822, 2.65547556555735, 2.54448709529566, 2.28154078928308, 2.05414576956807, 2.15512302854182, 2.02513576436545, 1.87145499501584, 1.62145222971342, 1.43214562583999, 1.54012048601883, 1.57796945154306, 1.60947965439388, 1.47991657877595, 1.43280284347803, 1.42363837350593, 1.27337927667739, 1.24573074923507, 1.17312198311588, 1.31585192605426, 1.44015136034874, 1.66857903299136, 1.50962174415947, 1.54635945079439, 1.57589884658520, 1.70268496644981, 1.48816588773939, 1.43441178934950, 1.30716586222675, 1.30448229931249, 1.35949239558635, 1.45074973209516, 1.31924083608101, 1.42582943243071, 1.44731165671863, 1.31590290213155, 1.54561861639610, 1.14077717049381, 1.16094734995639, 1.23679487215449, 1.29243734753453, 1.09482574667417, 1.01743297741724, 1.28626862411626, 1.22763470870996, 1.24686001335088, 1.16866417198729, 1.11627212135544, 1.15750020242220, 1.21062567892525, 1.09896520713583, 1.18421265203534, 1.18059404644927, 1.17768529209078, 1.22455198272902, 1.04759883491168, 1.11647043353682, 1.33203747787611, 1.81313704286957, 1.60459248681251, 1.83411925167631, 1.86578565385545, 1.85751416668662, 1.74414424967352, 1.73943691889385, 1.65779696531455, 1.83539329972025, 1.66501410132260, 1.77233391597251, 2.12184254631078, 2.08316076727839, 2.56829457862793, 2.89616947379460, 2.76305598716123, 2.83952732577347, 2.84136601757010, 3.30459311061616, 3.71748320354009, 3.39326458870804, 3.72997251289344, 3.99814053512306, 3.56505218268485, 2.97146782046398, 2.53762825756458, 2.60267064123793, 2.80376324967721, 2.88830670266253, 2.63016047253275, 2.96651061385475, 2.84645722915763, 2.31591604193898, 2.33897006308885, 2.54556736907830, 2.49416021480434, 2.58507404041490, 2.92341952128821, 3.03701111325719, 3.13978838256269, 2.68511723989237, 2.82033378099678, 3.00537274866223, 3.09752033239746, 3.37058892306905, 2.81517596314294, 2.62448795757048, 2.27282371468636, 2.12617721213605, 2.00544200770971, 1.95385075829803])
    xgrid = np.linspace(0.,1.,k_true.shape[0])
    Y_true = np.log(k_true)
    if plot:
        plotting.plot_msmt(x_obs, xgrid, u_obs, k_true, Y_true)

    return x_obs, u_obs, k_true, xgrid

