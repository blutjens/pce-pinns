"""
Differential equations
"""
import numpy as np

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
