#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:27:14 2021

@author: ghobson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A function that can be called to implement Crank-Nicolson (implicit, 2nd order
in both space and time, as efficient as explicit) to solve the 1D diffusion 
equation on the domain (0,L). 

Inputs: N: number of grid points
dt: time step
L: length of grid
t: current time step number
g0: left boundary condition (for the moment, a scalar)
g1: right boundary condition (for the moment, a scalar)
kappa: the diffusion coefficient (constant for now)
un: previous time step (a vector)

Sets up a sparse matrix A and uses scipy.sparse.linalg.spsolve(A,b) to solve

Uses previous time step to find next time step. 

"""

def laplace_cd(N,L,g0,g1,kappa,rhs):
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    h = L/(N-1) # space step size
    r = kappa/(h**2) # define r, later used in matrix system
    
    # set up matrices for A x = b system
    
    # A is a tridiagonal and sparse matrix in CSR format
    upper = np.zeros(N-1) # for the off diagonals
    upper[:] = -r
    upper[0] = 0
    lower = np.zeros(N-1) # for the off diagonals
    lower[:] = - r
    lower[-1] = 0
    main_diag = np.ones(N) # for the center diagonal
    main_diag[1:-1] *= (0+2*r) 
    A = scipy.sparse.diags([main_diag, lower, upper],[0,-1,1],format="csr")
    #print('A',A.todense())
    # set first 'solution' to be the previous time step
    # create rhs that we can fill in
    b = np.zeros(N)
    b[0] = g0() # meeting bc
    '''
    b[1] = r*(g0()) + rhs(1*h)
    # second row to second-to-last row
    for j in range(2,N-2):
        b[j] = 0 + rhs(j*h)
    b[-2] = r*(g1()) + rhs((N-2)*h)
    '''
    # second row to second-to-last row
    for j in range(1,N-1):
        b[j] = 0 + rhs(j*h)
    b[-1] = g1() # meeting bc
    # using spsolve: currently A is in CSR form
    U_out = scipy.sparse.linalg.spsolve(A,b)
    t2 = perf_counter()
    dt = t2-t1
#    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    
    return U_out
