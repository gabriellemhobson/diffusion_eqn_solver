#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calling pde solver and plotting the results. Laplace eqn, centred difference
"""
import matplotlib.pyplot as plt
import numpy as np
import laplace_cd as solver
import sympy as sympy

def mms():
    x = sympy.Symbol('x')
    return x, 1 + 1.1*x + 3*sympy.sin(sympy.pi*x)

def mms_rhs():
    x, u = mms()
    du = u.diff(x)
    du2 = du.diff(x)
    return x, - du2

def test_cn():
    hs = np.zeros((5))
    nrm = np.zeros((5))
    for i in range(1,6):
        #c = int(2*i)
        N = int(16*2**i) # number of grid points
        print('N',N)
        L = float(2) # length of grid
        kappa = 1
        #a = 1
        
        def true(coor):
            x,u = mms()
            result = np.copy(coor)
            for i in range(len(result)):
                result[i] = float(u.subs(x, coor[i]))
            return result
            # the below true solution works on (0,1)
            # return np.exp(-np.pi**2*t)*np.sin(np.pi*x)+0.1*np.exp(-np.pi**2*1e4*t)*np.sin(100*np.pi*x)

        # define boundary conditions
        def g0():
            
            x, u = mms()
            eval_u = u.subs(x,0.0)
            print(eval_u)
            return float(eval_u)
            # return 0
    
        def g1():
            x, u = mms()
            eval_u = u.subs(x,L)
            return float(eval_u)
            # return 0
            
        def rhs(coor):
            x, rhs = mms_rhs()
            eval_rhs = float(rhs.subs(x,coor))
            return eval_rhs
            
        x = np.linspace(0,L,N)
        
    
        un = solver.laplace_cd(N,L,g0,g1,kappa,rhs)
        
        plt.clf()
#        plt.ylim(0,1)
        plt.plot(x,un,color='blue')
        plt.scatter(x,true(x),marker='o',color='orange')
        # plt.title()
        plt.show()
        
        hs[i-1] = L/(N-1)
        nrm[i-1] = np.linalg.norm(un-true(x))
        print('Norm',('%1.4e'%nrm[i-1]))
        print('')

    print('h',hs)
    print('nrm',nrm)
    
    fig = plt.figure()
    ax=plt.gca()
    ax.scatter(hs,nrm,c="blue")
    ax.set_yscale('log')
    ax.set_xscale('log')
    m, c = np.polyfit(np.log10(hs), np.log10(nrm), deg=1)
    ax.plot(hs, np.exp(m*np.log(hs) + c), color='red') # add reg line
    ax.grid(b='on')
    plt.show()
    print('Slope:',m)
    
if __name__ == '__main__':
    test_cn()
#    test_pde_cn()

