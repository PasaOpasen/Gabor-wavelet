# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:51:01 2020

@author: qtckp
"""



import numpy as np
import math
import cmath
from numba import jit

@jit(nopython = True, fastmath = True)
def GaborWavelet(omega, t, Gabor_coef):
    return 0.3251520240633*math.sqrt(omega)*cmath.exp(complex(-0.5*Gabor_coef*(t*omega*0.187390625129278)**2, omega*t))


@jit(parallel = True)
def DWT_signal(ut, a, b, t0, AA, BB, TT, omega0, Gabor_coef):

    h_step = t0[1]-t0[0]
    
    psi_t = np.empty(TT, dtype = np.complex128)
    Wab = np.empty((BB, AA), dtype = np.complex128)
    
    for j in range(AA):
        for i in range(BB):
            for k in range(TT):

                t_cur=(t0[k]-b[i])/a[j]               
                psi_t[k] = GaborWavelet(omega0, t_cur, Gabor_coef).conjugate()

            f_psi = psi_t * ut
            
                 
            Wab[i,j] =  0.5*(f_psi[0]+f_psi[-1]) + np.sum(f_psi[2:TT-1])
        
    Wab = np.multiply(Wab, h_step/np.sqrt(a))
    
    return Wab

