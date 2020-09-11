# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:03:31 2020

@author: qtckp
"""




import numpy as np
import math
import cmath


def GaborWavelet(omega, t, Gabor_coef):
    
    c1 = 0.3251520240633*math.sqrt(omega)
    c2 = -0.5*Gabor_coef
    c3 = omega*0.187390625129278
    
    res = np.array(c2*(t * c3)**2, dtype = np.complex128)
    
    res.imag =  omega*t
    
    return c1*np.exp(res)



def DWT_signal(ut, a, b, t0, AA, BB, TT, omega0, Gabor_coef):

    h = (t0[1]-t0[0])
    
    Wab = np.empty((BB, AA), dtype = np.complex128)
    
    for j in range(AA):
        for i in range(BB):
            
            t_cur=(t0-b[i])/a[j] 
                              
            psi_t = GaborWavelet(omega0, t_cur, Gabor_coef).conjugate()

            f_psi = psi_t * ut
            
            Wab[i,j] = 0.5*(f_psi[0]+f_psi[-1]) + np.sum(f_psi[2:TT-1])
            
    
    Wab = np.multiply(Wab, h/np.sqrt(a))
    
    return Wab

     
