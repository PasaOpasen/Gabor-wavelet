# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:58:00 2020

@author: qtckp
"""

import numpy as np
import math
import cmath
from plotting_heatmap import heatmap


def GaborWavelet(omega, t, Gabor_coef):
    return 0.3251520240633*math.sqrt(omega)*cmath.exp(-complex(0.5*Gabor_coef*t*t*(omega*0.187390625129278)**2, omega*t))



def DWT_signal(ut, a, b, t0, AA, BB, TT, omega0, Gabor_coef):

    h_step=t0[1]-t0[0];
    
    psi_t = np.empty(TT, dtype = np.complex128)
    Wab = np.empty((BB, AA))
    
    for j in range(AA):
        for i in range(BB):
            for k in range(TT):

                t_cur=(t0[k]-b[i])/a[j]               
                psi_t[k] = GaborWavelet(omega0, t_cur, Gabor_coef).conjugate()

            f_psi = psi_t * ut
            Wab[i,j] = 0.5*(f_psi[0]+f_psi[-1])
            for k in range(2, TT-1):
                Wab[i,j] = Wab[i,j] + f_psi[k]
        
        Wab[:,j] = Wab[:,j]*h_step/math.sqrt(a[j])
    
    return Wab







a = np.arange(1,30)
b= np.arange(0,50)

t = np.arange(0,101)
ut = np.sin(2*math.pi/50 * t)


Wab = DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

#%timeit DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

heatmap(Wab, a, b,)

