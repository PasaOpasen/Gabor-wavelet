# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:23:24 2020

@author: qtckp
"""

import numpy as np
import math
from plotting_heatmap import heatmap
import matplotlib.pyplot as plt

from numba_vectorization import DWT_signal, GaborWavelet



def St(t, Wab, a, b, omega, Gabor_coef):
    #psi = np.fromfunction(lambda i, j: GaborWavelet(omega, (t-b[i])/a[j], Gabor_coef), ( b.size,a.size), dtype = int)
    
    psi = np.empty((b.size, a.size), dtype = np.complex128)
    
    for i in range(b.size):
        for j in range(a.size):
            psi[i,j] = GaborWavelet(omega, (t-b[i])/a[j], Gabor_coef)

    answer = Wab * psi
    
    return np.divide(answer, a**2.5).sum() * (a[1]-a[0])*(b[1]-b[0])



a = np.arange(0.1,30,0.5)
b= np.arange(-20,105,0.5)


t = np.arange(0,101)
ut = np.sin(2*math.pi/50 * t)#+np.sin(2*math.pi/100 * t)

omega = 1
g = 8


Wab = DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, g)


heatmap(Wab.real, a, b,)



s = np.array([St(time, Wab, a, b, omega, g) for time in t])


plt.plot(t, ut)
plt.plot(t, s)

plt.show()




