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


def normalize(arr):
    return arr/np.abs(arr).max()


def St(t, Wab, a, b, omega, Gabor_coef):
    #psi = np.fromfunction(lambda i, j: GaborWavelet(omega, (t-b[i])/a[j], Gabor_coef), ( b.size,a.size), dtype = int)
    
    psi = np.empty((b.size, a.size), dtype = np.complex128)
    
    for i in range(b.size):
        for j in range(a.size):
            psi[i,j] = GaborWavelet(omega, (t-b[i])/a[j], Gabor_coef)

    answer = Wab * psi
    
    return np.divide(answer, a**2.5).sum() * (a[1]-a[0])*(b[1]-b[0])



def St_array(times, Wab, a, b, omega, Gabor_coef):
    
    res = []
    
    coef = (a[1]-a[0])*(b[1]-b[0])
    
    adeg = a**2.5
    
    psi = np.empty((b.size, a.size), dtype = np.complex128)
    
    for t in times:
        
        for i in range(b.size):
            
            for j in range(a.size):
                psi[i,j] = GaborWavelet(omega, (t-b[i])/a[j], Gabor_coef)
    
        answer = Wab * psi
        
        res.append(np.divide(answer, adeg).sum() * coef)
    
    return np.array(res)




a = np.arange(0.1,30,0.5)
b= np.arange(-20,105,0.5)


t = np.arange(0,101)

ut_1 = np.sin(2*math.pi/50 * t)
ut_2 = np.sin(2*math.pi/50 * t)+np.sin(2*math.pi/100 * t)
ut_3 = np.sin(2*math.pi/50 * t) + 4*np.sin(2*math.pi/10 * t)

omega = 1
g = 8


voc = {
       'sin(2pi/50 t)':ut_1,
       'sin(2pi/50 t)+sin(2pi/100 t)':ut_2,
       'sin(2pi/50 t) + 4sin(2pi/10 t)':ut_3
       }



for name, ut in voc.items():

    Wab = DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, g)
    
    
    heatmap(Wab.real, a, b, name = f'W(a,b) from {name}')
    
    
    #s = np.array([St(time, Wab, a, b, omega, g) for time in t])
    
    s = St_array(t, Wab, a, b, omega, g)
    
    plt.plot(t, normalize(ut), label = 'real', color = 'blue', linewidth = 4)
    plt.plot(t, normalize(s), label = 'predict', color = 'red', linestyle = '--', linewidth = 3)
    
    plt.xlabel('time')
    plt.legend()
    plt.title(f'forward and backward transforms\n{name}')
    
    plt.savefig(f'images/{name.replace("/", " div ")}.png', dpi = 350)
    
    plt.show()




