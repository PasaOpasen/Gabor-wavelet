# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:59:36 2020

@author: qtckp
"""

import numpy as np
import math
from plotting_heatmap import heatmap



import translation_only
import light_vectorization
import strong_vectorization
import numba_just
import numba_strong
import numba_vectorization
import numba_vec_parallel



# start data

a = np.arange(1,30,0.5)
b = np.arange(0,50,0.5)

t = np.arange(0,101, 1)
ut = np.sin(2*math.pi/50 * t) + 100*np.cos(0.4*t)/(t*t+1)

omega = 1

Gabor_coef = 8


# testing methods


#Wab = translation_only.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

#Wab = light_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

#Wab = strong_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

Wab = numba_just.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

#Wab = numba_strong.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

Wab = numba_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

Wab = numba_vec_parallel.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)


%timeit translation_only.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit light_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit strong_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_strong.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_just.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_vec_parallel.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)


heatmap(Wab.real, a, b,)







