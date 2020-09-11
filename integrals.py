# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:21:22 2020

@author: qtckp
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

ww = 1
Gabor_coef = 8

p = Gabor_coef*(ww*0.187390625129278)**2/2
k = 0.3251520240633 * math.sqrt(ww)
r = ww

def f(w):
    
    if w ==0:
        return 0
    
    p2 = 2*p
    
    return k/math.sqrt(p2) * math.exp(-(r+w)**2/p2)/math.fabs(w)

quad(f,-10, -0.00001)



x = np.linspace(-10,10,1000)
y = [f(w) for w in x]

plt.plot(x, y)





