# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:59:36 2020

@author: qtckp
"""

import numpy as np
import math
from plotting_heatmap import heatmap



import translation_only




# start data

a = np.arange(1,30)
b= np.arange(0,50)


t = np.arange(0,101)
ut = np.sin(2*math.pi/50 * t)



# testing methods


Wab = translation_only.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

#%timeit translation_only.DWT_signal(ut, a, b, t, a.size, b.size, t.size, 1, 8)

heatmap(Wab, a, b,)







