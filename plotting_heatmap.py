# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:56:15 2020

@author: qtckp
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()




def heatmap(data, anet, bnet,  cmap = 'twilight', dpi = 350, name = 'W(a,b)'):
        
    b, a = np.meshgrid(anet, bnet)
    
    #a = a.T
    #b = b.T
    
    l_a=a.min()
    r_a=a.max()
    l_b=b.min()
    r_b=b.max()
        
    l_c, r_c  = data.min(), data.max()
    
    levels = MaxNLocator(nbins=15).tick_values(l_c, r_c)
    cmap = plt.get_cmap(cmap)
    

    figure, axes = plt.subplots()
        
    c = axes.contourf(a, b, data, cmap=cmap, levels = levels, vmin=l_c, vmax=r_c)
        
    axes.set_title(name)
    axes.axis([l_a, r_a, l_b, r_b])
    
    axes.set_xlabel('b')
    axes.set_ylabel('a')
    
    figure.colorbar(c)
               
    figure.savefig(f'images/{name.replace("/", " div ")}.png', dpi = dpi)

    plt.show()    
    
    plt.close(figure)
        
        
        


if __name__ == '__main__':
    
    a = np.arange(-10,20)
    b = np.arange(5, 60)
    data = np.sin(np.arange(a.size*b.size).reshape(b.size, a.size))
    
    heatmap(data, a, b)










