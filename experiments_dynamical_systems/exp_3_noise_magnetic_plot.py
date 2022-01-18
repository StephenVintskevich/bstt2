#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:15:19 2022

@author: goette
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

res = np.load("exp_3_noise_magnetic.data.npy") #order x interaction x trainSampleSize (5,3,7)
print(res.shape)
res = np.mean(res,axis=4)
print(res.shape)

t = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1]
label_format = '{:,.0f}'
ticks_loc_x =  [2*i for i in range(1,10)]
ticks_loc_y =[20,50]
interaction = [5]
sigma = [1e-1,1e-2,1e-3,1e-4]



fig,axes = plt.subplots(figsize=(16, 6), nrows=3)
for k, ax in enumerate(axes.flat):
    pos = ax.imshow(res[:,0,:,k],cmap="Greens",norm=LogNorm(vmin=np.min(res.reshape(-1)[res.reshape(-1)> 0]), vmax=np.max(res)))
    
    ax.set_title(f'Sigma: {sigma[k]}')
    
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(ticks_loc_x))))
    ax.set_xticklabels([label_format.format(x) for x in ticks_loc_x])
    
    ax.yaxis.set_major_locator(mticker.FixedLocator([0,1]))
    ax.set_yticklabels([label_format.format(x) for x in ticks_loc_y])
    ax.set_ylabel('Order')
    ax.set_xlabel('Number of Samples (times 100)')

fig.tight_layout()

cbar = fig.colorbar(pos, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_ticks(t)
cbar.set_ticklabels(t)

plt.savefig('exp_3_noise_magnetic')
plt.show()
