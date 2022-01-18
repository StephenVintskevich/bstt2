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

res = np.load("exp_1_lennardjones.data.npy") #order x interaction x trainSampleSize (5,3,7)
print(res.shape)

t = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1]
label_format = '{:,.0f}'
ticks_loc_x =   [3000,5000,7000,10000]
ticks_loc_y =[10,20,30]
interaction = [5]


fig = plt.figure(figsize=(16, 4))

ax = plt.gca()
pos = ax.imshow(res[:,:,0],cmap="Greens",norm=LogNorm(vmin=1e-3, vmax=0.1))

ax.set_title(f'Interaction Range: {interaction[0]}')

ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(ticks_loc_x))))
ax.set_xticklabels([label_format.format(x) for x in ticks_loc_x])

ax.yaxis.set_major_locator(mticker.FixedLocator([0,1,2]))
ax.set_yticklabels([label_format.format(x) for x in ticks_loc_y])
ax.set_ylabel('Order')
ax.set_xlabel('Number of Samples')

fig.tight_layout()

cbar = fig.colorbar(pos, ax=ax, shrink=0.9)
cbar.set_ticks(t)
cbar.set_ticklabels(t)

plt.savefig('exp_1_lennard_jones_mod')
plt.show()
