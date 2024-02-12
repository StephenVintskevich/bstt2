#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:15:19 2022

@author: goette
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

font_dir = ['/Users/jonas/fonts/Source_Sans_Pro']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
SIZE=9
rcParams['font.family'] = 'Source Sans Pro'
rcParams['font.size']=SIZE
rcParams['axes.titlesize']=SIZE
rcParams['axes.labelsize']=SIZE
rcParams['xtick.labelsize']=SIZE
rcParams['ytick.labelsize']=SIZE
rcParams['legend.fontsize']=SIZE
rcParams['figure.titlesize']=SIZE

res = np.load("data/exp_3_noise_magnetic.data.npy")
print(res.shape)
res = np.mean(res,axis=4)

ticks_loc_x =  [2*i for i in range(1,10)]
size =[20,50]
interaction = [5]
sigma = [1e-1,1e-2,1e-3,1e-4]
sigma_str = ["$10^{-1}$","$10^{-2}$","$10^{-3}$","$10^{-4}$"]

markers = ["x","+"]
markersizes = [8,10]
linestyles = ["--","-.",":"]
colors = ['tab:blue','tab:orange','tab:green']
alphas = [.6,1.]

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,5/cmtoinch))

ax = plt.gca()

for i,s in enumerate(sigma[:-1]):
    for j,d in enumerate(size):
        ax.semilogy(ticks_loc_x,
                    res[j,0,:,i],
                    markers[j],
                    ms=markersizes[j],
                    ls=linestyles[i],
                    label=f"$\\sigma = {s}$, $d = {d}$",
                    mew=2,
                    c=colors[i],
                    alpha=alphas[j])

ax.set_xlabel("$\\times 10^2$ Number of samples")
ax.set_ylabel("Residuum")
ax.tick_params(direction="in")
ax.minorticks_off()

handles_sigma = [plt.plot([],marker="",ls=linestyles[i],c=colors[i])[0] for i in range(len(sigma)-1)]
labels_sigma = ["$\\sigma = $" + s for s in sigma_str[:-1]]
legend_sigma = plt.legend(handles_sigma,labels_sigma,frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.45,1.15),borderpad=0.)
fig.add_artist(legend_sigma)

handles_size = [plt.plot([],marker=markers[j],ms=markersizes[j],mew=2,ls="",c='k')[0] for j in range(len(size))]
labels_size = [f"$d = {d}$" for d in size]
legend_size = plt.legend(handles_size[::-1],labels_size[::-1],frameon=False)
fig.add_artist(legend_size)

plt.savefig('figures/exp_3_noise_magnetic.pdf',format='pdf',bbox_inches='tight')
plt.show()
