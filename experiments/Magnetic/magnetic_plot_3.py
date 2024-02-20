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
rcParams['legend.fontsize']=SIZE-1
rcParams['figure.titlesize']=SIZE
rcParams['mathtext.default']='regular'

folder = "experiments/Magnetic/"

res = np.load(folder+"data/exp_3_noise_magnetic.data.npy")
print(res.shape)
res = np.mean(res,axis=4)

ticks_loc_x =  [2*i for i in range(1,10)]
size =[20,50]
interaction = [5]
sigma = [1e-1,1e-2,1e-3,1e-4]
sigma_str = ["$10^{-1}$","$10^{-2}$","$10^{-3}$","$10^{-4}$"]

markers = [".","D"]
markersizes = [7,4]
linestyles = ["-",(0,(5,5)),(0,(3,5,1,5))]
colors = ['tab:orange','tab:blue','tab:green']
facecolors = np.array([colors,['None','None','None']])

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,4.5/cmtoinch))

ax = plt.gca()

for i,s in enumerate(sigma[:-1]):
    for j,d in enumerate(size):
        ax.semilogy(ticks_loc_x,
                    res[j,0,:,i],
                    markers[j],
                    ms=markersizes[j],
                    ls=linestyles[i],
                    lw=.5,
                    label=f"$\\sigma = {s}$, $d = {d}$",
                    mew=1,
                    c=colors[i],
                    markerfacecolor=facecolors[j,i]
                    )

ax.set_xlabel("$\\times 10^2$ Number of samples")
ax.set_ylabel("Residuum")
ax.tick_params(direction="in")
ax.minorticks_off()

handles_sigma = [plt.plot([],marker="",ls=linestyles[i],lw=.5,c=colors[i])[0] for i in range(len(sigma)-1)]
labels_sigma = ["$\\sigma = $" + s for s in sigma_str[:-1]]
legend_sigma = plt.legend(handles_sigma,labels_sigma,frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.45,1.15),borderpad=0.1)
fig.add_artist(legend_sigma)

facecolorblack = ['k','None']
handles_size = [plt.plot([],marker=markers[j],ms=markersizes[j],markerfacecolor=facecolorblack[j],mew=1,ls="",c='k')[0] for j in range(len(size))]
labels_size = [f"$d = {d}$" for d in size]
legend_size = plt.legend(handles_size,labels_size,frameon=False)
fig.add_artist(legend_size)

plt.savefig(folder+'figures/exp_3_noise_magnetic.pdf',format='pdf',bbox_inches='tight',pad_inches=0)
plt.show()
