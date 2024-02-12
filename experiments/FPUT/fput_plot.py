import numpy as np
import matplotlib.pyplot as plt
#import tikzplotlib
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

res1 = np.load("data/50ptcls_2-12h_smpls_even_nostop.data.npy")
res2 = np.load("data/50ptcls_14-26h_smpls_even_nostop.data.npy")
res = np.concatenate((res1,res2))
res_random = np.load("data/50ptcls_2-26h_smpls_even_noneven.data.npy")

ticks_x = [2,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600]
ticks_x = [2*i for i in range(1,14)]

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,4/cmtoinch))

ax = plt.gca()
ax.semilogy(ticks_x,res,'x',ls='--',markersize=7,mew=2,label="$\\kappa = 1, \\beta = 0.7$",c='tab:blue')
ax.semilogy(ticks_x,res_random,'+',ls=':',markersize=9,mew=2,label="random $\\kappa, \\beta$",alpha=.8,c='tab:orange')
ax.legend(frameon=False)
ax.set_xlabel('$\\times 10^2$ Number of samples')
ax.set_ylabel('Residuum')
ax.tick_params(direction="in")
ax.minorticks_off()

plt.savefig("figures/fput.pdf",format='pdf',bbox_inches='tight')
plt.show()
