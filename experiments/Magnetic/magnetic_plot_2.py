import matplotlib.pyplot as plt
import numpy as np
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

res = np.load("data/exp_2_magnetic.data.npy") #order x interaction x trainSampleSize (5,2,7)
res = res[0:3,:,:]

ticks_loc_x = [5+2*i for i in range(1,8)]
system_sizes =[10,20,30]
interaction = [5]

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,5./cmtoinch))

ax = plt.gca()
marker = ['x','+','.']
marker_sizes = [7,9,10]
line = ['--',':','-.']
alphas = [.5,.9,1.]

for i,size in enumerate(system_sizes):
    ax.semilogy(ticks_loc_x,res[i,0,:],marker[0],ms=marker_sizes[0],ls=line[i],label=f"$d = ${size}",mew=2,alpha=alphas[i])
    
ax.set_xlabel('$\\times 10^3$ Number of samples')
ax.set_ylabel('Residuum')
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1],labels[::-1],frameon=False)
ax.tick_params(direction="in")
ax.minorticks_off()

plt.savefig('figures/exp_2_magnetic_plot.pdf',format='pdf',bbox_inches='tight')
plt.show()
