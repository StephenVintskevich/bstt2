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
rcParams['legend.fontsize']=SIZE-1
rcParams['figure.titlesize']=SIZE
rcParams['mathtext.default']='regular'

folder = "experiments/Magnetic/"

res = np.load(folder+"data/exp_2_magnetic.data.npy") #order x interaction x trainSampleSize (5,2,7)
res = res[0:3,:,:]

ticks_loc_x = [5+2*i for i in range(1,8)]
system_sizes =[10,20,30]
interaction = [5]

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,4/cmtoinch))

ax = plt.gca()
marker = '.'
marker_size = 7
line = ['-',(0,(5,5)),(0,(3,5,1,5))]
colors = ['tab:orange','tab:blue','tab:green']

for i,size in enumerate(system_sizes):
    ax.semilogy(ticks_loc_x,res[i,0,:],marker,ms=marker_size,ls=line[i],lw=.5,label=f"$d = ${size}",mew=1,c=colors[i])
    
ax.set_xlabel('$\\times 10^3$ Number of samples')
ax.set_ylabel('Residuum')
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,frameon=False)
ax.tick_params(direction="in")
ax.minorticks_off()

plt.savefig(folder+'figures/exp_2_magnetic_plot.pdf',format='pdf',bbox_inches='tight',pad_inches=0)
plt.show()
