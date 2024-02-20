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

res = np.load(folder+"data/exp_1_magnetic.data.npy") #order x interaction x trainSampleSize (5,3,7)
res = np.mean(res,axis=3)

t = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1]
label_format = '{:,.0f}'
ticks_loc_x = [1*i for i in range(1,16)]
ticks_loc_y =[10,20,30,40,50]
interaction = [5,9]

cmtoinch = 2.54
fig,axes = plt.subplots(figsize=(8.5/cmtoinch,7.5/cmtoinch),nrows=2)

letters = ['(a)','(b)']
positions = np.array([[-1.5,-1.],[-1.5,-1.]])

for k, ax in enumerate(axes.flat):
    pos = ax.imshow(res[:,k,:],cmap="Greens",norm=LogNorm(vmin=np.min(res.reshape(-1)[res.reshape(-1)> 0]), vmax=np.max(res[:,:,:])))
    
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(ticks_loc_x))))
    ax.set_xticklabels([label_format.format(x) for x in ticks_loc_x])
    
    ax.yaxis.set_major_locator(mticker.FixedLocator([0,1,2,3,4]))
    ax.set_yticklabels([label_format.format(x) for x in ticks_loc_y])
    ax.set_ylabel('System size')
    ax.set_xlabel('$\\times 10^2$ Number of samples')

    ax.text(positions[k,0],
            positions[k,1],
            letters[k],
            fontweight='bold')

    cbar = fig.colorbar(pos,ax=ax,aspect=5,shrink=.6)
    cbar.ax.minorticks_off()

plt.savefig(folder+'figures/exp_1_magnetic_plot.pdf',format='pdf',pad_inches=0,bbox_inches='tight')
plt.show()