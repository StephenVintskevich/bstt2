import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
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

res1 = np.load("10ptcls_gp4-8ev_int5_smpls4-10_nostop.data.npy") #order x interaction x trainSampleSize (5,3,7)
res2 = np.load("10ptcls_gp4-8ev_int5_smpls1-3_nostop.data.npy")
res1 = np.array(res1[0,:,0,:,0,:])
res2 = np.array(res2[0,:,0,:,0,:])
maxGroupSizeLen = len(res1[0,:,0])
noSamplesLen = len(res1[:,0,0])+len(res2[:,0,0])
repsLen = len(res1[0,0,:])
res = np.zeros((maxGroupSizeLen, noSamplesLen, repsLen))
for i in range(maxGroupSizeLen):
    res[i,:,:] = np.concatenate((res2[:,i,:],res1[:,i,:]))
print(res.shape)

means = np.mean(res,axis=2)

t = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1]
label_format = '{:,.0f}'
ticks_loc_x = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
ticks_loc_x = range(1,11)
ticks_loc_y =[4,6,8]
interaction = [5]
colours = ['tab:blue','tab:orange','tab:green']
linestyles = [':','--','-.']

cmtoinch = 2.54
fig = plt.figure(figsize=(8.5/cmtoinch,5/cmtoinch))

ax = plt.gca()
lines = []
for i in range(maxGroupSizeLen):
    ax.semilogy(ticks_loc_x,res[i,:,:],'+',ms=7,mew=1,color=colours[i],alpha=.35)
    ax.semilogy(ticks_loc_x,means[i,:],'x',ms=7,mew=2,color=colours[i],alpha=1.,ls=linestyles[i],label=f"$\\rho = {ticks_loc_y[i]}$")

ax.legend(frameon=False)

ax.set_ylabel('Residuum')
ax.set_xlabel('$\\times 10^3$ Number of samples')
ax.tick_params(direction="in")
ax.minorticks_off()

plt.savefig('figures/exp_1_lennard_jones_mod.pdf',format='pdf',bbox_inches='tight')
plt.show()
