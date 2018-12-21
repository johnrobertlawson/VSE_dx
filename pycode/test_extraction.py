import pdb
import os

import matplotlib.pyplot as plt
import numpy as N


# First, check lat/lon boxes

froot = '/work/john.lawson/VSE_reso/pp/Xmas/20160331'

f1 = 'REFL_comp_d01_3km_20160331_2200_2215_m01.npy'
# f2 = 'REFL_comp_d01_5km_20160331_2200_2215_m01.npy'
f3 = 'REFL_comp_d02_1km_20160331_2200_2215_m01.npy'
f4 = 'REFL_comp_d02_3km_20160331_2200_2215_m01.npy'
# f5 = 'REFL_comp_d02_5km_20160331_2200_2215_m01.npy'
f6 = 'NEXRAD_nexrad_1km_20160331_2215.npy'
f7 = 'NEXRAD_nexrad_3km_20160331_2215.npy'

PLOTS = {
        f1:"d01_3km",
        # f2:"d01_5km",
        f3:"d02_1km",
        f4:"d02_3km",
        # f5:"d02_5km",
        f6:"nex_1km",
        f7:"nex_3km",
        }

fig,axes = plt.subplots(ncols=2,nrows=3)

for ax, f in zip(axes.flat,(f1,f3,f4,f6,f7)):
    fpath = os.path.join(froot,f)
    ax.imshow(N.load(fpath))
    ax.set_title(PLOTS[f])

fpath = "/home/john.lawson/VSE_dx/pyoutput/check_domains.png"
fig.savefig(fpath)
