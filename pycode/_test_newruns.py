import os
import pdb
import itertools
import datetime

import matplotlib.pyplot as plt
import numpy as N

import evac.utils.utils as utils
from evac.plot.scales import Scales
from evac.datafiles.wrfout import WRFOut


##### SETTINGS #####
outdir = '/home/john.lawson/VSE_reso/pyoutput/tests'

MEMS = ['m0{}'.format(n) for n in range(1,5)]

FOLDERS = (
            '/scratch/john.lawson/WRF/VSE_reso/ForReal/2017050302',
            '/scratch/john.lawson/WRF/VSE_reso/ForReal_netcdf4/2017050302',
            '/scratch/john.lawson/WRF/VSE_reso/ForReal_nco/2017050302',
            )

DOMS = (
        'd01',
        'd02',
        )

fcstmins = range(30,210,30)

# vrbl = 'REFL_comp'
VRBLS = (
        'REFL_comp',
        'WSPD10MAX',
        # 'QVAPOR',
        'W',
        )

plotutc = datetime.datetime(2017,5,3,2,0,0)

##### PROCEDURE #####

for dom in DOMS:
    for fcstmin, vrbl in itertools.product(fcstmins,VRBLS):
        utc = plotutc + datetime.timedelta(seconds=(60*fcstmin))
        fig,axes = plt.subplots(ncols=4,nrows=3,figsize=(10,8))

        axit = iter(axes.flatten())
        for folder in sorted(FOLDERS):
            if "netcdf4" in folder:
                ncno = 4
            elif "nco" in folder:
                ncno = " ncrcat"
            else:
                ncno = 3
            for mem in sorted(MEMS):
                ax = next(axit)
                ss = folder.split('/')[-1]
                fname = 'wrfout_{}_2017-05-03_02:00:00'.format(dom)

                lvidx = 0
                if vrbl == 'REFL_comp':
                    S = Scales(vrbl=vrbl)
                    cmap = S.cm
                    lvs = S.clvs
                elif vrbl == 'WSPD10MAX':
                    cmap = 'gist_earth_r'
                    lvs = N.arange(2.5,22.5,2.5)
                elif vrbl == 'W':
                    lvidx = 20
                else:
                    raise Exception

                fpath = os.path.join(folder,mem,fname)
                # ax.set_title("Dom {} for the {} run".format(dom,ss),fontsize=6)
                ax.set_title("{} for netCDF{}".format(mem,ncno))

                if ("onedom" in ss) and (dom == 'd02'):
                    continue
                W = WRFOut(fpath)
                data = W.get(vrbl=vrbl,utc=utc)[0,lvidx,:,:]

            
                ax.contourf(data,cmap=cmap,levels=lvs)
                ax.grid('off')
                ax.axis('off')

        fname = "NCOruns_{:03d}min_{}_{}.png".format(fcstmin,vrbl,dom)
        fpath = os.path.join(outdir,fname)
        utils.trycreate(fpath)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        print("Saved to",fpath)

