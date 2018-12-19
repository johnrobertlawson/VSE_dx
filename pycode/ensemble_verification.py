import os
import pdb
import datetime
import copy

import numpy as N
import matplotlib.pyplot as plt

from evac.datafiles.ensemble import Ensemble
from evac.plot.birdseye import BirdsEye
from evac.datafiles.radar import Radar
from evac.plot.scales import Scales
import evac.utils as utils

### SETTINGS ###
radardir = '/work/john.lawson/NEXRAD_data'
dataroot = '/scratch/john.lawson/WRF/VSE_reso/20160331'
initutc = datetime.datetime(2016,3,31,21,0,0)
outdir = '/home/john.lawson/VSE_reso/pyoutput/attempt1'
fhrs = [0.25,] + list(range(1,3))
plotlist = ['Verif',''] + list(range(1,11))

### OPTIONS ###
thumbnails = ['REFL_comp',]
scores = ['CRPS',]

### INSTANCES etc ###
E = Ensemble(dataroot,initutc,ctrl=False,loadobj=False,doms=2)
ST4 = StageIV(st4dir)
d01_limdict = E.get_limits(dom=1)
d02_limdict = E.get_limits(dom=2)
S = Scales('REFL_comp')

### FUNCTIONS ###
def plot_thumbnails(plotutc,vrbl):
    """ Plot data, including verification if it exists.

    There are three plotting options.
    The d01 and d02 domains, as is, and a third
    interpolation to common (inner) domain.
    """
    if vrbl == 'REFL_comp':
        R_large = Radar(plotutc,radardir)
        R_small = copy.copy(R_large)
        R_large.get_subdomain(**d01_limdict,overwrite=True)
        R_small.get_subdomain(**d02_limdict,overwrite=True)
    for nloop,dom in enumerate((1,1,2)):
        if nloop == 0:
            # This go round, do not zoom in on outer domain
            R = R_large
            ld = {}
            opt = "large"
        else:
            R = R_small
            ld = d02_limdict
            opt = "small"
        fhstr = utils.pretty_fhr(fhr)
        W = E.arbitrary_pick(dom=dom,dataobj=True)
        fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(10,8))
        # fname = "test_verif_d{:02d}_{}_{}".format(dom,opt)
        fname = "test_verif_d{:02d}_{}_{}".format(dom,opt,fhstr)
        lats,lons = E.get_latlons(dom=dom)
        nx,ny = E.get_nx_ny(dom=dom)
        cen_lat,cen_lon = E.get_cenlatlons(dom=dom)
        # xx,yy = E.get_xx_yy(dom=dom)
        for plot,ax in zip(plotlist,axes.flat):
            print("Plotting {}".format(plot))
            if plot == 'Verif':
                # first plot - observed radar
                R.plot_radar(fig=fig,ax=ax,drawcounties=True,cb=False)
                ax.set_title(plot)

            elif isinstance(plot,int):
                memname = 'm{:02d}'.format(plot)
                # members (plots 3 to 12)
                # if dom == 1:
                    # Plot only overlapping domain
                    # data = E.get(
                print("Getting data.")
                data = E.get('REFL_comp',utc=plotutc,dom=dom,members=memname)[0,0,0,:,:]
            
                print("Setting up axis.")
                # BE = BirdsEye(ax=ax,fig=fig,proj='lcc')
                BE = BirdsEye(ax=ax,fig=fig,proj='merc')
                print("Plotting on axis.")
                BE.plot2D(data,save=False,drawcounties=True,
                                # lats=lats,lons=lons,
                                clvs=S.clvs,cmap=S.cm,
                                # nx=nx,ny=ny,
                                W=W,cb=False,
                                # x=xx,y=yy,
                                cen_lat=cen_lat,cen_lon=cen_lon,**ld)

                print("Setting axis title.")
                title = 'Member {:02d}'.format(plot)
                ax.set_title(title)

            else: 
                ax.grid('off')
                ax.axis('off')

        fig.tight_layout()
        outfpath = os.path.join(outdir,fname)
        utils.trycreate(outfpath)
        fig.savefig(outfpath)
        print("Saving figure to",outfpath)

def plot_scores(plotutc):

### PROCEDURE ###
# Protect via name == main?

# dbz, rainfall, UH thumbnails
# maybe first five members of each, zoomed in

for fhr in fhrs:
    plotutc = initutc + datetime.timedelta(seconds=int(3600*fhr))
    for vrbl in thumbnails:
        plot_thumbnails(plotutc,vrbl)
    for score in scores:
        plot_scores(plotutc)

