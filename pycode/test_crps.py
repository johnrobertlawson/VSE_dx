"""
Benchmark found interpolation schemes made little difference to quality,
but substantial slowdown:

linear 90 s, cubic 97 s
"""
import os
import pdb
import datetime
import copy

import numpy as N
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from evac.stats.probscores import ProbScores
from evac.datafiles.ensemble import Ensemble
from evac.datafiles.stageiv import StageIV
import evac.utils as utils
from evac.utils.reproject_tools import reproject, WRF_native_grid, create_new_grid, VerifGrid
from evac.plot.map import Map
from evac.utils.misc_tools import time_me


dataroot = '/scratch/john.lawson/WRF/VSE_reso'
st4dir = '/work/john.lawson/STAGEIV_data'
outdir = '/home/john.lawson/VSE_reso/pyoutput'
crpsfpath = os.path.join(outdir,'crps_data.txt')

initutcs = [datetime.datetime(2016,3,31,x,0,0) for x in (19,20,21,22)]
# initutc = datetime.datetime(2016,3,31,21,0,0)
# itime = datetime.datetime(2016,3,31,22,0,0)
# ftime = datetime.datetime(2016,3,31,23,0,0)
vrbl = 'accum_precip'
nx = 50
ny = 50
# interp_method = 'cubic'
interp_method = 'linear'
plot_domains = False


def testplot(data,grid,fname,xx,yy,outdir,clvs=N.arange(1,26,1)):
    if xx.ndim == 1:
        xx, yy = N.meshgrid(xx,yy)
    mm = copy.copy(grid)
    if len(data.shape)== 3:
        assert data.shape[0] == 1
        data = data[0,...]
    f1 = mm.contourf(xx,yy,data,clvs)
    # f1 = mm.contourf(xx,yy,data,levels=N.arange(0.1,25,0.1))
    mm.drawstates()
    plt.colorbar(f1)
    fpath = os.path.join(outdir,fname)
    plt.gcf().savefig(fpath)
    plt.close(plt.gcf())
    print("Saved to",fpath)

    return

def create_newgrid(DF):
    xx = DF.xx
    yy = DF.yy
    lons = DF.lons
    lats = DF.lats
    
    m = DF.m
    return xx,yy,lats,lons,m


@time_me
def loop_through_domains(E,itime,ftime,initstr=None,doms='auto'):
    if doms == 'auto':
        doms = E.doms

    for dom in doms:
        W = WRF_native_grid(E.arbitrary_pick(give_path=True,dom=dom))

        # Generate neutral verification grid
        VG = VerifGrid(W,nx=nx,ny=ny)

        # Verification data reprojection
        print("Loading verification data")
        ST4 = StageIV(st4dir)
        xa = ST4.get(ftime)[0,0,:,:].data
        xa_xx_vg, xa_yy_vg = VG.m(ST4.lons,ST4.lats)
        pdb.set_trace()
        xa_vg = reproject(xa,xx_orig=xa_xx_vg,yy_orig=xa_yy_vg,xx_new=VG.xx,
                            yy_new=VG.yy,method=interp_method)

        # pdb.set_trace()
        # Forecast data reprojection
        xfs = E.get(vrbl,itime=itime,ftime=ftime,dom=dom)[:,0,0,:,:]
        xfs_vg = N.zeros([xfs.shape[0],ny,nx])
        xfs_xx_vg, xfs_yy_vg = VG.m(W.lons,W.lats)


        for ens in range(xfs.shape[0]):
            print("Reprojecting member #{}".format(ens+1))
            xfs_vg[ens,:,:] = reproject(xfs[ens,:,:],xx_orig=xfs_xx_vg,yy_orig=xfs_yy_vg,
                                xx_new=VG.xx,yy_new=VG.yy,method=interp_method)

        # print("Reprojecting onto coarsest grid")
        # xx,yy,lats,lons,newgrid = create_newgrid(ST4)

        W = E.arbitrary_pick(dataobj=True,dom=dom)
        W = WRF_native_grid(E.arbitrary_pick(give_path=True,dom=dom))
        # xfs_vg = N.zeros_like(xfs)
        # xx_new,yy_new = newgrid(W.lons,W.lats)
        # xx_old,yy_old = newgrid(W.lons,W.lats)
        # xx_new,yy_new = ST4.m(W.lons,W.lats)

        # for ens in range(xfs.shape[0]):
            # print("Reprojecting member #{}".format(ens+1))

        P = ProbScores(xfs=xfs_vg,xa=xa_vg)
        print("Computing CRPS")
        threshs = N.arange(0,100,1)
        crps = P.compute_crps(threshs)
        crps_str = "For domain {}, for {:02d}-{:02d} rainfall, CRPS = {:.4f}".format(
                                dom,itime.hour,ftime.hour,crps)
        print(crps_str)
        with open(crpsfpath,'a') as f:
            f.write(crps_str + '\n')

        testplot(xa,ST4.m,"StageIV_orig_d0{}.png".format(dom),ST4.xx,ST4.yy,outdir)
        testplot(xa_vg,VG.m,"StageIV_reproj_d0{}.png".format(dom),VG.xx,VG.yy,outdir)
        testplot(N.nanmax(xfs,axis=0),W.m,"ensmax_orig_d0{}.png".format(dom),W.xx,W.yy,outdir)
        testplot(N.nanmax(xfs_vg,axis=0),VG.m,"ensmax_reproj_d0{}.png".format(dom),VG.xx,VG.yy,outdir)

if __name__ == "__main__":
    # Forecast data and domain
    for initutc in initutcs:
        initstr = utils.string_from_time('title',initutc,)
        with open(crpsfpath,'a') as f:
            f.write('===== Init time: {} ===== \n'.format(initstr))
        initdir = utils.string_from_time('dir',initutc,strlen='hour')
        print("Loading forecast data for",initstr)

        datadir = os.path.join(dataroot,initdir)

        E = Ensemble(datadir,initutc,ndoms=2,ctrl=False)

        if plot_domains:
            W1 = WRF_native_grid(E.arbitrary_pick(give_path=True,dom=1))
            W2 = WRF_native_grid(E.arbitrary_pick(give_path=True,dom=2))
            VG1 = VerifGrid(W1,nx=nx,ny=ny)
            VG2 = VerifGrid(W2,nx=nx,ny=ny)
            Map().plot_domains(domains=(W1,W2,VG1,VG2),labels=('3km','1km','Outer verif','Inner verif'),latlons='auto',
                                outdir=outdir,)
        itimes = [initutc + datetime.timedelta(seconds=3600*n) for n in range(3)]
        for itime in itimes:
            ftime = itime + datetime.timedelta(seconds=3600)
            loop_through_domains(E,itime=itime,ftime=ftime,initstr=initstr)
