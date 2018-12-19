import os
import math
import datetime
import pdb

import matplotlib as M
import numpy as N
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors.kde import KernelDensity

# from statsmodels.api.nonparametric import KDEUnivariate
import statsmodels.api as smapi

from evac.datafiles.ensemble import Ensemble
import evac.utils.utils as utils

scipy_method = 0
seaborn_method = 0
statsmodels_method = 1

ensroot = '/scratch/john.lawson/WRF/VSE_reso/ensembledata'
outroot = '/home/john.lawson/VSE_reso/pyoutput/'

initutc = datetime.datetime(2016,3,31,20,0,0)
initstr = '2016033120'
fhr = 1
vrbl = 'accum_precip'; thresh = 10.0
# vrbl = 'T2'; thresh = 26.0

# Get raw data and obs
enspath = os.path.join(ensroot,initstr)
E = Ensemble(rootdir=enspath,initutc=initutc,ndoms=2,ctrl=False,
                allow_empty=False)
fcst_all = E.get(fcsthr=fhr,vrbl=vrbl,dom=1)

if vrbl == 'T2':
    fcst_all = fcst_all - 273.15

# Set up
nmems,*_,nlat,nlon = fcst_all.shape
nflat = nlat*nlon
fcst = fcst_all[:,0,0,:,:].reshape(nmems,nflat)
# fcst_swap = N.swapaxes(fcst,0,1)
# bw = 0.2
fcst = N.swapaxes(fcst,1,0)
xx,yy = N.meshgrid(N.arange(0,nlat),N.arange(0,nlon))

# pdb.set_trace()

def get_bw(arr,factor=1/5,remove_std=False):
    # have to remove std as it's included in code
    bw = (4/3)**(1/5) * N.std(arr) * arr.size**(-1/5)
    # bw = ((4/3)**(1/5)) * (arr.size**(-1/5))


    if remove_std:
        bw = bw/N.std(arr)
    # have to divide by 10 
    bw = bw * factor
    return bw

######### Scipy gaussian_kde #########
if scipy_method:
    # First, 1D
    # fcst = fcst_swap
    # pdb.set_trace()
    fcst1D = fcst[3556,:]
    bw = get_bw(fcst1D)
    kde = gaussian_kde(dataset=fcst1D,bw_method=bw)
    xmin = math.floor(fcst1D.min())
    xmax = math.ceil(fcst1D.max())
    xpoints = N.arange(xmin,xmax+0.05,0.05)
    y = kde.evaluate(xpoints)


    # rawpc = (len(fcst[fcst > 24,0])/nmems)*100
    rawpc = (len(fcst1D[fcst1D > thresh])/nmems)*100
    kdepc = (kde.integrate_box_1d(thresh,xmax))*100

    print("Chance of exceeding {} C:\n raw:\t{}% \n kde:\t{}%".format(
                    thresh,rawpc,kdepc))

    fig,ax = plt.subplots()
    ax.plot(xpoints,y,color='black',lw=1)
    ax.fill_between(xpoints,y,color='lightgray')
    ax.scatter(fcst1D,[0,]*len(fcst1D),marker='x',color='red',zorder=100)

    ax.axvline(thresh,color='k')

    fname = 'test_kde_scipy.png'
    fpath = os.path.join(outroot,fname)
    fig.savefig(fpath)

    ######## 2D #########
    fcst = fcst_all[:,0,0,:,:]
    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(8,5))
    levels = N.arange(5,105,5)

    ax = axes.flat[0]
    rawpc = utils.exceed_probs_2d(fcst,thresh)
    cf = ax.contourf(xx,yy,rawpc,cmap=M.cm.jet,levels=levels)
    ax.set_aspect('equal')

    # kde2d = gaussian_kde(dataset=fcst,bw_method=bw)
    # positions = N.vstack(xx.ravel(),yy.ravel())
    # kdepc_flat = kde.evaluate(positions)
    # kdepc = N.reshape(kdepc_flat.T,xx.shape)

    fcst_pp = N.empty([nlat,nlon])
    for i in range(nlat):
        if i%20 == 0:
            print("Looping...")
        for j in range(nlon):
            fcst1D = fcst[:,i,j]
            if N.sum(fcst1D) == 0.0:
                prob = 0.0
            else:
                # bw = (4/3)**(1/5) * fcst1D.std * nmems**(-1/5)
                bw = get_bw(fcst1D)
                kde = gaussian_kde(dataset=fcst1D,bw_method=bw)
                # y = kde.evaluate(xpoints)
                prob = kde.integrate_box_1d(thresh,xmax)*100
            fcst_pp[i,j] = prob
            

    ax = axes.flat[1]
    cf = ax.contourf(xx,yy,fcst_pp,cmap=M.cm.jet,levels=levels)
    ax.set_aspect('equal')

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.15,0.087,0.7,0.025])
    fig.colorbar(cf,cax=cbar_ax,orientation='horizontal')

    fname = 'test_kde2d_scipy.png'
    fpath = os.path.join(outroot,fname)
    fig.savefig(fpath)

# statsmodels
if statsmodels_method:
    do_1d = 0
    do_2d = 1
    fcst1D = fcst[3556,:]
    fcst1Da = N.hstack((fcst1D,N.array([0.0,0.0,0.0])))

    if do_1d:
        for bwfc in (1,3,7,10):
            bw = get_bw(fcst1D,factor=bwfc)

            # kde = statsmodels.api.nonparametric.KDEUnivariate(fcst1Da)
            kde = smapi.nonparametric.KDEUnivariate(fcst1Da)

            for plottype in ('cutmod','cut','clip','default'):
                if plottype == 'cutmod':
                    kde.fit(bw=bw,cut=(0,3))
                elif plottype == 'cut':
                    kde.fit(bw=bw,cut=0)
                elif plottype == 'clip':
                    kde.fit(bw=bw,clip=(0,N.inf))
                elif plottype == 'default':
                    kde.fit(bw=bw)
            # pdb.set_trace()

                nmems = fcst1Da.size
                rawpc = (len(fcst1Da[fcst1Da > thresh])/nmems)*100
                x = kde.support
                idx = utils.closest(x,thresh)
                kdepc = (1-kde.cdf[idx])*100

                print("Chance of exceeding {} :\n raw:\t{}% \n kde:\t{}%".format(
                            thresh,rawpc,kdepc))

                fig,ax = plt.subplots()
                ax.plot(x,kde.density,color='black',lw=1)
                ax.fill_between(x,kde.density,color='lightgray')
                ax.plot(x,kde.cdf,color='red',lw=1)
                ax.scatter(fcst1Da,[-0.015,]*len(fcst1Da),marker='x',color='red',zorder=100)

                ax.axvline(thresh,color='k')

                ax.set_xlim([-1,8])
                ax.set_ylim([-0.03,1.03])

                fname = 'test_kde_sm_{}_bwfc{}.png'.format(plottype,bwfc)
                fpath = os.path.join(outroot,'kde',fname)
                utils.trycreate(fpath)
                fig.savefig(fpath)

                # bwfc = 7 looks best
                # cut = (0,3) looks best

    ######## 2D #########
    if do_2d:
        fcst = fcst_all[:,0,0,:,:]
        fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(8,5))
        levels = N.arange(2,102,2)

        ax = axes.flat[0]
        rawpc = utils.exceed_probs_2d(fcst,thresh)
        cf = ax.contourf(xx,yy,rawpc,cmap=M.cm.jet,levels=levels)
        ax.set_aspect('equal')

        from multiprocessing import Pool

        def get_prob(itr,method=2):
            i, j, arr = itr
            if N.sum(arr) == 0.0:
                prob = 0.0
            else:
                if method == 1:
                    bw = get_bw(arr,factor=1/7)
                    kde = smapi.nonparametric.KDEUnivariate(arr)
                    kde.fit(bw=bw,cut=(0,3))
                    x = kde.support
                    idx = utils.closest(x,thresh)
                    prob = (1-kde.cdf[idx])*100
                elif method == 2:
                    bw = get_bw(arr,factor=3,remove_std=True)
                    kde = gaussian_kde(dataset=arr,bw_method=bw)
                    # xmax = arr.max()
                    prob = kde.integrate_box_1d(thresh,1000)*100
            # assert 0 <= prob <= 100
            prob = min(100.0,prob)
            prob = max(0.0,prob)
            return i,j,prob

        def itr_fcst(fcst,nlat,nlon):
            for i in range(nlat):
                for j in range(nlon):
                    yield i, j, fcst[:,i,j]
            

        fcst_pp = N.empty([nlat,nlon])
        itr = itr_fcst(fcst=fcst,nlat=nlat,nlon=nlon)
        with Pool(20) as pool:
            results = pool.map(get_prob,itr)
        # pdb.set_trace()
        # fcst_pp[i,j] = prob
        result_arr = N.array(results)
        fcst_pp[:,:] = result_arr[:,2].reshape(nlat,nlon)
                

        ax = axes.flat[1]
        cf = ax.contourf(xx,yy,fcst_pp,cmap=M.cm.jet,levels=levels)
        ax.set_aspect('equal')

        fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes([0.15,0.087,0.7,0.025])
        fig.colorbar(cf,cax=cbar_ax,orientation='horizontal')

        fname = 'test_kde2d_sm.png'
        fpath = os.path.join(outroot,'kde',fname)
        fig.savefig(fpath)


# seaborn
if seaborn_method:
    import seaborn as sns
    # First, 1D
    fcst1D = fcst[3556,:]
    bw = get_bw(fcst1D)
    ax0 = sns.kdeplot(fcst1D,cut=0,cumulative=False,bw=bw)
    fname0 = 'test_kde_seaborn0.png'
    fpath0 = os.path.join(outroot,fname0)
    ax0.get_figure().savefig(fpath0)

    ax1 = sns.kdeplot(fcst1D,cumulative=True,bw=bw)
    fname1 = 'test_kde_seaborn1.png'
    fpath1 = os.path.join(outroot,fname1)
    ax1.get_figure().savefig(fpath1)

    ax2 = sns.kdeplot(fcst1D,clip=(0,N.inf),bw=bw)
    fname2 = 'test_kde_seaborn2.png'
    fpath2 = os.path.join(outroot,fname2)
    ax2.get_figure().savefig(fpath2)

    fcst1Da = N.hstack((fcst1D,N.array([0.0,0.0,0.0])))
    # ax3 = sns.kdeplot(fcst1Da,clip=(-0.0001,N.inf),bw=bw)
    ax3 = sns.kdeplot(fcst1Da,cut=0,bw=bw)
    fname3 = 'test_kde_seaborn3.png'
    fpath3 = os.path.join(outroot,fname3)
    ax3.get_figure().savefig(fpath3)

    pdb.set_trace()
    # rawpc = (len(fcst[fcst > 24,0])/nmems)*100
    rawpc = (len(fcst1D[fcst1D > thresh])/nmems)*100
    kdepc = (kde.integrate_box_1d(thresh,xmax))*100

    print("Chance of exceeding {} C:\n raw:\t{}% \n kde:\t{}%".format(
                    thresh,rawpc,kdepc))

    fig,ax = plt.subplots()
    ax.plot(xpoints,y,color='black',lw=1)
    ax.fill_between(xpoints,y,color='lightgray')
    ax.scatter(fcst1D,[0,]*len(fcst1D),marker='x',color='red',zorder=100)

    ax.axvline(thresh,color='k')

    fname = 'test_kde_scipy.png'
    fpath = os.path.join(outroot,fname)
    fig.savefig(fpath)

    ######## 2D #########
    fcst = fcst_all[:,0,0,:,:]
    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(8,5))
    levels = N.arange(5,105,5)

    ax = axes.flat[0]
    rawpc = utils.exceed_probs_2d(fcst,thresh)
    cf = ax.contourf(xx,yy,rawpc,cmap=M.cm.jet,levels=levels)
    ax.set_aspect('equal')

    # kde2d = gaussian_kde(dataset=fcst,bw_method=bw)
    # positions = N.vstack(xx.ravel(),yy.ravel())
    # kdepc_flat = kde.evaluate(positions)
    # kdepc = N.reshape(kdepc_flat.T,xx.shape)

    fcst_pp = N.empty([nlat,nlon])
    for i in range(nlat):
        if i%20 == 0:
            print("Looping...")
        for j in range(nlon):
            fcst1D = fcst[:,i,j]
            if N.sum(fcst1D) == 0.0:
                prob = 0.0
            else:
                # bw = (4/3)**(1/5) * fcst1D.std * nmems**(-1/5)
                bw = get_bw(fcst1D)
                kde = gaussian_kde(dataset=fcst1D,bw_method=bw)
                # y = kde.evaluate(xpoints)
                prob = kde.integrate_box_1d(thresh,xmax)*100
            fcst_pp[i,j] = prob
            

    ax = axes.flat[1]
    cf = ax.contourf(xx,yy,fcst_pp,cmap=M.cm.jet,levels=levels)
    ax.set_aspect('equal')

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.15,0.087,0.7,0.025])
    fig.colorbar(cf,cax=cbar_ax,orientation='horizontal')

    fname = 'test_kde2d_scipy.png'
    fpath = os.path.join(outroot,fname)
    fig.savefig(fpath)


    # sklearn 1D KernelDensity


    # statsmodels 2D KDEMultivariate



