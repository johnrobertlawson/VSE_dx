""" Figure generation for Lawson et al, 2019.

d01 (3 km):
    * Cut/native (3 km)
    * Cut/interpolated to 5-km (neutral; for StageIV)
d02 (1 km):
    * Native (1 km)
    * Interpolated to 3-km (d01 domain - points must be synced)
    * Interpolated to 5-km (neutral; for StageIV)
"""
import pdb
import logging
import os
import datetime
import collections
import random
import pickle 
import itertools
import multiprocessing
import argparse
import netCDF4

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap

from evac.stats.fi import FI
from evac.stats.casati import Casati
from evac.plot.histogram import Histogram
from evac.plot.efss_intensity import EFSS_Intensity
from evac.plot.salgraph import SALGraph
from evac.plot.boxplot import BoxPlot
from evac.stats.forecastvalue import ForecastValue
from evac.plot.linegraph import LineGraph
from evac.plot.birdseye import BirdsEye
from evac.plot.performance import Performance
from evac.datafiles.radar import Radar
from evac.stats.verif import Verif
from evac.datafiles.ensemble import Ensemble
from evac.datafiles.obsgroup import ObsGroup
from evac.stats.detscores import DetScores
from evac.stats.probscores import ProbScores
from evac.stats.objectbased import ObjectBased
from evac.plot.scales import Scales
from evac.plot.thumbnails import ThumbNails
from evac.stats.esal import ESAL
from evac.stats.efss import EFSS
from evac.utils.grid import Grid
from evac.plot.areafill import AreaFill
from evac.datafiles.wrfout import WRFOut
import evac.utils as utils

##### ARGUMENT PARSING #####

parser = argparse.ArgumentParser()

parser.add_argument('-N','--ncpus',dest='ncpus',default=1,type=int)
ncpus = parser.parse_args().ncpus

# This will re-run cut/interp methods
parser.add_argument('-oe','--overwrite_extraction',action='store_true',default=False)
overwrite_extraction = parser.parse_args().overwrite_extraction

# This will re-run scores
parser.add_argument('-or','--overwrite_scores',action='store_true',default=False)
overwrite_scores = parser.parse_args().overwrite_scores

# This will re-run plots
parser.add_argument('-oo','--overwrite_output',action='store_true',default=False)
overwrite_output = parser.parse_args().overwrite_output

parser.add_argument('-T','--tests',action='store_true',default=False)
basic_tests = parser.parse_args().tests
parser.add_argument('-C','--check_only',action='store_true',default=False)
check_only = parser.parse_args().check_only
parser.add_argument('-S','--subcpus',dest='subcpus',default=1,type=int)
subcpus = parser.parse_args().subcpus

##### ERROR LOGGING #####

mp_log = multiprocessing.log_to_stderr()
mp_log.setLevel(logging.INFO)

### SETTINGS ###

# Folder key for final experimental wrfouts
key_wrf = 'ForReal_nco'
# Folder key for post-processed fields (objects, lat/lon, etc)
key_pp = 'Xmas'
# Folder key for scores (FSS, etc)
key_scores = 'Xmas'
# Folder key for plots
key_plot = 'Xmas'

ensroot = '/scratch/john.lawson/WRF/VSE_reso/{}'.format(key_wrf)
reproj_obs_root = '/work/john.lawson/VSE_reso/reproj_obs/{}'.format(key_pp)
reproj_fcst_root = '/work/john.lawson/VSE_reso/reproj_fcst/{}'.format(key_pp)
reproj_grid_root = '/work/john.lawson/VSE_reso/reproj_grid/{}'.format(key_pp) 
npyroot = '/work/john.lawson/VSE_reso/scores/{}'.format(key_scores)
outroot = '/home/john.lawson/VSE_reso/pyoutput/{}'.format(key_plot)

st4dir = '/work/john.lawson/STAGEIV_data'
mrmsdir = 'work/john.lawson/MRMS_data/VSE_reso'


CASES = collections.OrderedDict()
CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                        datetime.datetime(2016,3,31,19,0,0),
                        datetime.datetime(2016,3,31,20,0,0),
                        datetime.datetime(2016,3,31,21,0,0),
                        datetime.datetime(2016,3,31,22,0,0),
                        # BROKEN
                        datetime.datetime(2016,3,31,23,0,0),  
                        ]
CASES[datetime.datetime(2017,5,1,0,0,0)] = [
                ObsGroup(radardir,'radar')
    elif vrbl == 'accum_precip':
        OBS = ST4
    else:
        raise Exception

    # LOAD DATA
    E = Ensemble(ensdir,initutc,ndoms=2,ctrl=False,allow_empty=False)
    NEWGRID = try_load_newgrid(initstr=initstr,E=E)
    fcst_grid = NEWGRID
    obs_grid = NEWGRID

    if vrbl == 'accum_precip':
        all_fcstmins = (60,120,180)
    elif vrbl == 'REFL_comp':
        all_fcstmins_dt = [v-E.initutc for v in E.validtimes]
        all_fcstmins = [int(t.seconds/60) for t in all_fcstmins_dt]
    else:
        raise Exception

    nfcstmin = len(all_fcstmins)
    for n,fcstmin in enumerate(all_fcstmins):
        utc = get_valid_time(initutc,fcstmin=fcstmin)
        xfs2 = try_reproj_fcst(initstr=initstr,vrbl=vrbl,fcstmin=fcstmin,dom_pp=dom_pp,
                    E=E,grid=NEWGRID,ensmember='all')
        xa2 = try_reproj_obs(dom_pp,obs=OBS,utc=utc,grid=NEWGRID)

        xfs = xfs2[:,1:-1,1:-1]
        xa = xa2[1:-1,1:-1]
        # pdb.set_trace()
        if n == 0:
            xfs_all = N.zeros([nfcstmin,*xfs.shape])
            xa_all = N.zeros([nfcstmin,*xa.shape])
        xfs_all[n,...] = xfs
        xa_all[n,...] = xa

        # pdb.set_trace()

    # threshs = (0.5,1,2,4,8,16,32)
    if vrbl in ("REFL_comp",):
        threshs = N.arange(0,100,5)
        temporal_windows = (1,3,5)
    elif vrbl == 'accum_precip':
        threshs = N.arange(2.5,37.5,2.5)
        temporal_windows = (1,)

    maxsize = max(xfs_all.shape[2:])
    spatial_windows = N.arange(1,maxsize,5)
    # Assuming every 15 min for radar data?

    fcst4d = N.swapaxes(xfs_all,0,1)

    efss = EFSS(fcst4d=fcst4d,obs3d=xa_all,threshs=threshs,ncpus=40,
                spatial_ns=spatial_windows,temporal_ms=temporal_windows)

    # pdb.set_trace()

    fdir_efss = npydir
    fpath = generate_fname(vrbl=vrbl,metric='eFSS',
                    fcstmin=0,dom_pp=dom_pp,ensmember='all',
                    fdir=fdir_efss,extension='.pickle')
    trysave(fpath=fpath,data=efss)
    print("Saved EFSS class to",fpath)

def plot(itr):
    itr = list(itr)
    thresh = 10

    # _vrbl, _caseutc, _initutc, _fcstmin, _dom_pp = itr

    if plot_fi:
        plot_indiv = True
        # plot_indiv = False
        vrbl = "REFL_comp"
        dBZs = (10,30,50)
        # dBZs = N.arange(10,70,10)
        # If we do raw domains, not reproj, then 3 and 1!
        thedoms = {'d01_reproj':5,'d02_reproj':5}
        domkeys = sorted(thedoms.keys())
        scores = ("FI","RES","REL","UNC","FISS","RES_REL","FISS_WoF","eFSS","FISS2")
        ninit = len(initutcs)
        neighs = (1,3,5,7,9,19)
        nneighs = len(neighs)

        # DATA = {dom: {dBZ: {score: N.zeros([ninit,4]) for score in scores} for dBZ in dBZs} for dom in thedoms.keys()}
        DATA = {s: N.zeros([2, len(dBZs), nneighs, len(fcstmins),len(initutcs)]) for s in scores}
                    # '-' if score == "FI" else '--'
        LS = dict(FI='-',FISS='-.',RES='--',REL='--',UNC='--',FISS2='-.',RES_REL=':',FISS_WoF='-.',eFSS='-')
        assert sorted(LS.keys()) == sorted(scores)
        all_neighs = []
        for dom_pp, initutc, fcstmin, score in itertools.product(domkeys,initutcs,fcstmins,scores):
            # if score in ("RES_REL","FISS_WoF"):
                # continue
            domidx = domkeys.index(dom_pp)
            fidx = fcstmins.index(fcstmin)
            iidx = initutcs.index(initutc)
            dx = thedoms[dom_pp]
            initstr = utils.string_from_time('dir',initutc,strlen='hour')
            fi_fpath = os.path.join(outroot,initstr,'FI_{}_{}min_{}.png'.format(dom_pp,fcstmin,score))

            fdir_fi = os.path.join(npyroot,initstr,"FI")
            fpath = generate_fname(vrbl=vrbl,metric='FI',suffix="efss",
                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember='all',
                        fdir=fdir_fi,extension='.pickle')
            
            with open(fpath,'rb') as f:
                fi = pickle.load(f)


            toplot = {}
            # dBZs = sorted(fi.keys())
            neighs = sorted(fi[dBZs[0]].keys())
            all_neighs.append(neighs)
            # if dom_pp.startswith("d02"):
                # pdb.set_trace()

            for dbzidx,dBZ in enumerate(dBZs):
                toplot[dBZ] = []
                for nidx,neigh in enumerate(neighs):
                    if score == "FISS2":
                    # if score == "FISS":
                        s = (fi[dBZ][neigh][1][0]["RES"] - fi[dBZ][neigh][1][0]["REL"])/(
                                fi[dBZ][neigh][1][0]["UNC"])
                    elif score == "FISS_WoF":
                    # if score == "FISS_WoF":
                        s = (0.67*fi[dBZ][neigh][1][0]["RES"] - 0.33*fi[dBZ][neigh][1][0]["REL"])/(
                                fi[dBZ][neigh][1][0]["UNC"])
                    elif score == "RES_REL":
                        s = (fi[dBZ][neigh][1][0]["RES"] - fi[dBZ][neigh][1][0]["REL"])

                    else:
                        s = fi[dBZ][neigh][1][0][score]
                    toplot[dBZ].append(s)
                    DATA[score][domidx,dbzidx,nidx,fidx,iidx] = s

            if plot_indiv:
                with LineGraph(fpath=fi_fpath) as LG:
                    # fig, ax = plt.subplots(1)
                    for dBZ in dBZs:
                        # ax.plot(neighs,toplot[dBZ],label="{} dBZ".format(dBZ))
                        linestyle = LS[score]
                        pkw = dict(label="{} dBZ".format(dBZ),linestyle=linestyle)
                        LG.plot(neighs,toplot[dBZ],plotkwargs=pkw)
                    LG.ax.legend()


                    if "FISS" in score: # in ('FISS','FISS_WoF'):
                        LG.ax.set_ylim([-1,1])
                        LG.ax.axhline(y=0.0,color='k')
                        LG.ax.set_ylabel("Fractional Ignorance Skill Score")
                    elif score in ("RES_REL"):
                        LG.ax.set_ylim([-0.5,0.5])
                        LG.ax.axhline(y=0.0,color='k')
                        LG.ax.set_ylabel("Resolution - Reliability (bits)")
                    elif score in ("eFSS"):
                        LG.ax.set_ylim([0,1.01])
                        # LG.ax.axhline(y=1.0,color='k')
                        LG.ax.set_ylabel("Extended Fractions Skill Score")
                    else:
                        LG.ax.set_ylim([0,1.25])
                        LG.ax.axhline(y=1.0,color='k')
                        LG.ax.set_ylabel("Fractional Ignorance: {} component (bits)".format(
                                                score))

                    # if score in ('FISS',):
                        # LG.ax.set_ylim([-1,1])
                        # LG.ax.axhline(y=0.0,color='k')
                    # else:
                        # LG.ax.set_ylim([0,1.25])
                        # LG.ax.set_ylim([0,2])
                        # LG.ax.axhline(y=1.0,color='k')
                    LG.ax.set_xticks(neighs)
                    LG.ax.set_xticklabels((0.5*dx*N.array(neighs)))
                    LG.ax.set_xlabel("Neighbourhood radius (km)")
                    # ax.set_ylabel("FI and components (bits)")
                    # LG.ax.set_ylabel("Fractional Ignorance (bits)")

        # Plotting over all cases
        # nneigh = len(neighs)
        # ALL_DATA = {dom: {d: {} for d in dBZs} for dom in thedoms.keys()}
        DIFFS = {d: {} for d in dBZs}
        AVES = {dom: {d: {} for d in dBZs} for dom in domkeys}
        # for dom_pp, dBZ, score in itertools.product(thedoms.keys(),dBZs,scores):
        for dBZ, score in itertools.product(dBZs,scores):
            if score in ("FISS_WoF","RES_REL"):
                continue
            dbzidx = list(dBZs).index(dBZ)
            # ALL_DATA[dom_pp][dBZ][score] = N.mean(DATA[dom_pp][dBZ][score],axis=0)
            # if dom_pp == "d01_reproj": 
            # DIFFS[dBZ][score] = ALL_DATA[dom_pp][dBZ][score] - ALL_DATA["d01_reproj"][dBZ][score]
            for domidx,dom in enumerate(domkeys):
                AVES[dom][dBZ][score] = N.nanmean(DATA[score][domidx,dbzidx,:,:,:],axis=2)
            # DIFFS[dBZ][score] = (N.nanmean(DATA[score][1,dbzidx,:,:,:],axis=2) -
                                    # N.nanmean(DATA[score][0,dbzidx,:,:,:],axis=2))
            DIFFS[dBZ][score] = AVES["d02_reproj"][dBZ][score] - AVES["d01_reproj"][dBZ][score]

        # Separate FISS calc
        for dBZ in dBZs:
            for domidx,dom in enumerate(domkeys):
                # AVES[dom][dBZ]['FISS'] = ((AVES[dom][dBZ]['RES'] - AVES[dom][dBZ]['REL'])/
                                            # AVES[dom][dBZ]['UNC'])
                AVES[dom][dBZ]['FISS_WoF'] = (0.67*(AVES[dom][dBZ]['RES'] - 0.33*AVES[dom][dBZ]['REL'])/
                                            AVES[dom][dBZ]['UNC'])
                AVES[dom][dBZ]['RES_REL'] = AVES[dom][dBZ]['RES'] - AVES[dom][dBZ]['REL']
            # DIFFS[dBZ]["FISS"] = AVES["d02_reproj"][dBZ]['FISS'] - AVES["d01_reproj"][dBZ]['FISS']
            DIFFS[dBZ]["FISS_WoF"] = AVES["d02_reproj"][dBZ]['FISS_WoF'] - AVES["d01_reproj"][dBZ]['FISS_WoF']
            DIFFS[dBZ]["RES_REL"] =  AVES["d02_reproj"][dBZ]['RES_REL'] - AVES["d01_reproj"][dBZ]['RES_REL']

                                    

        print("About to plot.")
        prods = ("diff","d01_ave","d02_ave")
        for score,prod in itertools.product(scores,prods):
            for fidx, fcstmin in enumerate(fcstmins):
                fi_fpath = os.path.join(outroot,'FI_{}_{}min_{}.png'.format(prod,fcstmin,score))
                with LineGraph(fpath=fi_fpath) as LG:
                    for dBZ in dBZs:
                        # ax.plot(neighs,toplot[dBZ],label="{} dBZ".format(dBZ))
                        linestyle = LS[score]
                        pkw = dict(label="{} dBZ".format(dBZ),linestyle=linestyle)
                        # pdb.set_trace()
                        if prod == 'diff':
                            LG.plot(neighs,DIFFS[dBZ][score][:,fidx],plotkwargs=pkw)
                        elif prod == 'd01_ave':
                            LG.plot(neighs,AVES['d01_reproj'][dBZ][score][:,fidx],plotkwargs=pkw)
                        elif prod == 'd02_ave':
                            LG.plot(neighs,AVES['d02_reproj'][dBZ][score][:,fidx],plotkwargs=pkw)
                        else:
                            raise Exception
                            
                    LG.ax.legend()
                    if "FISS" in score: # in ('FISS','FISS_WoF'):
                        if prod == 'diff':
                            LG.ax.set_ylim([-0.2,0.2])
                        else:
                            LG.ax.set_ylim([-1,1])
                        LG.ax.axhline(y=0.0,color='k')
                        LG.ax.set_ylabel("Fractional Ignorance Skill Score")
                    elif score in ("RES_REL"):
                        if prod == 'diff':
                            LG.ax.set_ylim([-0.15,0.15])
                        else:
                            LG.ax.set_ylim([-0.5,0.5])
                        LG.ax.axhline(y=0.0,color='k')
                        LG.ax.set_ylabel("Resolution - Reliability (bits)")
                    elif score in ("eFSS"):
                        if prod == 'diff':
                            LG.ax.set_ylim([-0.1,0.1])
                            LG.ax.axhline(y=0.0,color='k')
                        else:
                            LG.ax.set_ylim([0,1.01])
                        LG.ax.set_ylabel("Extended Fractions Skill Score")
                        
                    else:
                        if prod == 'diff':
                            LG.ax.set_ylim([-0.15,0.15])
                            LG.ax.axhline(y=0.0,color='k')
                        else:
                            LG.ax.set_ylim([0,1.25])
                            LG.ax.axhline(y=1.0,color='k')
                        LG.ax.set_ylabel("Fractional Ignorance (bits)")
                    LG.ax.set_xticks(neighs)
                    LG.ax.set_xticklabels((dx*N.array(neighs)))
                    LG.ax.set_xlabel("Neighbourhood diameter (km)")
                

    if plot_detscores:
        if vrbl == 'REFL_comp':
            thresh = 25
            thstr = '{:02d}dBZ'.format(thresh)
        elif vrbl == 'accum_precip':
            thresh = 5
            thstr = '{:02d}mmh'.format(thresh)
        # thresh = 10
        det_box = False
        det_line = True

        if det_line:

            COLOR = dict(d01_reproj = 'blue',
                            d02_reproj = 'red',
                            )

            for score in ("POD","BIAS","FAR","CSI","PFD"):
                SCORES = {d: {fm: {e:[] for e in member_names} for fm in fcstmins} for d in doms_pp}
                labelset = set()
                for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
                    for n,ens in enumerate(member_names):
                        initstr = utils.string_from_time('dir',initutc,strlen='hour')
                        npydir = os.path.join(npyroot,initstr,)
                        fdir_det = os.path.join(npydir,'contingency')
                        fpath = generate_fname(vrbl=vrbl,metric='contingency',
                                    fcstmin=fcstmin,dom_pp=dom_pp,ensmember=ens,suffix=thstr,
                                    fdir=fdir_det,extension='.npz')
                        S = N.load(fpath)
                        SCORES[dom_pp][fcstmin][ens].append(S[score])
                        labelset.add(dom_pp)
                line_fpath = os.path.join(outroot,'linegraph_{}_{}.png'.format(score,vrbl))
                with LineGraph(fpath=line_fpath) as LG:
                    for dom_pp in doms_pp:
                        plotarray = N.zeros([len(member_names),len(fcstmins)])
                        for fidx,fm in enumerate(fcstmins):
                            for n,ens in enumerate(member_names):
                                plotarray[n,fidx] = N.mean(SCORES[dom_pp][fm][ens])
                        pkw = dict(lw=2,color=COLOR[dom_pp])
                        LG.plot(ydata=N.mean(plotarray,axis=0),xdata=fcstmins,
                                    plotkwargs=pkw,label=dom_pp)
                        for n,ens in enumerate(member_names):
                            pkw = dict(lw=1,color=COLOR[dom_pp])
                            LG.plot(ydata=plotarray[n,:],xdata=fcstmins,
                                    plotkwargs=pkw,label=dom_pp)
                            # pdb.set_trace()
                    LG.ax.set_title("Blue = d01    Red = d02")
        print("Done LineGraph.")
            
        if det_box:
            thstr = '{:02d}mmh'.format(thresh)
            for score in ("CSI","PFD"):
                SCORES = {d: {fm: [] for fm in fcstmins} for d in doms_pp}
                bxp_fpath = os.path.join(outroot,'boxplot_{}.png'.format(score))
                with BoxPlot(fpath=bxp_fpath) as BXP:
                    plotlist = []
                    labelset = set()
                    for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
                        for n,ens in enumerate(member_names):
                            initstr = utils.string_from_time('dir',initutc,strlen='hour')
                            npydir = os.path.join(npyroot,initstr,)
                            fdir_det = os.path.join(npydir,'contingency')
                            fpath = generate_fname(vrbl=vrbl,metric='contingency',
                                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember=ens,suffix=thstr,
                                        fdir=fdir_det,extension='.npz')
                            S = N.load(fpath)
                            SCORES[dom_pp][fcstmin].append(S[score])
                            labelset.add(dom_pp)
                    for fm in fcstmins:
                        for dom_pp in sorted(doms_pp):
                            plotlist.append(SCORES[dom_pp][fm])
                    # positions = (1,2,3,4,
                                # 6,7,8,9,
                                # 11,12,13,14)
                    positions=(1,2,4,5,7,8)
                    # pdb.set_trace()
                    
                    BXP.plot(plotlist,positions=positions,vert=True,
                            # labels=['d01','d02']*3,autorange=True)
                            labels=list(labelset)*3,autorange=True)
                print("Plotted",score)

    if plot_performance:
        # scores = DetScores.assign_scores_lookup(DetScores,remove_duplicates=True)
        # for score in scores:
            # pass

        # Colour-blind-safe colour set
        # http://mkweb.bcgsc.ca/colorblind/img/colorblindness.palettes.trivial.png
        # lk = {'bbox_to_anchor':(-0.15,1.0),'ncol':1,}
        performance_pps = ['d01_cut','d02']
        lk = {}
        sz = 25
        alpha = 0.6
        MARKERS = {}
        SIZES = {}
        marks = ['o','v','s','*']
        for i,pp in enumerate(doms_pp):
            MARKERS[pp] = marks[i]
            SIZES[pp] = sz
                
        # Blue, Vermillion, Reddishpurple - no alpha.
        COLORS = utils.colorblind_friendly(keys=fcstmins)
        # COLOURS = {60:[(0,114,178)],120:[(213,93,0)],180:[(204,121,167)]}
        thstr = '{:02d}mmh'.format(thresh)
        pd_fpath = os.path.join(outroot,'performance_{}.png'.format(vrbl))
        POD = {f:{dom_pp:{ens:N.array([]) for ens in member_names} for dom_pp in doms_pp} for f in fcstmins}
        FAR = {f:{dom_pp:{ens:N.array([]) for ens in member_names} for dom_pp in doms_pp} for f in fcstmins}
        with Performance(fpath=pd_fpath,legendkwargs=lk,legend=True) as PD:
            for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
                if dom_pp not in performance_pps:
                    continue
                for n,ens in enumerate(member_names):
                    initstr = utils.string_from_time('dir',initutc,strlen='hour')
                    npydir = os.path.join(npyroot,initstr,)
                    fdir_det = os.path.join(npydir,'contingency')
                    fpath = generate_fname(vrbl=vrbl,metric='contingency',
                                fcstmin=fcstmin,dom_pp=dom_pp,ensmember=ens,suffix=thstr,
                                fdir=fdir_det,extension='.npz')
                    S = N.load(fpath)
                    POD[fcstmin][dom_pp][ens] = N.append(POD[fcstmin][dom_pp][ens],S['POD'])
                    FAR[fcstmin][dom_pp][ens] = N.append(FAR[fcstmin][dom_pp][ens],S['FAR'])
            # pdb.set_trace()
            for n,ens in enumerate(member_names):
                for dom_pp, fm in itertools.product(doms_pp,fcstmins):
                    if dom_pp not in performance_pps:
                        continue
                    podmean = N.nanmean(POD[fm][dom_pp][ens])
                    farmean = N.nanmean(FAR[fm][dom_pp][ens])
                    pk = {'marker':MARKERS[dom_pp],'c':COLORS[fm],'s':SIZES[dom_pp],
                                'alpha':alpha}
                    lstr = "Domain {} after {}h".format(dom_pp,int(fm/60))
                    print("Plotting {} point for".format(ens),lstr)
                    if n>0:
                        lstr = None
                    PD.plot_data(pod=podmean,far=farmean,plotkwargs=pk,label=lstr)
                    print("POD = {:.3f} and FAR = {:.3f}.".format(podmean,farmean))
            PD.ax.set_xlim([0,0.3])
            PD.ax.set_ylim([0,0.3])

    if plot_crps:
        inithrs = {i:int(i.hour) for i in initutcs}
        crps_dict = {i: {d: {fm:[] for fm in fcstmins} for d in doms_pp} for i in inithrs.values()}
        # for d,fm in itertools.product(doms,fcstmins):
            # crps_dict[d][fm] = []
        for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
            i = inithrs[initutc]
            initstr = utils.string_from_time('dir',initutc,strlen='hour')

            ensdir = os.path.join(ensroot,initstr,)
            npydir = os.path.join(npyroot,initstr,)
            outdir = os.path.join(outroot,initstr,)

            print("----- {} on domain {} -----".format(initstr,dom_pp))
            fpath = generate_fname(vrbl=vrbl,metric='CRPS',
                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember='mean',
                        fdir=npydir,extension='.npy')
            crps_dict[i][dom_pp][fcstmin].append(float(N.load(fpath)))
        
        COLORS = utils.colorblind_friendly(keys=sorted(inithrs.values()))
        LINES = {'d01_cut':'-','d01_reproj':'--','d02':'-.','d02_reproj':':'}
        fname = 'all_crps_{}'.format(vrbl)
        with LineGraph(outdir=outroot,fname=fname,legend=True) as LG:
            for initutc, inithr in sorted(inithrs.items()):
                actualtimes = [initutc + datetime.timedelta(seconds=60*int(v)) for v in fcstmins]
                for dom_pp in doms_pp:
                    lstr = 'Domain {}, {}Z'.format(dom_pp,inithr)
                    pkw = dict(label=lstr,linestyle=LINES[dom_pp],c=COLORS[inithr])
                    timeseries = N.ones(len(fcstmins))
                    for n,fm in enumerate(fcstmins):
                        timeseries[n] = N.array(N.nanmean(crps_dict[inithr][dom_pp][fm]))
                    LG.plot(xdata=actualtimes,ydata=timeseries,save=None,plotkwargs=pkw)
            LG.legendloc = 3
            LG.legendfontsize = 6

    if plot_brier:
        thresh = 10
        inithrs = {i:int(i.hour) for i in initutcs}
        bs_dict = {i: {d: {fm:[] for fm in fcstmins} for d in doms_pp} for i in inithrs.values()}
        for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
            i = inithrs[initutc]
            initstr = utils.string_from_time('dir',initutc,strlen='hour')

            ensdir = os.path.join(ensroot,initstr,)
            npydir = os.path.join(npyroot,initstr,)
            outdir = os.path.join(outroot,initstr,)

            print("----- {} on domain {} -----".format(initstr,dom_pp))
            fpath = generate_fname(vrbl=vrbl,metric='BS',suffix='th_{}'.format(thresh),
                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember='all',
                        fdir=npydir,extension='.npy')
            bs_dict[i][dom_pp][fcstmin].append(float(N.load(fpath)))
        
        COLORS = utils.colorblind_friendly(keys=sorted(inithrs.values()))
        LINES = {'d01_cut':'-','d01_reproj':'--','d01_kde':':',
                    'd02':'-','d02_reproj':'--','d02_kde':':'}
        LWS = {'d01_cut':1,'d01_reproj':1,'d01_kde':1,
                    'd02':2,'d02_reproj':2,'d02_kde':2}
        with LineGraph(outdir=outroot,fname='all_bs_{}'.format(thresh),legend=True) as LG:
            for initutc, inithr in sorted(inithrs.items()):
                actualtimes = [initutc + datetime.timedelta(seconds=60*int(v)) for v in fcstmins]
                for dom_pp in doms_pp:
                    lstr = 'Domain {}, {}Z'.format(dom_pp,inithr)
                    pkw = dict(label=lstr,linestyle=LINES[dom_pp],c=COLORS[inithr],
                                lw=LWS[dom_pp])
                    timeseries = N.ones([3])
                    for n,fm in enumerate(fcstmins):
                        timeseries[n] = N.array(N.nanmean(bs_dict[inithr][dom_pp][fm]))
                    LG.plot(xdata=actualtimes,ydata=timeseries,save=None,plotkwargs=pkw)
            LG.legendloc = 2
            LG.legendfontsize = 5


    if plot_object_hist:
        all_data = N.empty([0])
        labels = []
        vrbl_obj = 'REFL_comp'; thresh = 25; fp = 15
        cell_vrbl = 'W' # UH - need to fix
        hist_pps = ['d01_reproj','d02_reproj']
        for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
            if dom_pp not in hist_pps:
                continue
            initstr = utils.string_from_time('dir',initutc,strlen='hour')
            npydir = os.path.join(npyroot,initstr,)
            npydir_obj = os.path.join(npydir,"objects_{}".format(dom_pp))
            # ensdir = os.path.join(ensroot,initstr,)
            # outdir = os.path.join(outroot,initstr,)
            for n,member in enumerate(member_names):
                fpath = generate_fname(vrbl=cell_vrbl,metric='{}_objs'.format(vrbl),
                                fcstmin=fcstmin,dom_pp=dom_pp,ensmember=member,
                                fdir=npydir_obj,suffix="{}px_{}th".format(fp,thresh))
                data = N.load(fpath)
                data = data[~N.isnan(data)]
                all_data = N.hstack((all_data,data))
                labels.append(dom_pp)
                # pdb.set_trace()
        fpath = os.path.join(outroot,'all_object_histogram.png')
        data = N.array(all_data)
        with Histogram(fpath=fpath,data=data) as H:
            H.plot(labels=labels)
            
    if plot_esal:
        itr = list(itr)
        vrblset = set()
        for i in itr:
            vrblset.add(i[0])
        for vrbl in vrblset:
            MARKER = dict(d01_reproj = 'o', d02_reproj = 's')
            # vrbl = 'REFL_comp'; thresh = 25; fp = 15
            if vrbl == 'REFL_comp':
                thresh = 25; fp = 15
            elif vrbl == 'accum_precip':
                thresh = 5; fp = 15

            for pp in ("d01_reproj","d02_reproj"):
                outfpath = os.path.join(outroot,'all_eSAL_{}_{}.png'.format(pp,vrbl))

                # with SALGraph(fpath=outfpath) as salgraph:
                salgraph = SALGraph(fpath=outfpath)
                for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
                    if dom_pp != pp:
                        continue
                    initstr = utils.string_from_time('dir',initutc,strlen='hour')

                    ensdir = os.path.join(ensroot,initstr,)
                    npydir = os.path.join(npyroot,initstr,)
                    initstr = utils.string_from_time('dir',initutc,strlen='hour')
                    npydir = os.path.join(npyroot,initstr,)
                    npydir_obj = os.path.join(npydir,"objects_{}".format(dom_pp))
                    fpath = generate_fname(vrbl=vrbl,metric='ESAL',
                                    fcstmin=fcstmin,dom_pp=dom_pp,ensmember='all',
                                    fdir=npydir_obj,suffix="{}px_{}th".format(fp,thresh),
                                    extension='.npy')
                    eS, eA, eL = N.load(fpath)
                    skip = 0
                    print(eS, eA, eL)
                    #pdb.set_trace()
                    eS = min(2.0,eS)
                    eS = max(-2.0,eS)
                    eA = min(2.0,eA)
                    eA = max(-2.0,eA)
                    eL = min(2.0,eL)
                    eL = max(0,eL)
                    print("Plotting!")
                    print(eS, eA, eL)
                    salgraph.plot(S=eS,A=eA,L=eL,marker=MARKER[dom_pp])
                    #pdb.set_trace()
                salgraph.plot_medians()
                salgraph.plot_quartiles()
                salgraph.plot_colorbar()
                salgraph.save()

    if plot_fss:
        thedoms = ('d01_reproj','d02_reproj')
        for dom_pp, initutc in itertools.product(thedoms,initutcs):
            initstr = utils.string_from_time('dir',initutc,strlen='hour')

            ensdir = os.path.join(ensroot,initstr,)
            npydir = os.path.join(npyroot,initstr,)
            outdir = os.path.join(outroot,initstr,)

            # out_fpath = os.path.join(outdir,'efss_{}.png'.format(dom_pp))
            out_fpath = os.path.join(outdir,'efss_nonorm_{}_{}.png'.format(dom_pp,vrbl))
            # if os.path.exists(out_fpath) and not overwrite:
                # print("SKIPPING.")
                # continue

            # vrbl = 'REFL_comp'
            vrbl = 'accum_precip'

            fdir_efss = npydir
            fpath = generate_fname(vrbl=vrbl,metric='eFSS',
                            fcstmin=0,dom_pp=dom_pp,ensmember='all',
                            fdir=fdir_efss,extension='.pickle')
            efss_inst = pickle.load(open(fpath,"rb"))


            # if dom_pp.startswith("d01"):
                # domdx = 3
            # else:
                # domdx = 1
                # pdb.set_trace()
            with EFSS_Intensity(fpath=out_fpath,efss_inst=efss_inst) as EFSSi:
                pkw = dict(vmin=0.0,vmax=1.0,cmap=M.cm.jet)
                # pkw = dict(cmap=M.cm.jet)
                xl = 'auto'
                x2l = [str(15*n) for n in efss_inst.temporal_ms] * efss_inst.nthreshs
                yl = efss_inst.spatial_ns * 5
                yl = [str(int(l)) for l in yl] 
                EFSSi.plot_intensity(plotkwargs=pkw,xticklabels=xl,
                                        x2ticklabels=x2l,yticklabels=yl)
                EFSSi.ax.set_xlabel("Threshold (mm/hr)")
                EFSSi.ax_top.set_xlabel("Temporal window (min)")
                EFSSi.ax.set_ylabel("Spatial window (km)")
                
            # pdb.set_trace()
            pass

    if plot_casati:
        nmems = len(member_names)
        init_switch = False
        # fcsthrs = (1,2,3)
        vrbl = "REFL_comp"


        # for fcsthr in fcsthrs:
        for fcstmin in fcstmins:
            if vrbl == "REFL_comp":
                xlabel = "Thresholds (dBZ)"
                # thresh_labs = [2.5,5,10,20,30,40,50,60]
                pass
                # threshs = range(10,60,10)
            elif vrbl == "accum_precip":
                # thresh_labs = (0.125,0.25,0.5,1,2,4,8,16,32)
                # fcsthr = int(fcstmin / 60)
                pass
            else:
                raise Exception
            
            for score in ("MSE","SS"):
                thedoms = ('d01_reproj','d02_reproj')
                ntimes = len(initutcs)
                out_fpath = os.path.join(outroot,'casati_mse_compare_{:02d}min_{}_{}.png'.format(fcstmin,vrbl,score))
                arrT = []
                # if os.path.exists(out_fpath) and not overwrite:
                    # print("SKIPPING.")
                    # return
                for ndom,dom_pp in enumerate(thedoms):
                    for nidx, initutc in enumerate(initutcs):
                        nidx = initutcs.index(initutc)
                        initstr = utils.string_from_time('dir',initutc,strlen='hour')

                        ensdir = os.path.join(ensroot,initstr,)
                        npydir = os.path.join(npyroot,initstr,)
                        outdir = os.path.join(outroot,initstr,)
                        fdir_casati = os.path.join(npydir,'casati')


                        for nmem,ens in enumerate(member_names):
                            fpath = generate_fname(vrbl=vrbl,metric='casati',suffix=score,
                                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember=ens,
                                        fdir=fdir_casati,extension='.pickle')
                            casati_dict = pickle.load(open(fpath,"rb"))
                            # pdb.set_trace()
                            if not init_switch:
                                threshs = sorted(list(casati_dict.keys()))
                                thresh_labs = N.around([2**th for th in threshs],2)
                                nthresh = len(threshs)

                                Ls = sorted(list(casati_dict[threshs[0]].keys()))
                                nLs = len(Ls)
                                area_arr = N.zeros([nmems,ntimes,nthresh,nLs])
                                init_switch = True
                            for nth, th in enumerate(threshs):
                                for nL, L in enumerate(Ls):
                                    area_arr[nmem,nidx,nth,nL] = casati_dict[th][L]

                    # Averaging over ensemble member
                    ave_arr1 = N.nanmean(area_arr,axis=0)
                    # Then case
                    ave_arr2 = N.nanmean(ave_arr1,axis=0)
                    arrT.append(ave_arr2.T)
                
                arr_diff = arrT[1] - arrT[0]
                # pdb.set_trace()
                # lvs = N.arange(0.0,0.065,0.005)
                if score == "MSE":
                    if vrbl == "REFL_comp":
                        lvs = N.arange(-0.02,0.022, 0.002)
                        # lvs = None
                    else:
                        lvs = None
                elif score == "SS":
                    lvs = None
                else:
                    raise Exception
                # cm = M.cm.seismic
                # cm = M.cm.inferno
                cm = M.cm.RdGy_r
                pkw = dict(cmap=cm,levels=lvs)
                # pdb.set_trace()
                # with AreaFill(fpath=out_fpath,data=arr_diff,Ls=Ls,thresholds=threshs) as AF:
                with AreaFill(fpath=out_fpath,data=arr_diff,Ls=Ls,thresholds=thresh_labs) as AF:
                    AF.plot(plotkwargs=pkw,cb=True,L_multiplier=5)
                    AF.ax.set_xlabel(xlabel)
                    AF.ax.set_ylabel("Spatial scale (km)")
                print("Plotted.")

    if plot_fv:
        # Forecast value
        # Do for:
        # Probabilistic
        # Random det member
        # Closest-to-mean member
        # Ensemble mean
        # Prob-matched-mean
        thresh = 5
        for vrbl, caseutc, initutc, fcstmin, dom_pp in itr:
            initstr = utils.string_from_time('dir',initutc,strlen='hour')
            npydir = os.path.join(npyroot,initstr,)
            outdir = os.path.join(outroot,initstr,)
            outdir_fv = os.path.join(outdir,"FV")
            FVs = dict()
            for nm, member in enumerate(member_names):
                fpath = generate_fname(vrbl=vrbl,metric='FVdet',
                            fcstmin=fcstmin,dom_pp=dom_pp,ensmember=member,
                            fdir=npydir,suffix="th{}".format(thresh),
                            extension='.npy')
                FVs[member] = N.load(fpath)
            # Probabilistic:
            fpath = generate_fname(vrbl=vrbl,metric='FVens',
                        fcstmin=fcstmin,dom_pp=dom_pp,ensmember='all',
                        fdir=npydir,suffix="th{}".format(thresh),
                        extension='.npy')
            FVens = N.load(fpath)

            fpath = generate_fname(vrbl=vrbl,metric='FV',
                    fcstmin=fcstmin,dom_pp=dom_pp,ensmember='all',
                    fdir=outdir_fv,extension='.png',suffix="th{}".format(thresh))
            # COLORS = utils.colorblind_friendly(keys=sorted(inithrs.values()))
            # LINES = {1:'-',2:'--'}
            with LineGraph(outdir=outroot,fpath=fpath,legend=True,log='x') as LG:
                for nm, member in enumerate(member_names):
                    # pkw = dict(label=lstr,linestyle=LINES[dom],c=COLORS[inithr])
                    pkw = dict(c='red',lw=1)
                    LG.plot(xdata=N.arange(0.01,1.01,0.01),ydata=FVs[member],
                                save=None,plotkwargs=pkw)
                pkwe = dict(c='black',lw=2)
                LG.plot(xdata=N.arange(0.01,1.01,0.01),ydata=FVens,
                            save=None,plotkwargs=pkwe)
                LG.ax.set_ylim([0,1])
                LG.ax.set_xlabel("Cost/loss ratio")
                LG.ax.set_ylabel("Forecast value")
                cls_lab = [0.01,0.1,0.5,1.0]
                LG.ax.set_xticks(cls_lab)
                LG.ax.set_xticklabels(cls_lab)
            # pdb.set_trace()
                


    return

    
### PROCEDURE ###
if __name__ == "__main__":


    if check_only:
        iii = generate_run_check()
        if ncpus == 1:
            results = []
            for i in iii:
                results.append(check_wrfout(i))
        else:
            with multiprocessing.Pool(ncpus) as pool:
                results = pool.map(check_wrfout,iii,)#chunksize=1)
        print("Now check results.")

        res = [r for r in results if r != None]
        thefile = open('/scratch/john.lawson/WRF/VSE_reso/ForReal_nco/failed.txt','w')
        for r in res:
            thefile.write("{}\n".format(r))
        thefile.close()
        assert True == False
    # itr = generate_iterable()
    itr = list(generate_iterable())
    random.shuffle(itr)

    if compute_switch:
        if ncpus == 1:
            for i in itr:
                run_parallel(i)
            # map(run_parallel,itr)
        else:
            with multiprocessing.Pool(ncpus) as pool:
                results = pool.map(run_parallel,itr,)#chunksize=1)

    # Parallelise over all init times.
    fss_itr = generate_iterable(fss=True)
    if compute_fss:
        for i in fss_itr:
            fss(i)
    # Again because the first one was "used up?"
    itr = generate_iterable()
    if plot_switch:
        plot(itr)




