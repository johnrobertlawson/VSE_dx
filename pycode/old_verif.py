import pdb
import os
import datetime
import collections

import numpy as N

from evac.stats.verif import Verif
from evac.datafiles.ensemble import Ensemble
# from evac.datafiles.stageiv import StageIV
from evac.datafiles.obsgroup import ObsGroup
from evac.stats.detscores import DetScores
import evac.utils as utils

ncpus = 1
ensroot = '/scratch/john.lawson/WRF/VSE_reso'
npyroot = '/work/john.lawson/VSE_reso/scores'
radardir = '/work/john.lawson/NEXRAD_data'
st4dir = '/work/john.lawson/STAGEIV_data'
outroot = '/home/john.lawson/VSE_reso/pyoutput'
rd = dict(nx=50,ny=50)
fctimes = list(range(1,4))
quarterly = list(N.arange(0.25,3.25,0.25))

CASES = collections.OrderedDict()
CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                        datetime.datetime(2016,3,31,19,0,0),
                        datetime.datetime(2016,3,31,20,0,0),
                        datetime.datetime(2016,3,31,21,0,0),
                        datetime.datetime(2016,3,31,22,0,0),
                        ]
CASES[datetime.datetime(2017,5,1,0,0,0)] = [
                        datetime.datetime(2017,5,1,19,0,0),
                        ]
CASES[datetime.datetime(2017,5,2,0,0,0)] = [
                        datetime.datetime(2017,5,3,2,0,0),
                        ]
CASES[datetime.datetime(2017,5,4,0,0,0)] = [
                        datetime.datetime(2017,5,4,23,0,0),
                        datetime.datetime(2017,5,5,0,0,0),
                        datetime.datetime(2017,5,5,1,0,0),
                        ]
# To do - 20180429 (texas panhandle)

# These case times will be skipped to speed up everything
skiplist = [
        datetime.datetime(2016,3,31,0,0,0),
        ]

# QUICK CHECKS/HACKS
skiplist = []
CASES = {
        # datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,19,0,0),],
        datetime.datetime(2017,5,4,0,0,0):[datetime.datetime(2017,5,5,0,0,0),],
        }
# fctimes = (1,)

# SETTINGS THAT REQUIRE LOADING ENSEMBLE DATA
plot_thumbs = 1
compute_detscores = 0
compute_crps = 0
compute_obj = 0

load_ens_switch = max(compute_crps,compute_detscores,plot_thumbs,
                    compute_obj)

# PLOTTING COMPUTED SCORES
plot_detscores = 0
plot_crps = 0
plot_object_hist = 0

# OBSERVATION GROUPS
ST4 = ObsGroup(st4dir,'stageiv')
RADARS = ObsGroup(radardir,'radar')
OBS = (ST4,RADARS)

init_dict = {}
for caseutc,initutcs in CASES.items():
    if caseutc in skiplist:
        print("Skipping {} - in skiplist.".format(caseutc))
        continue
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    for initutc in initutcs:
        initdir = utils.string_from_time('dir',initutc,strlen='hour')
        ensdir = os.path.join(ensroot,initdir)
        npydir = os.path.join(npyroot,initdir)
        outdir = os.path.join(outroot,casestr,initdir)
        
        init_dict[initutc] = npydir

        if not load_ens_switch:
            continue
        E = Ensemble(ensdir,initutc,ndoms=2,ctrl=False)
        V = Verif(ensemble=E,obs=OBS,outdir=outdir,
                    datadir=npydir,reproject_dict=rd)
        # Need a "only_compute_if_not_saved_already" method in V.
       
        if compute_crps:
            V.compute_stats("CRPS",'accum_precip',fctimes,dom='all',
                ncpus=ncpus,crps_thresholds=N.arange(0,101),method=1)
        if compute_detscores:
            V.compute_stats(stats="detscores",vrbl='accum_precip',verif_times=fctimes,
                        dom='all',ncpus=ncpus,det_thresholds=N.arange(0,35,5),method=1)
        if plot_thumbs:
            V.plot_thumbnails(vrbl='REFL_comp',fchrs=fctimes,radardir=radardir,
                            overwrite=False)
            # V.plot_thumbnails(vrbl='accum_precip',fchrs=range(1,4))
            # V.plot_thumbnails(vrbl='CAPE',fchrs=range(1,4),verif_first=False)
        if compute_obj:
            # V.outdir = os.path.join(outdir,
            V.compute_objectbased(vrbl1='REFL_comp',vrbl2='W',
                        fchrs=quarterly,thresh=35,footprint=10,
                        quickplot=True,dx=(3,1),doms=(1,2))

    # pdb.set_trace()
    if plot_detscores:
        scores = DetScores.assign_scores_lookup(DetScores,remove_duplicates=True)
        V2 = Verif(obs=OBS,outdir=outdir,
                    datadir=npydir,reproject_dict=rd)
        # scores = ['CSI',]
        for score in scores:
            # Violin plots box plots too.
            V2.plot_stats(plot='violin',init_dict=init_dict,
                outdir=os.path.join(outdir,'detscores'),score=score,
                vrbl='accum_precip',ndoms=2,interval=1,lv='sfc',
                ens='all',thresh=10,figsize=(10,6),datadir=npydir)

    if plot_crps:
        V2 = Verif(obs=OBS,outdir=outdir,
                    datadir=npydir,reproject_dict=rd)
        V2.plot_stats('trails',init_dict=init_dict,outdir=outdir,score='CRPS',vrbl='accum_precip',
                    ndoms=2,interval=1,lv='sfc',ens='mean',fname='crps_trails.png',datadir=npydir)

    if plot_object_hist:
        V2 = Verif(obs=OBS,outdir=outdir,
                    datadir=npydir,reproject_dict=rd)
        V2.plot_object_pdfs(vrbl1='REFL_comp',vrbl2='W',)
