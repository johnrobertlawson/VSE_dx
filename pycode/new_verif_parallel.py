import pdb
import logging
import os
import glob
import collections
import datetime
import copy
import pickle
import itertools
import multiprocessing
import argparse
import time

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
import scipy
import pandas
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

import evac.utils as utils
from evac.datafiles.wrfout import WRFOut
from evac.plot.domains import Domains
from evac.stats.detscores import DetScores
from evac.plot.performance import Performance
from evac.stats.fi import FI
from evac.object.objectid import ObjectID
from evac.object.catalogue import Catalogue

### ARG PARSE ####
parser = argparse.ArgumentParser()

parser.add_argument('-N','--ncpus',dest='ncpus',default=1,type=int)
parser.add_argument('-oo','--overwrite_output',dest='overwrite_output',
                            action='store_true',default=False)
parser.add_argument('-op','--overwrite_pp',dest='overwrite_pp',
                            action='store_true',default=False)

PA = parser.parse_args()
ncpus = PA.ncpus
overwrite_output = PA.overwrite_output
overwrite_pp = PA.overwrite_pp

### SWITCHES ###
do_domains = False
do_performance = False
do_efss = False # Also includes FISS, which is broken?

do_object_performance = False
do_object_distr = False
do_object_pca = True
do_object_lda = False
object_switch = max(do_object_performance,do_object_distr,
                    do_object_pca,do_object_lda)

### SETTINGS ###
#mp_log = multiprocessing.log_to_stderr()
#mp_log.setLevel(logging.INFO)

CASES = collections.OrderedDict()
CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                        # datetime.datetime(2016,3,31,19,0,0),
                        # datetime.datetime(2016,3,31,20,0,0),
                        datetime.datetime(2016,3,31,21,0,0),
                        # datetime.datetime(2016,3,31,22,0,0),
                        # datetime.datetime(2016,3,31,23,0,0),
                        ]
CASES[datetime.datetime(2017,5,1,0,0,0)] = [
                        # datetime.datetime(2017,5,1,19,0,0),
                        # datetime.datetime(2017,5,1,20,0,0),
                        datetime.datetime(2017,5,1,21,0,0),
                        # datetime.datetime(2017,5,1,22,0,0),
                        # datetime.datetime(2017,5,1,23,0,0),
                        ]
CASES[datetime.datetime(2017,5,2,0,0,0)] = [
                        # datetime.datetime(2017,5,2,23,0,0),
                        # datetime.datetime(2017,5,3,0,0,0),
                        datetime.datetime(2017,5,3,1,0,0),
                        # datetime.datetime(2017,5,3,2,0,0),
                        # datetime.datetime(2017,5,3,3,0,0),
                        ]
CASES[datetime.datetime(2017,5,4,0,0,0)] = [
                        # datetime.datetime(2017,5,4,22,0,0),
                        # datetime.datetime(2017,5,4,23,0,0),
                        datetime.datetime(2017,5,5,0,0,0),
                        # datetime.datetime(2017,5,5,1,0,0),
                        # datetime.datetime(2017,5,5,2,0,0),
                        ]
# To do - 20180429 (texas panhandle)
# CASES[datetime.datetime(2018,4,29,0,0,0)] = []

### DIRECTORIES ###
extractroot = "/home/nothijngrad/Xmas_Shutdown/Xmas"
objectroot = os.path.join(extractroot,'object_instances')
#outroot = "/home/nothijngrad/Xmas_Shutdown/pyoutput"
outroot = "/mnt/d/pyoutput"



##### OTHER STUFF #####
stars = "*"*10
# ncpus = 8
dom_names = ("d01","d02")
domnos = (1,2)
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
# doms = (1,2)
fcst_vrbls = ("Wmax","UH02","UH25","RAINNC","REFL_comp")
obs_vrbls = ("AWS02","AWS25","DZ","ST4","NEXRAD")
all_fcstmins = N.arange(0,185,5)
fcst_fmts = ("d01_3km","d02_1km","d02_3km")



#### FOR DEBUGGING ####
# CASES = {datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,21,0,0),]}

# fcst_vrbls = ("REFL_comp",)
# fcst_vrbls = ("Wmax",)
fcst_vrbls = ("REFL_comp","Wmax")
obs_vrbls = ("NEXRAD",)
# obs_vrbls = list()

#########################


def get_extraction_fpaths(vrbl,fmt,validutc,caseutc,initutc=None,mem=None):
    """ Return the file path for the .npy of an interpolated field.

    Vrbl could be the forecast or observed field.
    utc is the valid time of forecast or observation.
    fmt is the extracted-grid name

    inistr is only required for forecasts
    mem too

    Args:
        initstr: String for

    Something like:

    FORECASTS:
    uh02_d02_3km_20160331_0100_0335_m01.npy

    OBS:
    aws02_mrms_rot_3km_20160331_0335.npy

    """
    if vrbl in fcst_vrbls: # ("Wmax","UH02","UH25","RAINNC"):
        # TODO: are we not just doing 5-min or 1-hr accum_precip?
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        initHHMM = "{:02d}{:02d}".format(initutc.hour, initutc.minute)
        validHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "{}_{}_{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,initHHMM,
                                        validHHMM,mem)
    elif vrbl in obs_vrbls: # ("AWS02","AWS25","DZ","ST4"):
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        # utcYYYYMMDD = validutc
        utcHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,utcHHMM)
    return os.path.join(extractroot,caseYYYYMMDD,fname)

def get_object_picklepaths(vrbl,fmt,validutc,caseutc,initutc=None,mem=None):
    if vrbl in fcst_vrbls: # ("Wmax","UH02","UH25","RAINNC"):
        # TODO: are we not just doing 5-min or 1-hr accum_precip?
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        initHHMM = "{:02d}{:02d}".format(initutc.hour, initutc.minute)
        validHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "objects_{}_{}_{}_{}_{}_{}.pickle".format(vrbl,fmt,caseYYYYMMDD,initHHMM,
                                        validHHMM,mem)
    elif vrbl in obs_vrbls: # ("AWS02","AWS25","DZ","ST4"):
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        # utcYYYYMMDD = validutc
        utcHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "objects_{}_{}_{}_{}.pickle".format(vrbl,fmt,caseYYYYMMDD,utcHHMM)
    return os.path.join(objectroot,caseYYYYMMDD,fname)

def create_bmap(urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon,ax=None):
    bmap = Basemap(
                # width=12000000,height=9000000,
                urcrnrlat=urcrnrlat,urcrnrlon=urcrnrlon,
                llcrnrlat=llcrnrlat,llcrnrlon=llcrnrlon,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection='lcc',ax=ax,
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
    return bmap

def get_random_domain(caseutc,dom):
    initutc = CASES[caseutc][0]
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    initstr = utils.string_from_time('dir',initutc,strlen='hour')

    m01dir = os.path.join(ensroot,initstr,"m01")
    dom_fname = get_wrfout_fname(initutc,dom)
    dom_fpath = os.path.join(m01dir,dom_fname)
    dom_nc = Dataset(dom_fpath)
    return dom_nc

def get_wrfout_fname(t,dom):
    fname = 'wrfout_d{:02d}_{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(dom,
                t.year,t.month,t.day,t.hour,t.minute,t.second)
    return fname

def fc2ob(fcst_vrbl,fcst_fmt):
    """ Look up the corresponding variable/format name for obs, from vrbls.
    """
    # This is the observational variable
    VRBLS = {"REFL_comp":"NEXRAD",}
    obs_vrbl = VRBLS[fcst_vrbl]

    # This is the dx for the grid (e.g. 1km)
    *_, dx = fcst_fmt.split("_")

    # These are the field codes
    OO = {'NEXRAD':'nexrad'}
    obs_fmt = "_".join((OO[obs_vrbl],dx))

    # pdb.set_trace()
    return obs_vrbl,obs_fmt

def ob2fc(obs_vrbl,obs_fmt):
    VRBLS = {"NEXRAD":"REFL_comp",}
    fcst_vrbl = VRBLS[obs_vrbl]

    # This is the dx for the grid (e.g. 1km)
    *_, dx = obs_fmt.split("_")

    if dx == "3km":
        fcst_fmt = "d01_3km"
    elif dx == "1km":
        fcst_fmt = "d02_1km"

    return fcst_vrbl, fcst_fmt
    # fcst_vrbl, fcst_fmt = ob2fc(obs_vrbl,obs_fmt)

def loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,members):
    for mem in members:
        for fcstmin in fcstmins:
            for caseutc, initutcs in CASES.items():
                for initutc in initutcs:
                    validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))
                    yield fcst_vrbl, fcst_fmt, validutc, caseutc, initutc, mem

def loop_obj_obs(obs_vrbl,all_times=True):
    if all_times is True:
        ts = N.arange(0,185,5)
    else:
        ts = all_times
    obtimes = set()
    casetimes = {}
    for caseutc, initutcs in CASES.items():
        casetimes[caseutc] = set()
        for initutc in initutcs:
            for t in ts:
                validutc = initutc+datetime.timedelta(seconds=60*int(t))
                obtimes.add(validutc)
                casetimes[caseutc].add(validutc)

    # pdb.set_trace()
    for t in obtimes:
        case = None
        for caseutc in CASES.keys():
            if t in casetimes[caseutc]:
                case = caseutc
                break
        assert case is not None
        yield obs_vrbl, obs_fmt, t, case

def loop_efss(vrbl,fcstmin,fmt):
    for caseutc, initutcs in CASES.items():
        for initutc in initutcs:
            validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))
            yield vrbl, caseutc, initutc, fmt, validutc


def loop_perf(vrbl,thresh,fcstmin=None,fcst_fmt=None):
    if fcstmin:
        fcstmins = (fcstmin,)

    if fcst_fmt:
        fcst_fmts = (fcst_fmt,)

    #for vrbl in ("REFL_comp",):
    for caseutc, initutcs in CASES.items():
        for initutc in initutcs:
            for mem in member_names:
                for validmin in fcstmins:
                    validutc = initutc+datetime.timedelta(seconds=60*int(validmin))
                    for fmt in fcst_fmts:
                        yield vrbl, caseutc, initutc, mem, fmt, validutc, thresh

def load_both_data(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem):
        if mem == "all":
            for midx, mem in enumerate(member_names):
                fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                        validutc=validutc,caseutc=caseutc,initutc=initutc,
                        mem=mem)
                temp_data = N.load(fcst_fpath)
                if midx == 0:
                    fcst_data = N.zeros([len(member_names),*temp_data.shape])
                fcst_data[midx,:,:] = temp_data
        else:
            fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                    validutc=validutc,caseutc=caseutc,initutc=initutc,
                    mem=mem)
            fcst_data = N.load(fcst_fpath)

        # These are the corresponding obs formats/data
        obs_vrbl, obs_fmt = fc2ob(fcst_vrbl,fcst_fmt)
        obs_fpath = get_extraction_fpaths(vrbl=obs_vrbl,fmt=obs_fmt,
                        validutc=validutc,caseutc=caseutc)
        obs_data = N.load(obs_fpath)

        return fcst_data, obs_data

def load_fcst_dll(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem):
    if mem == "all":
        for midx, mem in enumerate(member_names):
            fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                    validutc=validutc,caseutc=caseutc,initutc=initutc,
                    mem=mem)
            temp_data = N.load(fcst_fpath)
            if midx == 0:
                fcst_data = N.zeros([len(member_names),*temp_data.shape])
            fcst_data[midx,:,:] = temp_data
    else:
        fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                validutc=validutc,caseutc=caseutc,initutc=initutc,
                mem=mem)
        fcst_data = N.load(fcst_fpath)

    fcst_lats, fcst_lons = load_latlons(fcst_fmt,caseutc)
    return fcst_data, fcst_lats, fcst_lons

def load_obs_dll(validutc,caseutc,obs_vrbl=None,fcst_vrbl=None,obs_fmt=None,
                    fcst_fmt=None,):
    if (obs_vrbl is None) and (obs_fmt is None):
        obs_vrbl, obs_fmt = fc2ob(fcst_vrbl,fcst_fmt)
    elif (fcst_vrbl is None) and (fcst_fmt is None):
        fcst_vrbl, fcst_fmt = ob2fc(obs_vrbl,obs_fmt)
    else:
        print("Check args:",obs_vrbl,obs_fmt,fcst_vrbl,fcst_fmt)
        raise Exception

    obs_fpath = get_extraction_fpaths(vrbl=obs_vrbl,fmt=obs_fmt,
                    validutc=validutc,caseutc=caseutc)
    obs_data = N.load(obs_fpath)

    obs_lats, obs_lons = load_latlons(fcst_fmt,caseutc)
    return obs_data, obs_lats, obs_lons


def load_timewindow_both_data(vrbl,fmt,validutc,caseutc,initutc,window=1,mem='all'):
    nt = window
    ws = int((nt-1)/2)

    if mem != 'all':
        raise Exception
    # Load central time first
    fcst_c, obs_c = load_both_data(vrbl,fmt,validutc,caseutc,initutc,'all')
    nens, nlat, nlon = fcst_c.shape
    fcst_data = N.zeros([nens,nt,nlat,nlon])
    fcst_data[:,ws,:,:] = fcst_c

    obs_data = N.zeros([nt,nlat,nlon])
    obs_data[ws,:,:] = obs_c

    t_offs = N.linspace(-ws,ws,window)
    for tidx,t_off in enumerate(t_offs):
        if t_off == 0:
            continue

        offsetutc = validutc + datetime.timedelta(seconds=t_off*5*60)
        fcst_i, obs_i = load_both_data(vrbl,fmt,offsetutc,caseutc,initutc,'all')
        fcst_data[:,tidx,:,:] = fcst_i
        obs_data[tidx,:,:] = obs_i

    return fcst_data, obs_data

def load_latlons(fmt,caseutc):
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    ret = []
    for f in ("lats.npy","lons.npy"):
        fpath = os.path.join(extractroot,fmt,casestr,f)
        ret.append(N.load(fpath))
    return ret

def save_pickle(obj,fpath):
    utils.trycreate(fpath)
    with open(fpath,'wb') as f:
        pickle.dump(obj=obj,file=f)
    return

def load_pickle(fpath):
    with open(fpath,'rb') as f:
        pp = pickle.load(f)
    return pp

def load_obj():
    """ Pass in time, vrbl etc. Return the obj instance.
    """
    pass

def do_pca_plots(pca,PC_df,features,data,fmt):
    ### TEST PLOT 1 ###
    fname = "pca_test_scatter_{}.png".format(fmt)
    fpath = os.path.join(outroot,"pca",fname)

    fig,ax = plt.subplots(1,figsize=(8,8))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    cm = M.cm.plasma
    sca = ax.scatter(PC_df.loc[:,'PC1'],PC_df.loc[:,'PC2'],
                    c=PC_df.loc[:,'PC3'],
                    alpha=0.7,edgecolor='k',linewidth=0.15,cmap=cm)
    plt.colorbar(sca,ax=ax)
    utils.trycreate(fpath)
    fig.savefig(fpath)

    ### PCs in feature space ###
    fname = "pca_test_bar_{}.png".format(fmt)
    fpath = os.path.join(outroot,"pca",fname)
    fig,ax = plt.subplots(1,figsize=(8,8))
    names = []
    scores = []
    for n,pc in enumerate(pca.components_):
        label = "PC #{}".format(n)
        names.append(label)
        scores.append(pc*100)

    bw = 0.2
    pos = N.arange(len(features))
    poss = [pos+(n*bw) for n in (1,0,-1)]

    for n, (pos,label, scorearray) in enumerate(zip(poss,names,scores)):
        ax.barh(pos, scorearray, bw, alpha=0.6, align='center',
                color=RANCOL[n],label=label,
                tick_label=features,)
    ax.set_xlabel("Variance explained (%)")
    ax.set_ylabel("Storm-object characteristic")
    ax.legend()
    utils.trycreate(fpath)
    fig.subplots_adjust(left=0.3)
    fig.savefig(fpath)

    ### Show distribution of PC scores.
    fname = "pca_test_distr_{}.png".format(fmt)
    fpath = os.path.join(outroot,"pca",fname)
    fig,ax = plt.subplots(1,figsize=(8,8))

    ax.hist(data['PC1'],bins=50)
    ax.set_xlabel("PC score")
    ax.set_ylabel("Frequency")
    # ax.legend()
    utils.trycreate(fpath)
    fig.subplots_adjust(left=0.3)
    fig.savefig(fpath)

    ### Show examples of objects
    fname = "pca_test_examples_{}.png".format(fmt)
    fig,ax = plt.subplots(1,figsize=(8,8))
    exarr = N.zeros([40,20])

    # utils.trycreate(fpath)
    #fig.subplots_adjust(left=0.3)
    #W fig.savefig(fpath)

    print("Test plots saved to",os.path.dirname(fpath))
    return

def get_kw(prodfmt,utc,mem=None):
    pca_fname = "pca_model_{}.pickle".format(prodfmt)
    pca_fpath = os.path.join(objectroot,pca_fname)
    if os.path.exists(pca_fpath):
        P = load_pickle(fpath=pca_fpath)
        kw = dict(classify=True,pca=P['pca'],features=P['features'],
                        scaler=P['scaler'])
    else:
        kw = dict()

    if mem is None:
        prod = '_'.join((prodfmt,"obs"))
    else:
        prod = '_'.join((prodfmt,mem))

    kw['prod_code'] = prod
    kw['time_code'] = utc
    return kw

# /home/nothijngrad/Xmas_Shutdown/Xmas/20170501/NEXRAD_nexrad_3km_20170501_0300.npy

### COLOUR SETTINGS ###
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
COLORS = {
        "d01":"#1A85FF",
        "d02":"#D41159",
        "d01_3km":"#1A85FF",
        "d02_1km":"#D41159", # See v v v v v v
        "d02_3km":"#D41159", # Same - differentiated by shape
        }

RANCOL = {
        0:"#DDAA33",
        1:"#BB5566",
        2:"#004488",
        }

MARKERS = {
        "d01_3km":'s',
        "d02_1km":'s',
        "d02_3km":'o', # Different, as it's interpolate
        }
SIZES = {}
ALPHAS = {
        15:1.0,
        25:0.9,
        30:0.75,
        45:0.5,
        50:0.4,
        55:0.2,
        }
alpha = 0.8
size = 25
for fmt in fcst_fmts:
    SIZES[fmt] = size
    ALPHAS[fmt] = alpha

PROD = {
        "d01_3km":1,
        "d02_1km":2,
        "nexrad_3km":3,
        "nexrad_1km":4,
        }

### PROCEDURE ###

# Domains
# fig, ax = plt.subplots(1)
# bmap = create_bmap(50.0,-75.0,-105.0,30.0,ax=ax)

# Plot 4xd01, 4xd02 domains for the 4 cases
# Colour 3km and 1km domains differently


if do_domains:
    fname = "domains.png"
    fpath = os.path.join(outroot,fname)

    with Domains(fpath,figsize=(8,5)) as D:
        for caseutc in CASES.keys():
            casestr = utils.string_from_time('dir',caseutc,strlen='day')
            # for dn, dname in zip( (1,2), ("d01","d02")):
            for dname in ("d01","d02"):
                # d0x_nc = get_random_domain(caseutc,dn)
                # lats = d0x_nc.variables['XLAT'][

                fdir = dname + "_raw"
                lats = N.load(os.path.join(extractroot,fdir,casestr,"lats.npy"))
                lons = N.load(os.path.join(extractroot,fdir,casestr,"lons.npy"))

                name = "_".join((casestr,dname))

                if dname == "d02":
                    label = casestr
                else:
                    label = None

                D.add_domain(name=name,label=label,lats=lats,lons=lons,color=COLORS[dname])
        D.plot_domains()
    print("Domains plot saved to",fpath)



if do_performance:
    # TODO: a zoomed in version where each member is plotted individually as a swarm.
    # For every forecast valid time, load 36 members and 1 obs
    # Parallelise

    def compute_perf(i):
        fcst_vrbl, caseutc, initutc, mem, fcst_fmt, validutc,thresh = i
        fcst_data, obs_data = load_both_data(vrbl=fcst_vrbl,fmt=fcst_fmt,
                        validutc=validutc,caseutc=caseutc,initutc=initutc,
                        mem=mem)
        DS = DetScores(fcst_arr=fcst_data,obs_arr=obs_data,thresh=thresh,
                        overunder='over')
        # print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return DS

    # Create a "master" figure for the paper
    fname = "perfdiag_REFL_comp_30_multi.png"
    fpath = os.path.join(outroot,fname)
    PD0 = Performance(fpath=fpath,legendkwargs=None,legend=True)
    #_ths = (15,30,40,50)
    #_fms = (30,60,90,120,150,180,)
    _vs = ("REFL_comp",)
    _ths = (30,)
    _fms = (60,120,180)

    for thresh, vrbl, fcstmin in itertools.product(_ths,_vs,_fms):
        fname = "perfdiag_{}_{}th_{}min".format(vrbl,thresh,fcstmin)
        fpath = os.path.join(outroot,fname)
        PD1 = Performance(fpath=fpath,legendkwargs=None,legend=True)
        for fmt in fcst_fmts:
            itr = loop_perf(vrbl=vrbl,thresh=thresh,fcstmin=fcstmin,fcst_fmt=fmt)
            itr2 = itertools.tee(itr)
            nrun = len(list(itr2))

            print("About to parallelise over {} grids ({} per member)".format(
                            nrun,nrun//36))

            with multiprocessing.Pool(ncpus) as pool:
                results = pool.map(compute_perf,itr)
            # for i in list(itr):
            #     compute_perf(i)
            POD = []
            FAR = []
            for r in results:
                POD.append(r.get("POD"))
                FAR.append(r.get('FAR'))
            pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                            'alpha':ALPHAS[fmt]}
            lstr = "Domain {}".format(fmt)
            print("Plotting",fmt,"to",fpath)
            # pdb.set_trace()
            PD1.plot_data(pod=N.mean(POD),far=N.mean(FAR),plotkwargs=pk,label=lstr)

            if (fcstmin in (60,120,180)) and (vrbl == "REFL_comp") and (thresh==30):
                if fcstmin > 60:
                    lstr = None
                PD0.plot_data(pod=N.mean(POD),far=N.mean(FAR),plotkwargs=pk,label=lstr)
                if fmt.startswith("d01"):
                    annostr = "{} min".format(fcstmin)
                    PD0.ax.annotate(annostr,xy=(1-N.mean(FAR)-0.055,N.mean(POD)+0.03),
                                    xycoords='data',fontsize='8',color='black',
                                    fontweight='bold')

        # PD.ax.set_xlim([0,0.3])
        PD1.save()
    PD0.save()

if do_efss:
    def compute_efss(i,threshs,spatial_windows,temporal_window):
        fcst_vrbl, caseutc, initutc, fcst_fmt, validutc, = i
        fcst_data, obs_data = load_timewindow_both_data(vrbl=fcst_vrbl,fmt=fcst_fmt,
                        validutc=validutc,caseutc=caseutc,initutc=initutc,
                        mem='all',window=temporal_window)
        efss = FI(xfs=fcst_data,xa=obs_data,thresholds=threshs,ncpus=ncpus,
                        neighborhoods=spatial_windows,temporal_window=temporal_window,
                        efss=True)
        # print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return efss.results

    threshs = (15,30,45,55)
    spatial_windows = {"d01_3km":(1,3,5,7,9,15),
                        "d02_1km":(1,3,9,15,21,27,45)}

    temporal_windows = (1,3,)
    # temporal_windows = (1,)

    #_fms = (30,60,90,120,150,180,)
    # fcstmins = (30,60,90,120,150)
    fcstmins = (150,)
    # fcstmins = (30,90,150)
    # vrbl = "accum_precip"
    # vrbl = "UH02"
    # vrbl = "UH25"
    vrbl = "REFL_comp"

    for fcstmin, temporal_window in itertools.product(fcstmins,temporal_windows):
        efss_data = {}
        fiss_data = {}
        e_npy0 = "d01_3km_efss_{}tw_{}min.npy".format(temporal_window,fcstmin)
        e_npy1 = "d02_1km_efss_{}tw_{}min.npy".format(temporal_window,fcstmin)
        f_npy0 = "d01_3km_fiss_{}tw_{}min.npy".format(temporal_window,fcstmin)
        f_npy1 = "d02_1km_fiss_{}tw_{}min.npy".format(temporal_window,fcstmin)
        if os.path.exists(e_npy0):
            efss_data['d01_3km'] = N.load(e_npy0)
            efss_data['d02_1km'] = N.load(e_npy1)
            fiss_data['d01_3km'] = N.load(f_npy0)
            fiss_data['d02_1km'] = N.load(f_npy1)
        else:
            for fmt in ("d01_3km","d02_1km"):
                itr = loop_efss(vrbl=vrbl,fcstmin=fcstmin,fmt=fmt)

                efss = []
                for i in list(itr):
                    efss.append(compute_efss(i,threshs,spatial_windows[fmt],temporal_window))

                #for vrbl, fmt, fcstmin in itertools.product(_vs,fcst_fmts,_fcms):
                # fname = "fss_{}_{}min".format(vrbl,fcstmin)
                # fpath = os.path.join(outroot,fname)

                # threshold is the line style (dash, solid, dot)
                # fcst_fmt is the colour (raw/raw)

                # [spatial x thresh x all dates]
                nsw = len(spatial_windows[fmt])
                nth = len(threshs)
                # TODO: this is hard coded for the 4 initutcs used for prelim results
                nt = len(fcstmins) * 4

                # efss_data[fmt] = N.zeros([nsw,nth,nt])
                efss_data[fmt] = N.zeros([nsw,nth])
                fiss_data[fmt] = N.zeros([nsw,nth])

                for (thidx,th),(nhidx,nh) in itertools.product(enumerate(threshs),
                                    enumerate(spatial_windows[fmt])):
                    tw = temporal_window
                    efss_load = []
                    fiss_load = []
                    for eidx,e in enumerate(efss):
                        efss_load.append(e[th][nh][tw]["eFSS"])
                        fiss_load.append(e[th][nh][tw]["FISS"])
                    efss_data[fmt][nhidx,thidx] = N.mean(efss_load)
                    fiss_data[fmt][nhidx,thidx] = N.mean(fiss_load)

            N.save(file=e_npy0,arr=efss_data['d01_3km'])
            N.save(file=e_npy1,arr=efss_data['d02_1km'])
            N.save(file=f_npy0,arr=fiss_data['d01_3km'])
            N.save(file=f_npy1,arr=fiss_data['d02_1km'])

        fig,ax = plt.subplots(1)
        fname = "efss_{}min_{}tw.png".format(fcstmin,temporal_window)
        fpath = os.path.join(outroot,fname)
        # Plotted in terms of diameter (3 = 3 grid spaces diameter = 9km for d01)
        for thidx, thresh in enumerate(threshs):
            label = "d02 (1km, raw) {} dBZ".format(thresh)
            ax.plot(spatial_windows["d02_1km"],efss_data["d02_1km"][:,thidx],
                    color=COLORS["d02_1km"],label=label,alpha=ALPHAS[thresh])
        sw3 = [3*s for s in spatial_windows['d01_3km']]
        for thidx, thresh in enumerate(threshs):
            label = "d01 (3km, raw) {} dBZ".format(thresh)
            ax.plot(sw3,efss_data["d01_3km"][:,thidx],
                    color=COLORS["d01_3km"],label=label,alpha=ALPHAS[thresh])
        ax.set_xlim([0,45])
        ax.set_xlabel("Neighborhood diameter (km)")
        ax.set_ylabel("Fractions Skill Score")
        ax.legend()
        fig.savefig(fpath)
        plt.close(fig)

        fig,ax = plt.subplots(1)
        fname = "fiss_{}min_{}tw.png".format(fcstmin,temporal_window)
        fpath = os.path.join(outroot,fname)
        # Plotted in terms of diameter (3 = 3 grid spaces diameter = 9km for d01)
        for thidx, thresh in enumerate(threshs):
            label = "d02 (1km, raw) {} dBZ".format(thresh)
            ax.plot(spatial_windows["d02_1km"],fiss_data["d02_1km"][:,thidx],
                    color=COLORS["d02_1km"],label=label,alpha=ALPHAS[thresh])
        sw3 = [3*s for s in spatial_windows['d01_3km']]
        for thidx, thresh in enumerate(threshs):
            label = "d01 (3km, raw) {} dBZ".format(thresh)
            ax.plot(sw3,fiss_data["d01_3km"][:,thidx],
                    color=COLORS["d01_3km"],label=label,alpha=ALPHAS[thresh])
        ax.set_xlim([0,45])
        ax.set_ylim([-1,1])
        ax.set_xlabel("Neighborhood diameter (km)")
        ax.set_ylabel("Fractional Ignorance Skill Score")
        ax.axhline(0)
        ax.legend()
        fig.savefig(fpath)
        plt.close(fig)



if object_switch:
    # To discriminate between line/cell
    eccentricity = 0.5

    # Plot type
    #plot_what = 'ratio'; plot_dir = 'check_ratio'
    # plot_what = 'qlcs'; plot_dir = 'qlcs_v_cell'
    # plot_what = 'ecc'; plot_dir = "compare_ecc_ratio"
    # plot_what = 'extent'; plot_dir ='check_extent'
    # plot_what = "4-panel"; plot_dir = "four-panel"
    plot_what = "pca"; plot_dir = "pca_check"

    def compute_obj_fcst(i):
        fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem = i

        if validutc == initutc:
            # Forecast is just zeros.
            return None

        # Check for object save file for obs/fcst
        pk_fpath = get_object_picklepaths(fcst_vrbl,fcst_fmt,validutc,caseutc,
                                        initutc=initutc,mem=mem)
        print(stars,"FCST DEBUG:",fcst_fmt,caseutc,initutc,validutc,mem)

        if (not os.path.exists(pk_fpath)) or overwrite_pp:
        #if True:
            dx = 3 if fcst_fmt.startswith("d01") else 1
            print("dx set to",dx)

            fcst_data, fcst_lats, fcst_lons = load_fcst_dll(fcst_vrbl,fcst_fmt,
                                    validutc,caseutc,initutc,mem)

            kw = get_kw(prodfmt=fcst_fmt,utc=validutc,mem=mem)
            obj = ObjectID(fcst_data,fcst_lats,fcst_lons,dx=dx,**kw)
            save_pickle(obj=obj,fpath=pk_fpath)
            print("Object instance newly created.")
        else:
            print("Object instance already created.")
            obj = load_pickle(pk_fpath)

        # QUICK LOOKS
        fcstmin = int(((validutc-initutc).seconds)/60)
        if fcstmin in range(30,210,30):
        #if ((fcstmin % 30) == 0): # and (mem == "m01"):
            ql_fname = "obj_{}_{:%Y%m%d%H}_{}min.png".format(fcst_fmt,initutc,fcstmin)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what,ecc=eccentricity)
            else:
                print("Figure already created")

        return obj.objects

    def compute_obj_obs(i):
        obs_vrbl, obs_fmt, validutc, caseutc = i

        # Check for object save file for obs/fcst
        fpath = get_object_picklepaths(obs_vrbl,obs_fmt,validutc,caseutc)
        print(stars,"OBS DEBUG:",obs_fmt,caseutc,validutc)

        if (not os.path.exists(fpath)) or overwrite_pp:
        #if True:
            print("DEBUG:",obs_fmt,caseutc,validutc)
            dx = 3 if obs_fmt.endswith("3km") else 1
            print("dx set to",dx)

            obs_data, obs_lats, obs_lons = load_obs_dll(validutc,caseutc,
                                obs_vrbl=obs_vrbl,obs_fmt=obs_fmt)
            kw = get_kw(prodfmt=fcst_fmt,utc=validutc)
            obj = ObjectID(obs_data,obs_lats,obs_lons,dx=dx,**kw)
            save_pickle(obj=obj,fpath=fpath)
            print("Object instance newly created.")
        else:
            print("Object instance already created.")
            obj = load_pickle(fpath)

        if validutc.minute == 30:
            ql_fname = "obj_{}_{:%Y%m%d%H%M}.png".format(obs_fmt,validutc)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what,ecc=eccentricity)
            else:
                print("Figure already created")

        return obj.objects


    fcst_vrbl = "REFL_comp"
    obs_vrbl = "NEXRAD"

    # fcstmins = N.arange(30,210,30)
    fcstmins = N.arange(5,185,5)

    fcst_fmts = ("d01_3km","d02_1km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)

    # member_names = ['m{:02d}'.format(n) for n in range(2,3)]

    for fcst_fmt, obs_fmt in zip(fcst_fmts,obs_fmts):
        fcst_fname = "all_obj_dataframes_{}.pickle".format(fcst_fmt)
        fcst_fpath = os.path.join(objectroot,fcst_fname)
        if (not os.path.exists(fcst_fpath)) or overwrite_pp or overwrite_output:
            itr_fcst = loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,member_names)
            if ncpus > 1:
                with multiprocessing.Pool(ncpus) as pool:
                    results_fcst = pool.map(compute_obj_fcst,itr_fcst)
            else:
                for i in itr_fcst:
                    compute_obj_fcst(i)
            save_pickle(obj=results_fcst,fpath=fcst_fpath)
        else:
            results_fcst = load_pickle(fcst_fpath)

        obs_fname = "all_obj_dataframes_{}.pickle".format(obs_fmt)
        obs_fpath = os.path.join(objectroot,obs_fname)
        if (not os.path.exists(obs_fpath)) or overwrite_pp or overwrite_output:
            itr_obs = loop_obj_obs(obs_vrbl,all_times=fcstmins)
            if ncpus > 1:
                with multiprocessing.Pool(ncpus) as pool:
                    results_obs = pool.map(compute_obj_obs,itr_obs)
            else:
                for i in itr_obs:
                    compute_obj_obs(i)
            save_pickle(obj=results_obs,fpath=obs_fpath)
        else:
            results_fcst = load_pickle(obs_fpath)

if do_object_pca:
    # results = []
    for fmt in all_fmts:
        fname = "all_obj_dataframes_{}.pickle".format(fmt)
        fpath = os.path.join(objectroot,fname)
        results = load_pickle(fpath)
        all_df = pandas.concat(results,ignore_index=True)

        fname = "pca_model_{}.pickle".format(fmt)
        fpath = os.path.join(objectroot,fname)

        if (not os.path.exists(fpath)) or overwrite_pp:
            CAT = Catalogue(all_df)
            pca, PC_df, features, scaler = CAT.do_pca()
            save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,features=features),fpath=fpath)
        else:
            P = load_pickle(fpath=fpath)
            pca = P['pca']
            features = P['features']
            scaler = P['scaler']
            PC_df = P['data']

        do_pca_plots(pca,PC_df,features,PC_df,fmt=fmt)
        # TODO: do for objects in d01, d02, obs



    # assert 1==0

if do_object_lda:
    CAT = Catalogue(results)
    lda = CAT.do_lda()
    pdb.set_trace()



if do_object_performance:
    # Do 2x2 table from objects matched
    pass

if do_object_distr:
    # Plot distributions of objects in each domain and obs

    # Then match them
    pass
