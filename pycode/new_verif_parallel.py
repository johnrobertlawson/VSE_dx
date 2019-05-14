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
import itertools
import random
import warnings


# JRL: hack to stop the bloody annoying scikit warnings
# Others might be surpressed too... not always a good idea.
# however, TODO, the warning is, for example:
# https://github.com/scikit-image/scikit-image/issues/3667

def warn(*args, **kwargs):
    pass
warnings.warn = warn

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
import scipy
import pandas
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.stats import rv_histogram
import seaborn as sns

import evac.utils as utils
from evac.datafiles.wrfout import WRFOut
from evac.plot.domains import Domains
from evac.stats.detscores import DetScores
from evac.plot.performance import Performance
# from evac.stats.fi import FI
from fiss import FISS
from evac.object.objectid import ObjectID
from evac.object.catalogue import Catalogue
from evac.plot.scales import Scales

### ARG PARSE ####
parser = argparse.ArgumentParser()

parser.add_argument('-N','--ncpus',dest='ncpus',default=1,type=int)
parser.add_argument('-oo','--overwrite_output',dest='overwrite_output',
                            action='store_true',default=False)
parser.add_argument('-op','--overwrite_pp',dest='overwrite_pp',
                            action='store_true',default=False)
parser.add_argument('-nq','--no_quick',dest="no_quick",
                            action='store_true',default=False)

PA = parser.parse_args()
ncpus = PA.ncpus
overwrite_output = PA.overwrite_output
overwrite_pp = PA.overwrite_pp
do_quicklooks = not PA.no_quick

### SWITCHES ###
do_plot_quicklooks = False

do_domains = False
do_percentiles = False

do_performance = False
do_efss = False # Also includes FISS, which is broken?


# From scratch, do object_switch, do_object_pca, then
# delete the object dfs (in subdirs too), and
# and re-run object_switch to do the MDI of each cell

object_switch = False
do_object_pca = False
do_object_performance = True
do_object_distr = False
do_object_matching = True
do_object_examples = False
do_object_waffle = False
do_object_cluster = False
do_object_brier_uh = True
                #max(do_object_performance,do_object_distr,
                #    do_object_pca,do_object_lda,do_object_matching,)

### SETTINGS ###
#mp_log = multiprocessing.log_to_stderr()
#mp_log.setLevel(logging.INFO)

CASES = collections.OrderedDict()
CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                        datetime.datetime(2016,3,31,19,0,0),
                        datetime.datetime(2016,3,31,20,0,0),
                        datetime.datetime(2016,3,31,21,0,0),
                        datetime.datetime(2016,3,31,22,0,0),
                        datetime.datetime(2016,3,31,23,0,0),
                        ]
CASES[datetime.datetime(2017,5,1,0,0,0)] = [
                        datetime.datetime(2017,5,1,19,0,0),
                        datetime.datetime(2017,5,1,20,0,0),
                        datetime.datetime(2017,5,1,21,0,0),
                        datetime.datetime(2017,5,1,22,0,0),
                        datetime.datetime(2017,5,1,23,0,0),
                        ]
CASES[datetime.datetime(2017,5,2,0,0,0)] = [
                        datetime.datetime(2017,5,2,23,0,0),
                        datetime.datetime(2017,5,3,0,0,0),
                        datetime.datetime(2017,5,3,1,0,0),
                        datetime.datetime(2017,5,3,2,0,0),
                        datetime.datetime(2017,5,3,3,0,0),
                        ]
CASES[datetime.datetime(2017,5,4,0,0,0)] = [
                        datetime.datetime(2017,5,4,22,0,0),
                        datetime.datetime(2017,5,4,23,0,0),
                        datetime.datetime(2017,5,5,0,0,0),
                        datetime.datetime(2017,5,5,1,0,0),
                        datetime.datetime(2017,5,5,2,0,0),
                        ]
# To do - 20180429 (texas panhandle)
# CASES[datetime.datetime(2018,4,29,0,0,0)] = []

### DIRECTORIES ###
# extractroot = "/home/nothijngrad/Xmas_Shutdown/Xmas"
key_pp = 'AprilFool'
#extractroot = '/work/john.lawson/VSE_reso/pp/{}'.format(key_pp)
extractroot = '/Users/john.lawson/data/{}'.format(key_pp)

objectroot = os.path.join(extractroot,'object_instances')
#objectroot = "/Volumes/LaCie/VSE_dx/object_instances"
# outroot = "/home/john.lawson/VSE_dx/pyoutput"
# outroot = "/scratch/john.lawson/VSE_dx/figures"
#outroot = "/work/john.lawson/VSE_dx/figures"
# outroot = "/mnt/jlawson/VSE_dx/figures/"
outroot = "/Users/john.lawson/data/figures"
#tempdir = "/Users/john.lawson/data/intermediate_files"
tempdir = "/Volumes/LaCie/VSE_dx/intermediate_files"

##### OTHER STUFF #####
stars = "*"*10
# ncpus = 8
dom_names = ("d01","d02")
domnos = (1,2)
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
half_member_names = ['m{:02d}'.format(n) for n in range(1,19)]
test_member_names = ['m{:02d}'.format(n) for n in range(1,6)]

# doms = (1,2)
# RAINNC
fcst_vrbls = ("Wmax","UH02","UH25","accum_precip","REFL_comp")
obs_vrbls = ("AWS02","AWS25","DZ","ST4","NEXRAD")
all_fcstmins = N.arange(5,185,5)
# fcst_fmts = ("d01_3km","d02_1km","d02_3km")
fcst_fmts =  ("d01_3km","d02_1km")

NICENAMES = {"d01_3km":"EE3km",
                    "d01_raw":"EE3km",
                    "d02_raw":"EE1km",
                    "d02_1km":"EE1km",
                    "d02_3km":"EE1km-i",}

#### FOR DEBUGGING ####
#CASES = { datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,22,0,0),], }

# fcst_vrbls = ("REFL_comp",)
# fcst_vrbls = ("Wmax",)
# fcst_vrbls = ("REFL_comp","Wmax")
# fcst_vrbls = ("UH25","UP_HELI_MAX")

# obs_vrbls = ("NEXRAD",)
# obs_vrbls = list()
# obs_vrbls = ("AWS25")

#########################
##### OTHER ####
########################

VRBL_CODES = {
                "REFL_comp":"NEXRAD",
                # "REFL_comp":"DZ",
                "UH02":"AWS02",
                "UH25":"AWS25",
                "accum_precip":"ST4",
                }
VRBL_CODES2 = {v:k for k,v in VRBL_CODES.items()}
OBS_CODES = {
                'NEXRAD':'nexrad',
                "AWS02": 'mrms_aws',
                "AWS25":'mrms_aws',
                "ST4":'stageiv',
                "DZ":"mrms_dz",
                }


###############################################################################
################################# THE FUNCTIONS ###############################
###############################################################################
def get_nice(fmt):
    return NICENAMES[fmt]

#from quantile_lookup import PC_LKUP
#def get_list_pc(vrbl,fmt):
#    pcs = sorted(PC_LKUP[vrbl][fmt].keys())
#    vals = (PC_LKUP[vrbl][fmt][p] for p in pcs)
#    return vals

class Threshs:
    def __init__(self,):
        self.threshs = self.get_dict()

    def get_dict(self,):
        lookup = {
                    "REFL_comp":{
                        "3km": (15.0,23.5,34.7,41.9,52.0,63.5),
                        "1km": (15.8,24.8,36.2,43.4,53.7,65.7),
                        },
                    "NEXRAD":{
                        "3km": (14.9,23.5,32.9,38.7,49.0,57.3),
                        "1km": (14.5,23.4,32.8,38.6,49.1,57.5),
                        },
                    "UH02":{
                        "3km": (0.2,1.5,8.1,21.1),
                        "1km": (0.5,4.2,21.7,59.0),
                        },
                    "AWS02":{
                        "1km": (1.7E-3,4.1E-3,7.8E-3,13.8E-3),
                        "3km": (1.7E-3,4.0E-3,7.7E-3,13.2E-3),
                        },
                    "UH25":{
                        "1km": (0.8,9.9,52.7,143.3),
                        "3km": (0.3,3.2,15.4,37.7),
                        },
                    "AWS25":{
                        "1km": (1.9E-3,4.4E-3,8.1E-3,13.5E-3),
                        "3km": (1.9E-3,4.3E-3,7.9E-3,13.2E-3),
                        },
                }
        return lookup

    def get_val(self,vrbl,fmt,pc=None,qt=None):
        assert pc or qt
        dx = self.ensure_fmt(fmt)
        pcidx = self.get_pc_idx(vrbl,pc=pc,qt=qt)
        return self.threshs[vrbl][dx][pcidx]

    def ensure_fmt(self,fmt):
        if fmt in ("d01_3km","d01","3km","lo-res","EE3km",3,"3"):
            return "3km"
        elif fmt in ("d02","d02_raw","d02_1km","1km","hi-res","EE1km",1,"1"):
            return "1km"
        else:
            raise Exception

    def get_pc_idx(self,vrbl,pc=None,qt=None):
        if qt:
            assert 0.0 <= qt <= 1.0
        elif pc:
            assert 0.0 <= pc <= 100.0
            raise NotImplementedError
        else:
            raise Exception

        try:
            x = self.get_quantiles(vrbl).index(qt)
        except:
            print("Maybe a rounding error with floating points")
            x = utils.closest(arr=self.get_quantiles(vrbl),val=qt)
            print("Using",f"{x:.2f}","to represent",qt)
            raise Exception
        return x

    def _get_percentiles(self,vrbl):
        print("Warning: may have rounding error that I need to fix.")
        pc = self.get_quartiles(vrbl) * 100.0
        return pc

    def get_quantiles(self,vrbl):
        # Percentiles used for paper and scoring
        if vrbl in ("AWS02","AWS25","UH02","UH25"):
            return (0.9, 0.99, 0.999,0.9999)
        elif (vrbl in ("REFL_comp","NEXRAD")) or (vrbl.endswith("cut")):
            return (0.7,0.8,0.9,0.95,0.99,0.999)
        else:
            raise Exception

    def get_threshs(self,vrbl,fmt):
        dx = self.ensure_fmt(fmt)
        return self.threshs[vrbl][dx]

# We need to do percentiles here, as we need the info everywhere!
PC_Thresh = Threshs()

def get_extraction_fpaths(vrbl,fmt,validutc,caseutc,initutc=None,mem=None):
    """ Return the file path for the .npy of an interpolated field.

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
    else:
        raise Exception
    return os.path.join(extractroot,caseYYYYMMDD,vrbl,fname)

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

def round_nearest(x,a,method='floor'):
    """
    Args:
        x: original value
        a: multiple to round x to
    """
    FUNCS = {'floor':N.floor,
                'ceil':N.ceil,
                'round':N.round,}
    return FUNCS[method](x/a) * a

def create_bmap(urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon,ax=None,
                    proj="lcc",):
    bmap = Basemap(
                # width=12000000,height=9000000,
                urcrnrlat=urcrnrlat,urcrnrlon=urcrnrlon,
                llcrnrlat=llcrnrlat,llcrnrlon=llcrnrlon,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection=proj,ax=ax,
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
    assert dom in (1,2)
    fname = 'wrfout_d{:02d}_{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(dom,
                t.year,t.month,t.day,t.hour,t.minute,t.second)
    return fname

def fc2ob(fcst_vrbl,fcst_fmt):
    """ Look up the corresponding variable/format name for obs, from vrbls.
    """
    obs_vrbl = VRBL_CODES[fcst_vrbl]

    # This is the dx for the grid (e.g. 1km)
    *_, dx = fcst_fmt.split("_")

    obs_fmt = "_".join((OBS_CODES[obs_vrbl],dx))

    # pdb.set_trace()
    return obs_vrbl,obs_fmt

def ob2fc(obs_vrbl,obs_fmt):
    fcst_vrbl = VRBL_CODES2[obs_vrbl]

    # This is the dx for the grid (e.g. 1km)
    *_, dx = obs_fmt.split("_")

    if dx == "3km":
        fcst_fmt = "d01_3km"
    elif dx == "1km":
        fcst_fmt = "d02_1km"
    else:
        raise Exception

    return fcst_vrbl, fcst_fmt
    # fcst_vrbl, fcst_fmt = ob2fc(obs_vrbl,obs_fmt)

def convert_fmt(fcst_fmt=None,obs_fmt=None,fcst_vrbl=None,obs_vrbl=None):
    if fcst_fmt and obs_vrbl:
        *_, dx = fcst_fmt.split("_")
        obs_fmt = "_".join((OBS_CODES[obs_vrbl],dx))
        return obs_fmt
    else:
        raise Exception

def loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,members,load=None,shuffle=True):
    if shuffle is True:
        f = shuffled_copy
    else:
        f = list

    for mem in f(members):
        for fcstmin in f(fcstmins):
            for caseutc, initutcs in CASES.items():
                for initutc in f(initutcs):
                    validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))
                    yield fcst_vrbl, fcst_fmt, validutc, caseutc, initutc, mem, load

def loop_obj_obs(obs_vrbl,all_times=True):
    if all_times is True:
        ts = fcstmins
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

def loop_efss(caseutc,vrbl,fcstmin,fmt):
    # for caseutc, initutcs in CASES.items():
    for initutc in CASES[caseutc]:
        validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))
        yield vrbl, caseutc, initutc, fmt, validutc

def get_initutcs_strs():
    for caseutc, initutcs in CASES.items():
        for initutc in CASES[caseutc]:
            casestr = f"{caseutc.year:04d}{caseutc.month:02d}{caseutc.day:02d}"
            initstr = f"{initutc.hour:02d}{initutc.minute:02d}"
            # the_str = "_".join(casestr,initstr)
            yield initutc, casestr, initstr
            # validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))

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
    fcst_data = load_fcst_dll(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,
                                mem,return_ll=False)

    # These are the corresponding obs formats/data
    obs_vrbl, obs_fmt = fc2ob(fcst_vrbl,fcst_fmt)
    obs_data = load_obs_dll(validutc,caseutc,obs_vrbl=obs_vrbl,
                            obs_fmt=obs_fmt,return_ll=False)
    # pdb.set_trace()

    return fcst_data, obs_data

def load_fcst_dll(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem,return_ll=True):
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

    ######### JRL: ALL FCST DATA EDITING GOES ON HERE ##########
    if fcst_vrbl in ("UH02","UH25"):
        fcst_data[fcst_data<0.0] = N.nan
    ############################################################

    if return_ll:
        fcst_lats, fcst_lons = load_latlons(fcst_fmt,caseutc)
        return fcst_data, fcst_lats, fcst_lons
    else:
        return fcst_data

def load_obs_dll(validutc,caseutc,obs_vrbl=None,fcst_vrbl=None,obs_fmt=None,
                    fcst_fmt=None,return_ll=True):
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

    ######### JRL: ALL OBS DATA EDITING GOES ON HERE ##########
    if obs_vrbl in ("AWS02","AWS25"):
        obs_data[obs_data<0.0] = N.nan
    ############################################################

    if return_ll:
        obs_lats, obs_lons = load_latlons(fcst_fmt,caseutc)
        return obs_data, obs_lats, obs_lons
    else:
        return obs_data

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
    if fmt == "d02_1km":
        fmt = "d02_raw"
    elif fmt == "d01_raw":
        pass
        # print("Loading raw (uncut?) d01 lat/lons")
    elif fmt == "d01_3km":
        pass
        # print("Loading raw (3km) d01 lat/lons")
    else:
        raise Exception

    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    ret = []
    for f in ("lats.npy","lons.npy"):
        fpath = os.path.join(extractroot,fmt,casestr,f)
        ret.append(N.load(fpath))
    return ret

def load_obj():
    """ Pass in time, vrbl etc. Return the obj instance.
    """
    pass

def do_pca_plots(pca,PC_df,features,fmt):
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

    # JRL TODO: this needs to be pretty for paper
    ### PCs in feature space ###
    for num_feats in (1,3):
        fname = "pca_test_bar_{}.png".format(fmt)
        fpath = os.path.join(outroot,"pca",fname)
        fig,ax = plt.subplots(1,figsize=(8,8))
        names = []
        scores = []
        if num_feats == 1:
            plot_these_components = pca.components_[:1]
        else:
            plot_these_components = pca.components_

        # for n,pc in enumerate(pca.components_):
        # for n,pc in enumerate(pca.components_[:1]):
        for n,pc in enumerate(plot_these_components):
            if n == 0:
                label = "Morph. Index (PC1)"
            else:
                label = "PC{}".format(n)
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

    #ax.hist(data['PC1'],bins=50)
    ax.hist(PC_df['PC1'],bins=50)
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

def get_all_initutcs():
    """ Find all unique initialisation times.
    """
    SET = set()
    for caseutc,initutcs in CASES.items():
        for i in initutcs:
            SET.add(i)
    return list(SET)

def get_kw(prodfmt,utc,mem=None,initutc=None,fpath=None):
    pca_fname = "pca_model_{}.pickle".format(prodfmt)
    pca_fpath = os.path.join(objectroot,pca_fname)
    # print("Determining kw arguments for Object ID...")
    if os.path.exists(pca_fpath):
        # print("Returning a lot of PCA keywords for Object ID")
        P = utils.load_pickle(fpath=pca_fpath)
        kw = dict(classify=True,pca=P['pca'],features=P['features'],
                        scaler=P['scaler'])
    else:
        # print("No classification of mode will be performed")
        kw = dict()

    # pdb.set_trace()
    if mem is None:
        prod = '_'.join((prodfmt,"obs"))
    else:
        prod = '_'.join((prodfmt,mem))

    if initutc is None:
        # Must be obs
        assert mem is None
        fcstmin = 0
    else:
        fcstmin = int(((utc-initutc).total_seconds())/60)

    # JRL TODO: is this where to convert to NICENAMES?
    kw['prod_code'] = prod
    kw['time_code'] = utc
    kw['lead_time'] = fcstmin
    kw['fpath_save'] = fpath

    # print(fcstmin)
    return kw

def load_megaframe(fmts,add_ens=True,add_W=True,add_uh_aws=True,
                    add_resolution=True,add_mode=True,
                    debug_before_uh=False,add_init=True):
    # Overwrite fmts...
    del fmts
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)

    mega_fname = "MEGAFRAME.megaframe"
    mega_fpath = os.path.join(objectroot,mega_fname)
    # ens will create a similar df to megaframe, but separating
    # member and domain from "prod".

    # w adds on updraught information

    if os.path.exists(mega_fpath):
        print("Megaframe loaded.")
        return utils.load_pickle(mega_fpath)

    bypass_compute = False
    if debug_before_uh:
        mega_fname = "test.megaframe"
        mega_fpath = os.path.join(objectroot,mega_fname)
        if os.path.exists(mega_fpath):
            df_og = utils.load_pickle(mega_fpath)
            bypass_compute = True

    if not bypass_compute:
        print("Creating megaframe.")

        df_list = []
        for fmt in all_fmts:
            print("Loading objects from:",fmt)
            fname = "all_obj_dataframes_{}.pickle".format(fmt)
            fpath = os.path.join(objectroot,fname)
            results = utils.load_pickle(fpath)
            if results == "ERROR":
                print("ERROR!")
                pdb.set_trace()
            #df_list.append(pandas.concat(results,ignore_index=True))
            df_og = pandas.concat(results,ignore_index=True)
            # del fname, fpath, results

            #### HACKS #####
            print("Adding/modifying megaframe...")

            # JRL TODO: these should be parallelised, but I'm
            # scared of mixing up the order of things

            # These also save the .pickle files (like
            # W and ens dataframes below) to speed up
            # creating new dfs to hack into the megaframe

            if add_resolution:
                df_og = add_res_to_df(df_og,fmt=fmt)

            if add_mode:
                df_og = add_mode_to_df(df_og,fmt=fmt)

            if add_ens:
                ens_df = load_ens_df(df_og,fmt=fmt)
                # pdb.set_trace()
                df_og = pandas.concat((df_og,ens_df),axis=1)
                # Now we have megaframe_idx!
                print("Megaframe hacked: other ensemble stats added.")

            #if debug_before_uh:
            #    # save this half-baked megaframe.
            #    df_og.to_pickle(mega_fpath)

            # JRL TODO: put in MDI, RDI, Az.shear, QPF stuff!
            # Add on W
            if add_W:
                if fmt.startswith("d0"):
                    CAT = Catalogue(df_og,ncpus=ncpus,tempdir=objectroot)
                    print("Created/updated dataframe Catalogue object.")
                    W_lookup = load_lookup((fmt,),vrbl="Wmax",)#fmts)
                    W_df = load_W_df(W_lookup,CAT,fmt=fmt)
                    df_og = concat_W_df(df_og,W_df,fmt=fmt)
                    print("Megaframe hacked: updraught stats added.")
                else:
                    print("Skipping W stats for obs objects.")

            if add_uh_aws:
                """
                We want exceedence yes/nos for four UH/AWS "standard" values and
                    four percentile values that will vary for:

                    * UH02 v UH25
                    * EE3km v EE1km v AWS (both obs dx sizes using same value regardless)
                """
                if "nexrad" in fmt:
                    _fmt = fmt.replace('nexrad','mrms_aws')
                else:
                    _fmt = fmt

                for layer in ("UH02","UH25"):
                    CAT = Catalogue(df_og,ncpus=ncpus,tempdir=objectroot)
                    print("Created/updated dataframe Catalogue object.")

                    lookup = load_lookup((_fmt,),vrbl=layer)
                    uh_df = load_uh_df(lookup,CAT,layer=layer,fmt=_fmt)
                    #df_og = concat_uh_df(df_og,uh_df)
                    df_og = concat_W_df(df_og,uh_df,fmt=_fmt)
                    print("Megaframe hacked: UH stats added.")

            if add_init:
                df_og = add_initutc_to_df(df_og,fmt=fmt)

            df_list.append(df_og)
        # df_list.append(df_og)

    # pdb.set_trace()
    df_og = pandas.concat(df_list,ignore_index=True)
    # At this point, the megaframe indices are set in stone!
    print("Megaframe created.")

    # Save pickle
    df_og.to_pickle(mega_fpath)
    print("Megaframe saved to disk.")
    return df_og

def load_uh_df(lookup,CAT,layer,fmt):
    # layer determines the lookup of percentile values and absolute values
    assert layer in ("UH02","UH25")

    # fmt determines which of the four datasets the rotation data is from, and
    # hence percentiles
    #assert fmt in all_fmts

    sfx = f"_{layer}_{fmt}"
    fname = f"{layer}_df_{fmt}.pickle"
    fpath = os.path.join(objectroot,fname)
    if not os.path.exists(fpath):
        # uh_df = create_uh_df(df_og)
        # rot_exceed_vals = get_rot_exceed_vals(fmt,layer)
        # rot_exceed_vals = get_list_pc(vrbl,ensure_fmt(fmt))
        rot_exceed_vals = PC_Thresh # Pass in the class!
        uh_df = CAT.compute_new_attributes(lookup,do_suite=layer,
                                            rot_exceed_vals=rot_exceed_vals,
                                            suffix=sfx)
        utils.save_pickle(obj=uh_df,fpath=fpath)
        print("UH DF Saved to",fpath)
    else:
        uh_df = utils.load_pickle(fpath=fpath)
        print("UH metadata DataFrame loaded from",fpath)
    return uh_df

def concat_W_df(df_og,W_df,fmt):
    if W_df is None:
        print("No items in new dataframe, so just returning old one.")
        return df_og
    # TODO: rename, as it can be used for any concatenation, not just W
    new_df = df_og.join(W_df.set_index('megaframe_idx_test'),on='megaframe_idx')
    return new_df

def load_W_df(W_lookup,CAT,fmt):
    fname = f"W_df_{fmt}.pickle"
    fpath = os.path.join(objectroot,fname)
    if not os.path.exists(fpath):
        sfx = f"_W_{fmt}"
        W_df = CAT.compute_new_attributes(W_lookup,do_suite="W",suffix=sfx)
        utils.save_pickle(obj=W_df,fpath=fpath)
        print("Saved to",fpath)
    else:
        W_df = utils.load_pickle(fpath=fpath)
        print("W_df loaded from",fpath)
    return W_df

def load_lookup(fcst_fmts,vrbl):#fmts):
    if len(fcst_fmts) == 1:
        fname = f"{vrbl}_lookup_{fcst_fmts[0]}.pickle"
    else:
        fname = f"{vrbl}_lookup.pickle"
    fpath = os.path.join(objectroot,fname)
    if not os.path.exists(fpath):
        lookup = create_lookup(fcst_fmts,vrbl=vrbl)
        utils.save_pickle(obj=lookup,fpath=fpath)
        print("Saved to",fpath)
    else:
        lookup = utils.load_pickle(fpath=fpath)
        print(vrbl,"lookup loaded from",fpath)
    return lookup

def loop_ens_data(fcst_vrbl,fcst_fmts):
    """ Generates the path to numpy fcst data
    columns: fcst_vrbl, valid_time, fcst_min, prod_code, path_to_pickle
    """

    #for vrbl in ("REFL_comp",):
    for caseutc, initutcs in CASES.items():
        for initutc in initutcs:
            for fcst_fmt in fcst_fmts:
                if (not fcst_fmt.startswith("d0")):
                    obs = True
                    fcst = False
                    if fcst_vrbl.startswith("UH"):
                        fcst_vrbl = fcst_vrbl.replace("UH","AWS")
                    mns = ('obs',)
                    fcmns = (0,)
                else:
                    obs = False
                    fcst = True
                    # FCST
                    mns = member_names
                    fcmns = all_fcstmins
                # pdb.set_trace()
                for mem in mns:
                    if obs:
                        prod_code = "_".join((fcst_fmt, "obs"))
                    elif fcst:
                        prod_code = "_".join((fcst_fmt, mem))
                    else:
                        raise Exception
                    for validmin in fcmns:
                        validutc = initutc+datetime.timedelta(seconds=60*int(validmin))
                        path_to_pickle = get_extraction_fpaths(vrbl=fcst_vrbl,
                                    fmt=fcst_fmt,validutc=validutc,
                                    caseutc=caseutc,initutc=initutc,mem=mem)
                        # JRL TODO: here, you'll want to generalise this to
                        # allow obs loading too, for obs verification/matching
                        # with fcst objects, not just 3-to-1km.
                        #if fcst_vrbl.startswith("UH"):
                        yield dict(fcst_vrbl=fcst_vrbl, valid_time=validutc,
                                fcst_min=validmin, prod_code=prod_code,
                                path_to_pickle=path_to_pickle,fcst_fmt=fcst_fmt)

def create_lookup(fcst_fmts,vrbl):
    itr = list(loop_ens_data(fcst_vrbl=vrbl,fcst_fmts=fcst_fmts))
    nobjs = len(itr)

    DTYPES = {
            "fcst_vrbl":"object",
            "valid_time":"datetime64",
            "fcst_min":"i4",
            "prod_code":"object",
            "path_to_pickle":"object",
            "fcst_fmt":"object",
            }

    new_df = utils.do_new_df(DTYPES,nobjs)

    print(f"Creating {vrbl} lookup.")
    for n,d in enumerate(itr):
        utils.print_progress(total=nobjs,idx=n,every=500)
        for key in DTYPES.keys():
            new_df.loc[n,key] = d[key]

    return new_df

def load_ens_df(df_og,fmt):
    fname = f"ens_df_{fmt}.pickle"
    fpath = os.path.join(objectroot,fname)
    if not os.path.exists(fpath):
        ens_df = create_ens_df(df_og)
        utils.save_pickle(obj=ens_df,fpath=fpath)
        print("Saved to",fpath)
    else:
        ens_df = utils.load_pickle(fpath=fpath)
        print("Ensemble metadata DataFrame loaded from",fpath)
    return ens_df

def create_ens_df(megaframe):
    # TODO: parallelise
    def get_case_code(utc):
        if utc < datetime.datetime(2016,4,1,12,0,0):
            case_code = "20160331"
        elif utc < datetime.datetime(2017,5,2,12,0,0):
            case_code = "20170501"
        elif utc < datetime.datetime(2017,5,3,12,0,0):
            case_code = "20170502"
        elif utc < datetime.datetime(2017,5,5,12,0,0):
            case_code = "20170504"
        #elif utc < datetime.datetime():
        #    case_code = "20180429"
        else:
            raise Exception
        return case_code

    def get_leadtime_group(lt):
        if int(lt) < 62:
            gr = "first_hour"
        elif int(lt) < 122:
            gr = "second_hour"
        elif int(lt) < 182:
            gr = "third_hour"
        else:
            raise Exception
        return gr

    DTYPES = {
            "member":"object",
            "domain":"object",
            "case_code":"object",
            "leadtime_group":"object",
            # "dx":"f4",

            # This allows lookup with megaframe on top level.
            "megaframe_idx":"i4"
            }

    ens_df = utils.do_new_df(DTYPES,len(megaframe))
    # ens_df = N.zeros([len(megaframe),len(DTYPES)])


    print("Creating DataFrame from ensemble metadata.")

    for o in megaframe.itertuples():
    # def f(o):
        oidx = o.Index
        #if o.prod_code.startswith("d"):
        #    dom, dxkm, mem = o.prod_code.split("_")
        #else:
        #    mem, dxkm = o.prod_code.split("_")
        #    dom = 0
        dom, dxkm, mem = o.prod_code.split("_")

        ens_df.loc[oidx,"member"] = mem
        ens_df.loc[oidx,"domain"] = dom
        # ens_df.loc[oidx,"dx"] = float(dxkm[0])
        ens_df.loc[oidx,"megaframe_idx"] = o.Index
        #pdb.set_trace()

        ens_df.loc[oidx,"case_code"] = get_case_code(o.time)
        ens_df.loc[oidx,"leadtime_group"] = get_leadtime_group(o.lead_time)


        # assert True == False
        #ens_df.loc[o,"megaframe_idx"] = o.index
        #ens_df.loc[o,"megaframe_idx"] = o.label
        utils.print_progress(total=len(megaframe),idx=oidx,every=500)
        # return

        #if ncpus > 1:
        #    with multiprocessing.Pool(ncpus) as pool:
        #        results = pool.map(f,megaframe.itertuples())
        #else:
        #    for o in megaframe.itertuples():
        #        f(o)


    return ens_df

def create_uh_df(megaframe):
    # TODO: parallelise
    DTYPES = {
            "member":"object",
            # This allows lookup with megaframe on top level.
            "megaframe_idx":"i4"
            }

    uh_df = utils.do_new_df(DTYPES,len(megaframe))
    print("Creating DataFrame for UH/AWS object metadata.")

    for o in megaframe.itertuples():
        oidx = o.Index

        # ens_df.loc[oidx,"case_code"] = get_case_code(o.time)

        utils.print_progress(total=len(megaframe),idx=oidx,every=500)
    return ens_df

def add_res_to_df(df_og,fmt):
    def compute_res(df_og):
        DTYPES = {"resolution":"object"}
        res_df = utils.do_new_df(DTYPES,len(df_og))
        for oidx,o in enumerate(df_og.itertuples()):
            if int(o.nlats) > 200:
                res = "EE1km"
            else:
                res = "EE3km"
            res_df.loc[oidx,"resolution"] = res
            # pdb.set_trace()
        return res_df

    fpath = os.path.join(objectroot,f"res_df_{fmt}.pickle")
    if not os.path.exists(fpath):
        res_df = compute_res(df_og)
        utils.save_pickle(obj=res_df,fpath=fpath)
        print("Saved to",fpath)
    else:
        res_df = utils.load_pickle(fpath=fpath)
        print("Resolution metadata DataFrame loaded from",fpath)

    df_og = pandas.concat((df_og,res_df),axis=1)
    print("Megaframe hacked: resolution added")
    return df_og

def add_initutc_to_df(df_og,fmt):
    """ Use HHMM for init time."""
    def compute_init(df_og):
        DTYPES = {"init_code":"object"}
        new_df = utils.do_new_df(DTYPES,len(df_og))
        for oidx,o in enumerate(df_og.itertuples()):
            if o.member == "obs":
                ic = None
            else:
                dt = o.time - datetime.timedelta(seconds=60*o.lead_time)
                ic = f"{dt.hour:02d}{dt.minute:02d}"
            new_df.loc[oidx,"init_code"] = ic
            # pdb.set_trace()
        return new_df

    fpath = os.path.join(objectroot,f"init_df_{fmt}.pickle")
    if not os.path.exists(fpath):
        init_df = compute_init(df_og)
        utils.save_pickle(obj=init_df,fpath=fpath)
        print("Saved to",fpath)
    else:
        init_df = utils.load_pickle(fpath=fpath)
        print("Init code metadata DataFrame loaded from",fpath)

    df_og = pandas.concat((df_og,init_df),axis=1)
    print("Megaframe hacked: init code added")
    return df_og

def add_mode_to_df(df_og,fmt):
    def compute_mode(df_og):
        DTYPES = {"conv_mode":"object"}
        mode_df = utils.do_new_df(DTYPES,len(df_og))
        for oidx,o in enumerate(df_og.itertuples()):
            if o.qlcsness < -0.5:
                conv_mode = "cellular"
            elif o.qlcsness > 0.5:
                conv_mode = "linear"
            else:
                conv_mode = "ambiguous"
            mode_df.loc[oidx,"conv_mode"] = conv_mode
        return mode_df

    fpath = os.path.join(objectroot,f"mode_df_{fmt}.pickle")
    if not os.path.exists(fpath):
        mode_df = compute_mode(df_og)
        utils.save_pickle(obj=mode_df,fpath=fpath)
        print("Saved to",fpath)
    else:
        mode_df = utils.load_pickle(fpath=fpath)
        print("Mode metadata DataFrame loaded from",fpath)

    df_og = pandas.concat((df_og,mode_df),axis=1)
    print("Megaframe hacked: mode info added")
    return df_og

def get_color(fmt,mem):

    CCC = {"d01_3km": {
                1:"#71B4FD",
                2:"#469DFF",
                3:"#1A86FF",
                4:"#077CFF",
                5:"#015EC7",
                6:"#000897",},
            "d02_1km": {
                1:"#E66393",
                2:"#DD3C77",
                3:"#D41159",
                4:"#A80441",
                5:"#850031",
                6:"#650025",}}

    mem_no = int(mem[1:])
    assert 0 < mem_no < 37
    mem_combo = (mem_no % 6)+1
    # YSU, Dudhia, RRTM MM5
    # d01 or EE3km is #1A85FF or (26,133,255)
    # d02 or EE1km is #D41159 or (212,17,89)
    return CCC[fmt][mem_combo]

def shuffled_copy(seq):
    l = list(seq)
    return random.sample(l,len(l))



###############################################################################
############################### END OF FUNCTIONS ##############################
###############################################################################

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

RESCOLS = {
        #"lo-res":"#1A85FF",
        #"hi-res":"#D41159",
        "EE3km":"#1A85FF",
        "EE1km":"#D41159",
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

alpha_dbzs = N.arange(10,65,5)
alpha_vals = N.linspace(0.1,1.0,num=len(alpha_dbzs))

# ALPHAS = { 10:1.0, 15:0.9, 20:0.9, 25:0.9, 30:0.75,
                # 45:0.5, 50:0.4, 55:0.2, }
ALPHAS = {d:v for d,v in zip(alpha_dbzs[::-1],alpha_vals)}

alpha = 0.7
size = 20
for fmt in fcst_fmts:
    SIZES[fmt] = size
    ALPHAS[fmt] = alpha

### COLORBAR
lawson_cm = {'red':  ((0.0, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (1.0, 0.4, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.4),
                   (0.5, 1.0, 0.8),
                   (1.0, 0.0, 0.0)),
        'alpha': ((0.0, 1.0, 1.0),
                   (0.5, 0.3, 0.3),
                   (1.0, 1.0, 1.0))
                }


cmap_vse = {'red': (
                (0.0, 26/255, 26/255),
                # (0.25, 26/255, 26/255),
                (0.25, 20/255, 20/255),
                # (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                # (0.75, 212/255, 212/255),
                (0.75, 170/255, 170/255),
                (1.0, 212/255, 212/255)),
            'green': (
                (0.0, 133/255, 133/255),
                # (0.25, 133/255, 133/255),
                (0.25, 90/255, 90/255),
                # (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                # (0.75, 17/255, 17/255),
                (0.75, 12/255, 12/255),
                (1.0, 17/255, 17/255)),
            'blue': (
                (0.0, 1.0, 1.0),
                # (0.25, 1.0, 1.0),
                (0.25, 0.75, 0.75),
                # (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                # (0.75, 89/255, 89/255),
                (0.75, 65/255, 65/255),
                (1.0, 89/255, 89/255)),
            'alpha': (
                (0.0, 1.0, 1.0),
                (0.5, 0.2, 0.2),
                # (0.5, 0.4, 0.4),
                (1.0, 1.0, 1.0)),
            }


# plt.register_cmap(name="vse_diverg",data=lawson_cm)
plt.register_cmap(name="vse_diverg",data=cmap_vse)

PROD = {
        "d01_3km":1,
        "d02_1km":2,
        "nexrad_3km":3,
        "nexrad_1km":4,
        }

# These used for percentile plots
LIMS = {
    "REFL_comp":[0,75],
    "NEXRAD":[0,75],
    "REFL_comp_cut":[0,75],
    "NEXRAD_cut":[0,75],
    #"UH25":[0,400],
    "UH25":[0,150],
    "UH02":[0,60],
    "AWS25":[0.0,0.02],
    "AWS02":[0.0,0.02],
    }

UNITS = {
    # (multiplier, units)
    "REFL_comp":(1,"dBZ"),
    "NEXRAD":(1,"dBZ"),
    "REFL_comp_cut":(1,"dBZ"),
    "NEXRAD_cut":(1,"dBZ"),
    "AWS25":(1000,"10e^{-3} s^{-1}"),
    "AWS02":(1000,"10e^{-3} s^{-1}"),
    "UH25":(1,"m^2 s^{-2}"),
    "UH02":(1,"m^2 s^{-2}"),
    }

MINMAX = {
        "NEXRAD":(-32.0,90.0),
        "REFL_comp":(-35.0,90.0),
        "NEXRAD_cut":(0.0,90.0),
        "REFL_comp_cut":(0.0,90.0),
        "UH25":(0.0,400.0),
        "UH02":(0.0,100.0),
        "AWS25":(0.0,0.02),
        "AWS02":(0.0,0.02),
        }



PC_LKUP = {
            "AWS02":{
                "1km":{
                    0.9:1.7e-3,
                    0.99:4.1e-3,
                    0.999:7.8e-3,
                    0.9999:13.8e-3,
                },
                "3km":{
                    0.9:1.7e-3,
                    0.99:4.0e-3,
                    0.999:7.7e-3,
                    0.9999:13.2e-3,
                }
            },
            "AWS25":{
                "1km":{
                    0.9:1.9e-3,
                    0.99:4.4e-3,
                    0.999:8.1e-3,
                    0.9999:13.5e-3,
                },
                "3km":{
                    0.9:1.9e-3,
                    0.99:4.3e-3,
                    0.999:7.9e-3,
                    0.9999:13.2e-3,
                }
            },
            "UH02":{
                "1km":{
                    0.9:0.5,
                    0.99:4.2,
                    0.999:21.7,
                    0.9999:59.0,
                },
                "3km":{
                    0.9:0.2,
                    0.99:1.5,
                    0.999:8.1,
                    0.9999:21.1,
                }
            },
            "UH25":{
                "1km":{
                    0.9:0.8,
                    0.99:9.9,
                    0.999:52.7,
                    0.9999:143.3,
                },
                "3km":{
                    0.9:0.3,
                    0.99:3.2,
                    0.999:15.4,
                    0.9999:37.7,
                }
            }
        } # END!

"""PERCENTILES:
    * Representative AWS02 values:
    * AWS02 1km: 90% (1.7e-3), 99% (4.1e-3), 99.9% (7.8e-3), 99.99% (13.8e-3)
    * AWS02 3km: 90% (1.7e-3), 99% (4.0e-3), 99.9% (7.7e-3), 99.99% (13.2e-3)
    * AWS25 1km: 90% (1.9e-3), 99% (4.4e-3), 99.9% (8.1e-3), 99.99% (13.5e-3)
    * AWS25 3km: 90% (1.9e-3), 99% (4.3e-3), 99.9% (7.9e-3), 99.99% (13.2e-3)

    * UH02 1km: 90% (0.5), 99% (4.2), 99.9% (21.7), 99.99% (59.0)
    * UH02 3km: 90% (0.2), 99% (1.5), 99.9% (8.1), 99.99% (21.1)
    * UH25 1km: 90% (0.8), 99% (9.9), 99.9% (52.7), 99.99% (143.3)
    * UH25 3km: 90% (0.3), 99% (3.2), 99.9% (15.4), 99.99% (37.7)

    * NEXRAD 1km: 70% (14.5), 80% (23.4), 90% (32.8), 95% (38.6), 99% (49.1), 99.9 (57.5)
    * NEXRAD 3km: 70% (14.9), 80% (23.5), 90% (32.9), 95% (38.7). 99% (49.0), 99.9 (57.3)
    * REFL_comp 1km: 70% (15.8), 80% (24.8), 90% (36.2), 95% (43.4), 99% (53.7), 99.9 (65.7)
    * REFL_comp 3km: 70% (15.0), 80% (23.5), 90% (34.7), 95% (41.9), 99% (52.0), 99.9 (63.5)

    * NEXRAD_cut, REFL_comp (identical to above)

    # JRL: QPF - needs RAIN-H creating/pc computation first
    * ST4 1km
    * ST4 3km
    * RAIN-H 1km
    * RAIN-H 3km


"""

###############################################################################
### PROCEDURE ###
if do_plot_quicklooks:
    fcst_vrbl_1 = "UH02"
    fcst_vrbl_2 = "REFL_comp"

    def gen_4panel(v1,v2):
        for fcst_fmt in ("d01_3km","d02_1km"):
            for caseutc,initutcs in CASES.items():
                for initutc in shuffled_copy(initutcs):
                    for fm in shuffled_copy(all_fcstmins):
                        yield fcst_fmt, caseutc, initutc, fm, v1, v2

    def make_4panel(i):
        fcst_fmt, caseutc, initutc, fm, fcst_vrbl_1, fcst_vrbl_2 = i
        kmstr = "1km" if "1km" in fcst_fmt else "3km"
        fname = "test_{}_{}_{:03d}min_{}.png".format(fcst_vrbl_1,fcst_vrbl_2,
                                            int(fm),kmstr)

        casestr = utils.string_from_time('dir',caseutc,strlen='day')
        initstr = utils.string_from_time('dir',initutc,strlen='hour')
        fpath = os.path.join(outroot,'quicklooks_{}_{}'.format(fcst_vrbl_1,fcst_vrbl_2),
                                casestr,initstr,fname)

        if os.path.exists(fpath):
            print("Already plotted this. Skipping.")
            return

        validutc = initutc+datetime.timedelta(seconds=60*int(fm))
        # fcst_data, obs_data = load_both_data(fcst_vrbl=fcst_vrbl,fcst_fmt=fcst_fmt,
                    # validutc=validutc,caseutc=caseutc,initutc=initutc,mem="m01")
        lats, lons = load_latlons(fcst_fmt,caseutc)

        fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))

        # left figure: obs
        for nax, ax in zip(range(4),axes.flat):
            if nax == 0:
                ax.set_title("Obs (AWS)")
                fcst_vrbl = fcst_vrbl_1
            elif nax == 1:
                ax.set_title("Fcst (UH)")
                fcst_vrbl = fcst_vrbl_1
            elif nax == 2:
                ax.set_title("Obs (NEXRAD)")
                fcst_vrbl = fcst_vrbl_2
            elif nax == 3:
                ax.set_title("Fcst (REFL_comp)")
                fcst_vrbl = fcst_vrbl_2

            fcst_data, obs_data = load_both_data(fcst_vrbl=fcst_vrbl,fcst_fmt=fcst_fmt,
                    validutc=validutc,caseutc=caseutc,initutc=initutc,mem="m01")
            bmap = create_bmap(urcrnrlat=lats.max(),urcrnrlon=lons.max(),
                                llcrnrlat=lats.min(),llcrnrlon=lons.min(),
                                ax=ax,proj="merc")
            # bmap.drawcounties()
            bmap.drawstates()
            x,y = bmap(lons,lats)

            S = Scales('cref')
            if nax == 0:
                kw = dict(alpha=0.9,levels=N.arange(0.0005,0.0105,0.0005))
            elif nax == 1:
                if fcst_vrbl == "UH25":
                    kw = dict(alpha=0.9,levels=N.arange(0.2,100.2,0.2))
                elif fcst_vrbl == "UH02":
                    kw = dict(alpha=0.9,levels=N.arange(0.25,400.25,0.25))
            else:
                kw = dict(levels=N.arange(5,95),cmap=S.cm)

            if nax in (0,2):
                cf = bmap.contourf(x,y,obs_data,**kw)
            else:
                cf = bmap.contourf(x,y,fcst_data,**kw)

        fig.tight_layout()
        utils.trycreate(fpath)
        fig.savefig(fpath)
        print("saved to",fpath)
        plt.close(fig)
        # pdb.set_trace()
        pass

    fvs = (fcst_vrbl_1,fcst_vrbl_2)
    if ncpus > 1:
        with multiprocessing.Pool(ncpus) as pool:
            results = pool.map(make_4panel,gen_4panel(*fvs))
    else:
        for i in gen_4panel(*fvs):
            make_4panel(i)

if do_domains:
    print(stars,"DOING DOMAINS",stars)
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

if do_percentiles:
    pass
    # JRL: create cdfs of cref, qpf, az-shear in EE3, EE1, and obs.
    # Evaluate everything at "key" percentiles, e.g. 90, 95, 99% for extreme stuff

    # Later, we can compute debiased differences for matched objects?

    # TODO: should object ID be based on percentiles too? Or should a storm
    # always be >?? dBZ? Maybe show how different the location of e.g. 45 dBZ is
    # on the two distributions. If they are close, use the same mask threshold.

    # Do 10,001-bin cdf/pdf of each time for obs (3km and 1km) and fcst (EE3, EE1).
    # Compute in parallel then sum over each category to produce four hists:

    def gen_pc_loop(v,kmstr):
        if kmstr == "3km":
            fcst_fmt = "d01_3km"
        elif kmstr == "1km":
            fcst_fmt = "d02_1km"
        else:
            raise Exception


        mem_list = list()
        # if it's a fcst vrbl, yield a list of member names, else tuple("obs").

        # v_og here is the variable, raw, with no processing (e.g. cutting at 0)
        if v.endswith("cut"):
            v_og = v.replace("_cut","")
        else:
            v_og = v

        if v_og in fcst_vrbls:
            mem_list = member_names
            fmt = fcst_fmt
        elif v_og in obs_vrbls:
            obs_fmt = convert_fmt(fcst_fmt=fcst_fmt,obs_vrbl=v_og)
            mem_list = ("obs",)
            fmt = obs_fmt

        for caseutc,initutcs in CASES.items():
            for initutc in shuffled_copy(initutcs):
                for fm in shuffled_copy(all_fcstmins):
                    for mem in mem_list:
                        yield fmt, caseutc, initutc, fm, v, mem

    def pc_func(i):
        fmt, caseutc, initutc, fm, v, mem = i
        validutc = initutc + datetime.timedelta(seconds=60*int(fm))

        # v_og here is the variable, raw, with no processing (e.g. v has cutting at 0)
        if v.endswith("cut"):
            v_og = v.replace("_cut","")
        else:
            v_og = v

        if mem == 'obs':
            data, *_ = load_obs_dll(validutc,caseutc,obs_vrbl=v_og,obs_fmt=fmt)
        elif mem.startswith("m"):
            data, *_ = load_fcst_dll(v_og,fmt,validutc,caseutc,initutc,mem)
        else:
            raise Exception

        data[data<0.0] = 0.0
        nbins = 10001
        hist = N.histogram(data,bins=nbins,range=MINMAX[v],)
        # pdb.set_trace()
        return hist

    def plot_quantiles(RVH,ax,quantiles,mul_units,pad=0.08,vrbl=None):
        multiplier, units = mul_units
        ymax = dir(RVH)

        red_pc = PC_Thresh(vrbl)

        for q in quantiles:
            col = "red" if q in red_pc else "black"
            xval = RVH.ppf(q)
            ax.axvline(xval,color=col)
            if col == "red":
                xval_mul = xval * multiplier
                # ax.annotate("{:.2f}".format(q),xy=(xval,1+pad),color=col,**kws)
                #txt = "{:.1f}%\n{:.1f}{}".format(q*100,xval_mul,units),
                txt = "\n".join((r"${:.1f}\%$".format(q*100),
                            r"${:.1f}$".format(xval_mul)))
                ax.annotate(txt,xy=(xval,1+pad),annotation_clip=False,
                                fontsize=7,fontstyle="italic",color="brown",)
        return ax

    # JRL: generate RAIN-H (hourly RAINNC) to compare with ST4
    # Don't do percentiles, because it's likely over-fitting (only 60 verif times)
    #_vrbls = ("NEXRAD","REFL_comp","AWS02","AWS25","UH02","UH25")
    #_vrbls = ("UH02","UH25","AWS02","AWS25")
    #_vrbls = ["NEXRAD","REFL_comp"]
    _vrbls = ["NEXRAD_cut","REFL_comp_cut"]

    for _vrbl in _vrbls:
        for kmstr in ('3km','1km'):
            pcroot = os.path.join(outroot,"pc_distr",_vrbl,kmstr)

            fname_npy = "pc_distr_{}_{}.npy".format(_vrbl,kmstr)
            fpath_npy = os.path.join(outroot,tempdir,"pc_distr",fname_npy)
            if not os.path.exists(fpath_npy) or overwrite_pp:
                if ncpus > 1:
                    with multiprocessing.Pool(ncpus) as pool:
                        results = pool.map(pc_func,gen_pc_loop(_vrbl,kmstr))
                else:
                    for i in gen_pc_loop(_vrbl,kmstr):
                        pc_func(i)
                # Do merged histogram
                results = N.array(results)
                utils.trycreate(fpath_npy,isdir=False)
                N.save(file=fpath_npy,arr=results)
            else:
                results = N.load(fpath_npy)

            hist = N.sum(results[:,0],axis=0)

            # JRL: not sure why there is a 10001/10002 mismatch in pc bins
            old_bins = results[0,1]
            RVH = rv_histogram((hist,old_bins))

            print(_vrbl,kmstr,N.sum(hist),"points counted/sorted.")

            bins = N.linspace(old_bins.min(),old_bins.max(),num=hist.shape[0])
            #q1 = N.arange(0.1,1.0,0.1)
            #q2 = N.array([0.95,0.99,0.995,0.999,0.9999])
            #quantiles = N.concatenate([q1,q2])
            quantiles = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,
                                0.99,0.995,0.999,0.9999)
            # RVH.cdf(x=quantiles)

            dw = (bins[1] - bins[0])*0.8
            # JRL: shouldn't fit anything - just read off different percentiles
            fname_fig = "pc_histbar_{}_{}.png".format(_vrbl,kmstr)
            fpath_fig = os.path.join(pcroot,fname_fig)
            fig,ax = plt.subplots(1)
            ax.bar(x=bins,height=hist,width=dw,color='green',alpha=0.8)
            ax = plot_quantiles(RVH,ax,quantiles,UNITS[_vrbl],vrbl=_vrbl)
            #ax.set_xticks(bins[::1000])
            #ax.set_xticklabels(["{:.3f}".format(x) for x in bins[::1000]])
            #ax.set_xlim([bins.min(),bins.max()])
            utils.trycreate(fpath_fig,isdir=False)
            fig.savefig(fpath_fig)
            plt.close(fig)

            fname_fig = "pc_cdf_{}_{}.png".format(_vrbl,kmstr)
            fpath_fig = os.path.join(pcroot,fname_fig)
            fpath_fig2 = fpath_fig.replace("_cdf_","_cdfzoom_")

            fig,ax = plt.subplots(1)
            ax.plot(bins,RVH.cdf(bins))
            ax = plot_quantiles(RVH,ax,quantiles,UNITS[_vrbl],vrbl=_vrbl)
            fig.savefig(fpath_fig)
            ax.set_xlim(LIMS[_vrbl])
            # ax.set_ylim([RVH.ppf(50)])
            fig.savefig(fpath_fig2)
            pass


if do_performance:
    print(stars,"DOING PERFORMANCE",stars)

    # TODO: a zoomed in version where each member is plotted individually as a swarm.
    # For every forecast valid time, load 36 members and 1 obs
    # Parallelise

    def compute_perf(i):
        fcst_vrbl, caseutc, initutc, mem, fcst_fmt, validutc,thresh = i
        fcst_data, obs_data = load_both_data(fcst_vrbl=fcst_vrbl,fcst_fmt=fcst_fmt,
                        validutc=validutc,caseutc=caseutc,initutc=initutc,
                        mem=mem)
        DS = DetScores(fcst_arr=fcst_data,obs_arr=obs_data,thresh=thresh,
                        overunder='over')
        # print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return (DS,caseutc,mem)

    # Create a "master" figure for the paper
    fname = "perfdiag_REFL_comp_30_multi.png"
    fpath = os.path.join(outroot,fname)
    PD0 = Performance(fpath=fpath,legendkwargs=None,legend=False)
    #_ths = (15,30,40,50)
    #_fms = (30,60,90,120,150,180,)
    _vs = ("REFL_comp",)
    _ths = (15,30,40,50)
    #_fms = (60,120,180)
    _fms = (30,60,90,120,150,180)

    ZZZ = {}
    YYY = {}
    for thresh, vrbl, fcstmin in itertools.product(_ths,_vs,_fms):
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
            casestrs = [utils.string_from_time('dir',c,strlen='day')
                            for c in CASES.keys()]
            ZZZ[fmt] = dict()
            ZZZ[fmt]["POD"] = {casestr:[] for casestr in casestrs}
            ZZZ[fmt]["FAR"] = {casestr:[] for casestr in casestrs}

            YYY[fmt] = dict()
            YYY[fmt]["POD"] = {casestr:{m:[] for m in member_names} for casestr in casestrs}
            YYY[fmt]["FAR"] = {casestr:{m:[] for m in member_names} for casestr in casestrs}

            for r in results:
                mem = r[2]
                casestr = utils.string_from_time('dir',r[1],strlen='day')

                _pod = r[0].get("POD")
                _far = r[0].get("FAR")

                ZZZ[fmt]["POD"][casestr].append(_pod)
                ZZZ[fmt]["FAR"][casestr].append(_far)

                YYY[fmt]["POD"][casestr][mem].append(_pod)
                YYY[fmt]["FAR"][casestr][mem].append(_far)


        # pdb.set_trace()
        for casestr in casestrs:
            fname1 = "perfdiag_{}_{}th_{}min_{}".format(vrbl,thresh,fcstmin,casestr)
            fpath1 = os.path.join(outroot,"perfdiag",casestr,fname1)
            PD1 = Performance(fpath=fpath1,legendkwargs=None,legend=True)

            fname2 = "perfdiag_{}_{}th_{}min_{}_eachens".format(vrbl,thresh,fcstmin,casestr)
            fpath2 = os.path.join(outroot,"perfdiag","eachens",casestr,fname2)
            PD2 = Performance(fpath=fpath2,legendkwargs=None,legend=False)

            pod_all = []
            far_all = []
            pod_mem = {f:{m:[] for m in member_names} for f in fcst_fmts}
            far_mem = {f:{m:[] for m in member_names} for f in fcst_fmts}
            for fmt in fcst_fmts:
                POD = ZZZ[fmt]["POD"]
                FAR = ZZZ[fmt]["FAR"]

                pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                                'alpha':ALPHAS[fmt]}
                lstr = get_nice(fmt)
                print("Plotting",fmt,"to",fpath1)
                # pdb.set_trace()
                PD1.plot_data(pod=N.mean(POD[casestr]),far=N.mean(FAR[casestr]),plotkwargs=pk,label=lstr)

                #if (fcstmin in (60,120,180)) and (vrbl == "REFL_comp") and (thresh==30):
                if (fcstmin in (30,90,150)) and (vrbl == "REFL_comp") and (thresh==30):
                    if fcstmin > 60:
                        lstr = None
                    PD0.plot_data(pod=N.mean(POD[casestr]),far=N.mean(FAR[casestr]),plotkwargs=pk,
                                        label=lstr)
                    if fmt.startswith("d01"):
                        annostr = "{} min".format(fcstmin)
                        PD0.ax.annotate(annostr,xy=(1-N.mean(FAR[casestr])-0.055,N.mean(POD[casestr])+0.03),
                                        xycoords='data',fontsize='8',color='black',
                                        fontweight='bold')
                for mem in member_names:
                    _pod_ = YYY[fmt]["POD"][casestr][mem]
                    pod_all.append(_pod_)
                    _far_ = YYY[fmt]["FAR"][casestr][mem]
                    far_all.append(_far_)

                    pod_mem[fmt][mem].append(_pod_)
                    far_mem[fmt][mem].append(_far_)

            pod_max = N.nanmax(pod_all)
            pod_min = N.nanmin(pod_all)
            sr_min  = 1-N.nanmax(far_all)
            sr_max = 1-N.nanmin(far_all)
            pad = 0.1

            sr_min = max(0,sr_min)
            sr_max = min(1,sr_max)
            pod_max = min(1,pod_max)
            pod_min = max(0,pod_min)
            PD2.ax.set_ylim([pod_min-pad,pod_max+pad])
            PD2.ax.set_xlim([sr_min-pad,sr_max+pad])

            for fmt in fcst_fmts:
                for mem in member_names:
                    POD_ea = YYY[fmt]["POD"][casestr][mem]
                    FAR_ea = YYY[fmt]["FAR"][casestr][mem]

                    pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                                    'alpha':0.4}
                    PD2.plot_data(pod=N.mean(POD_ea),far=N.mean(FAR_ea),
                                        plotkwargs=pk, label=lstr)


            PD2.save()
            PD1.save()
        # pdb.set_trace()

        # PD.ax.set_xlim([0,0.3])
        fname3 = "perfdiag_{}_{}th_{}min_mean_ens".format(vrbl,thresh,fcstmin)
        fpath3 = os.path.join(outroot,"perfdiag","eachens",fname3)
        PD3 = Performance(fpath=fpath3,legendkwargs=None,legend=False)
        _podmms = []
        _farmms = []
        for fmt in fcst_fmts:
            for mem in member_names:
                annostr = "".format(mem)
                pod_mem_mean = N.nanmean(pod_mem[fmt][mem])
                far_mem_mean = N.nanmean(far_mem[fmt][mem])

                _podmms.append(pod_mem_mean)
                _farmms.append(far_mem_mean)

        pod_max = N.nanmax(_podmms)
        pod_min = N.nanmin(_podmms)
        sr_min  = 1-N.nanmax(_farmms)
        sr_max = 1-N.nanmin(_farmms)
        pad = 0.04
        apad = 0.01

        sr_min = max(0,sr_min)
        sr_max = min(1,sr_max)
        pod_max = min(1,pod_max)
        pod_min = max(0,pod_min)
        PD3.ax.set_ylim([pod_min-pad,pod_max+pad])
        PD3.ax.set_xlim([sr_min-pad,sr_max+pad])

        for fmt in fcst_fmts:
            for mem in member_names:
                annostr = "{}".format(mem)
                pod_mem_mean = N.nanmean(pod_mem[fmt][mem])
                far_mem_mean = N.nanmean(far_mem[fmt][mem])
                color_phys = get_color(fmt=fmt,mem=mem)
                pk = {'marker':MARKERS[fmt],'c':color_phys,'s':SIZES[fmt],
                                'alpha':0.4}
                PD3.plot_data(pod=pod_mem_mean,far=far_mem_mean,
                                    plotkwargs=pk, label=lstr)
                PD3.ax.annotate(annostr,xy=(1-far_mem_mean-apad,pod_mem_mean+apad),
                                xycoords='data',fontsize='6',color='black',
                                fontweight='bold')
        PD3.ax.set_ylim([pod_min-pad,pod_max+pad])
        PD3.ax.set_xlim([sr_min-pad,sr_max+pad])

        PD3.save()
    PD0.save()

if do_efss:
    print(stars,"DOING EFSS",stars)
    def compute_efss(i,spatial_windows,temporal_window):
        fcst_vrbl, caseutc, initutc, fcst_fmt, validutc, = i
        threshs_fcst = PC_Thresh.get_threshs(fcst_vrbl,fmt)
        obs_vrbl, _obs_fmt = fc2ob(fcst_vrbl=fcst_vrbl,fcst_fmt=fcst_fmt)
        threshs_obs = PC_Thresh.get_threshs(obs_vrbl,fmt)
        fcst_data, obs_data = load_timewindow_both_data(vrbl=fcst_vrbl,
                        fmt=fcst_fmt,validutc=validutc,caseutc=caseutc,
                        initutc=initutc,mem='all',window=temporal_window)
        efss = FISS(xfs=fcst_data,xa=obs_data,thresholds_obs=threshs_obs,
                        thresholds_fcst=threshs_fcst,ncpus=ncpus,efss=True,
                        neighborhoods=spatial_windows,
                        temporal_window=temporal_window,)
        # print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return efss.results

    # JRL TODO: change eFSS/FISS threshold variable so it is
    # independent (obs/fcst) and can be used to do a
    # percentile evaluation, for instance

    #threshs = (10,20,30,35,40,45,50,55)
    #threshs = (10,20,30,40,45,50)

    # do i also need fcst_qts?

    spatial_windows = {
                        # "d01_3km":(1,3,5,7,9,11,13),
                        "d01_3km":(1,3,5,7,9,),
                        # "d02_1km":(1,3,9,15,21,27,33,39)}
                        "d02_1km":(1,3,9,15,21,27,)}

    temporal_windows = (1,3)
    # temporal_windows = (1,)

    #_fms = (30,60,90,120,150,180,)
    # fcstmins = (30,60,90,120,150,180)
    fcstmins = (30,90,150)
    # fcstmins = (150,)
    # fcstmins = (30,90,150)
    # vrbl = "accum_precip"
    # vrbl = "UH02"
    # vrbl = "UH25"
    # vrbl = "REFL_comp"
    vrbl = "UH25"
    fssdir = os.path.join(extractroot,"efss_fiss",vrbl)
    utils.trycreate(fssdir,isdir=True)

    obs_qts = PC_Thresh.get_quantiles(vrbl)

    for fcstmin, temporal_window,caseutc, in itertools.product(
                fcstmins,temporal_windows,CASES.keys()):
        casestr = utils.string_from_time('dir',caseutc,strlen='day')

        print("Doing {} min, {} tw, for case {}.".format(
            fcstmin, temporal_window, casestr))
        efss_data = {}
        fiss_data = {}
        e_npy0_f = "d01_3km_efss_{}_{}tw_{}min.npy".format(casestr,temporal_window,fcstmin)
        e_npy1_f = "d02_1km_efss_{}_{}tw_{}min.npy".format(casestr,temporal_window,fcstmin)
        f_npy0_f = "d01_3km_fiss_{}_{}tw_{}min.npy".format(casestr,temporal_window,fcstmin)
        f_npy1_f = "d02_1km_fiss_{}_{}tw_{}min.npy".format(casestr,temporal_window,fcstmin)

        e_npy0 = os.path.join(fssdir,e_npy0_f)
        e_npy1 = os.path.join(fssdir,e_npy1_f)
        f_npy0 = os.path.join(fssdir,f_npy0_f)
        f_npy1 = os.path.join(fssdir,f_npy1_f)

        if os.path.exists(e_npy0):
            efss_data['d01_3km'] = N.load(e_npy0)
            efss_data['d02_1km'] = N.load(e_npy1)
            fiss_data['d01_3km'] = N.load(f_npy0)
            fiss_data['d02_1km'] = N.load(f_npy1)
        else:
            for fmt in ("d01_3km","d02_1km"):
                threshs_fcst = PC_Thresh.get_threshs(vrbl,fmt)
                threshs_obs = PC_Thresh.get_threshs(vrbl,fmt)
                itr = loop_efss(caseutc,vrbl=vrbl,fcstmin=fcstmin,fmt=fmt)

                efss = []
                for i in list(itr):
                    efss.append(compute_efss(i,spatial_windows[fmt],temporal_window))

                #for vrbl, fmt, fcstmin in itertools.product(_vs,fcst_fmts,_fcms):
                # fname = "fss_{}_{}min".format(vrbl,fcstmin)
                # fpath = os.path.join(outroot,fname)

                # threshold is the line style (dash, solid, dot)
                # fcst_fmt is the colour (raw/raw)

                # [spatial x thresh x all dates]
                nsw = len(spatial_windows[fmt])
                nth = len(obs_qts)
                # TODO: this is hard coded for the 4 initutcs used for prelim results
                nt = len(fcstmins) * 4

                # efss_data[fmt] = N.zeros([nsw,nth,nt])
                efss_data[fmt] = N.zeros([nsw,nth])
                fiss_data[fmt] = N.zeros([nsw,nth])

                for (thidx,qt),(nhidx,nh) in itertools.product(enumerate(obs_qts),
                                    enumerate(spatial_windows[fmt])):
                    tw = temporal_window
                    efss_load = []
                    fiss_load = []
                    # def get_val(vrbl,fmt,pc=None,qt=None):
                    th = PC_Thresh.get_val(qt=qt,fmt=fmt,vrbl=vrbl)
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
        fname = "efss_{}_{}min_{}tw.png".format(casestr,fcstmin,temporal_window)
        fpath = os.path.join(outroot,"efss",vrbl,fname)
        utils.trycreate(fpath)
        # Plotted in terms of diameter (3 = 3 grid spaces diameter = 9km for d01)
        def efss_subloop(vrbl,fmt,obs_qts=obs_qts):
            # Get alpha, which is darker for higher percentiles
            if vrbl in ("REFL_comp",):
                alphas = (0.25,0.40,0.55,0.70,0.85,1.00)
                unit = 'dBZ'
            elif vrbl in ("UH02","UH25"):
                alphas = (0.25,0.50,0.75,1.00)
                unit = 'm2/s2'
            else:
                raise Exception

            count = 0
            for th,al in zip(sorted(obs_qts),alphas):
                fcst_val = PC_Thresh.get_val(qt=th,vrbl=vrbl,fmt=fmt)
                obs_vrbl, obs_fmt = fc2ob(vrbl,fmt)
                # obs_val = PC_Thresh.get_val(qt=th,vrbl=obs_vrbl,fmt=obs_fmt)
                obs_val = PC_Thresh.get_val(qt=th,vrbl=obs_vrbl,fmt=fmt)
                labroot = ": {:.1f} pc: ({}/{} {})".format(th*100,fcst_val,obs_val,unit)
                yield count, th, labroot,al
                count += 1

        for thidx, thresh, labroot,alp in efss_subloop(vrbl,"d02_1km"):
            label = "EE1km"+labroot
            ax.plot(spatial_windows["d02_1km"],efss_data["d02_1km"][:,thidx],
                    color=COLORS["d02_1km"],label=label,alpha=alp)
        sw3 = [3*s for s in spatial_windows['d01_3km']]
        for thidx, thresh, labroot,alp in efss_subloop(vrbl,"d01_3km"):
            label = "EE3km"+labroot
            ax.plot(sw3,efss_data["d01_3km"][:,thidx],
                    color=COLORS["d01_3km"],label=label,alpha=alp)
        ax.set_xlim([0,30])
        ax.set_ylim([0,1])
        ax.set_xlabel("Neighborhood diameter (km)")
        ax.set_ylabel("Fractions Skill Score")
        ax.legend(prop=dict(size=8),bbox_to_anchor=(1.05,1),
                    loc="upper left",borderaxespad=0.0)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

        fig,ax = plt.subplots(1)
        fname = "fiss_{}_{}min_{}tw.png".format(casestr,fcstmin,temporal_window)
        fpath = os.path.join(outroot,"fiss",vrbl,fname)
        utils.trycreate(fpath)
        # Plotted in terms of diameter (3 = 3 grid spaces diameter = 9km for d01)
        for thidx, thresh, labroot, alp in efss_subloop(vrbl,"d02_1km"):
            label = "EE1km"+labroot
            ax.plot(spatial_windows["d02_1km"],fiss_data["d02_1km"][:,thidx],
                    color=COLORS["d02_1km"],label=label,alpha=alp)
        sw3 = [3*s for s in spatial_windows['d01_3km']]
        for thidx, thresh,labroot,alp in efss_subloop(vrbl,"d01_3km"):
            label = "EE3km"+labroot
            ax.plot(sw3,fiss_data["d01_3km"][:,thidx],
                    color=COLORS["d01_3km"],label=label,alpha=alp)
        ax.set_xlim([0,30])
        ax.set_ylim([-2,1])
        ax.set_xlabel("Neighborhood diameter (km)")
        ax.set_ylabel("Fractional Ignorance Skill Score")
        ax.axhline(0,color='k')
        ax.legend(prop=dict(size=6),bbox_to_anchor=(1.05,1),
                    loc="upper left",borderaxespad=0.0)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

        # Do heatmap-style plots here showing diffs per percentile
        # Use functions, and replace above with functions

        # The darker the square/pixel, the more that EE was better
        # Normalised by entire paper
        # Split by case
        # Show with and without time window

        # Show for REFL_comp and az shear. Maybe QPF

        ### EFSS DIFFS HEATMAP ###
        # 6 percentiles
        # 6 spatial thresholds (one just for 1km; others lining up like 9 <-> 27)
        # 2 time windows (separate subplots)
        # Custom colour bar, normalised by differences across all, mirrored at 0 (white)
        # lawson_cm

        for score in ("efss","fiss"):
            fname = "{}-diffs_{}_{}min_{}tw.png".format(score,casestr,fcstmin,temporal_window)
            fpath = os.path.join(outroot,"{}-diffs".format(score),vrbl,fname)
            utils.trycreate(fpath,isdir=False)

            fig,ax = plt.subplots(1)

            # y-axis is obs_qts
            # x-axis is neighbourhoods

            if score == "efss":
                _diffs = efss_data['d02_1km'][1:,:] - efss_data['d01_3km']
                kw = dict(vmin=-0.1,vmax=0.1)
            elif score == "fiss":
                _diffs = fiss_data['d02_1km'][1:,:] - fiss_data['d01_3km']
                kw = dict(vmin=-_diffs.max(), vmax=_diffs.max())

            diffs = _diffs.T
            im = ax.imshow(diffs,cmap='vse_diverg', **kw)

            nx,ny = diffs.shape
            ax.set_yticks(N.arange(nx))
            ax.set_xticks(N.arange(ny))

            neigh_ticks = spatial_windows['d02_1km'][1:]
            ax.set_xticklabels(neigh_ticks)
            ax.set_xlabel("Neighborhood diameter (km)")

            if vrbl == "REFL_comp":
                qt_labs = ("0.7","0.8","0.9","0.95","0.99","0.999")
            elif vrbl in ("UH02","UH25"):
                qt_labs = ("0.9","0.99","0.999","0.9999")
            else:
                raise Exception
            ax.set_yticklabels(qt_labs)
            ax.set_ylabel("Quantile to exceed")

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            for i,j in itertools.product(range(nx),range(ny)):
                t = "{:.2f}".format(diffs[i,j])
                text = ax.text(j,i,t,ha="center",va="center",color="k",size=7,
                                    fontweight="bold")
            plt.colorbar(im,ax=ax)
            fig.tight_layout()

            fig.savefig(fpath)
            # pdb.set_trace()
        # pdb.set_trace()
        pass
    pass



if object_switch:
    print(stars,"DOING OBJECT COMPUTATIONS",stars)
    # Plot type
    #plot_what = 'ratio'; plot_dir = 'check_ratio'
    # plot_what = 'qlcs'; plot_dir = 'qlcs_v_cell'
    # plot_what = 'ecc'; plot_dir = "compare_ecc_ratio"
    # plot_what = 'extent'; plot_dir ='check_extent'
    # plot_what = "4-panel"; plot_dir = "four-panel"
    plot_what = "pca"; plot_dir = "pca_check"

    def compute_obj_fcst(i):
        fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem, load = i

        #if validutc == initutc:
            # Forecast is just zeros.
        #    return None

        # Check for object save file for obs/fcst
        pk_fpath = get_object_picklepaths(fcst_vrbl,fcst_fmt,validutc,caseutc,
                                        initutc=initutc,mem=mem)
        print(stars,"FCST DEBUG:",fcst_fmt,caseutc,initutc,validutc,mem)

        if "3km" in fcst_fmt:
            dx = 3.0
        elif "1km" in fcst_fmt:
            dx = 1.0
        else:
            raise Exception
        if (not os.path.exists(pk_fpath)) or overwrite_pp:

            fcst_data, fcst_lats, fcst_lons = load_fcst_dll(fcst_vrbl,fcst_fmt,
                                    validutc,caseutc,initutc,mem)

            kw = get_kw(prodfmt=fcst_fmt,utc=validutc,mem=mem,
                        initutc=initutc,fpath=pk_fpath)
            obj = ObjectID(fcst_data,fcst_lats,fcst_lons,dx=dx,**kw)
            utils.save_pickle(obj=obj,fpath=pk_fpath)
            # print("Object instance newly created.")
        elif (load is False):
            return
        else:
            # print("Object instance already created.")

            # JRL TODO: if this keeps breaking, check for a return of
            # a string of "ERROR"? And then delete that file,
            # and it gets recomputed (i.e. go back to above)
            obj = utils.load_pickle(pk_fpath)

            if obj == "ERROR":
                print("Overwriting",pk_fpath)
                fcst_data, fcst_lats, fcst_lons = load_fcst_dll(fcst_vrbl,fcst_fmt,
                                        validutc,caseutc,initutc,mem)

                kw = get_kw(prodfmt=fcst_fmt,utc=validutc,mem=mem,initutc=initutc,
                            fpath=pk_fpath)
                obj = ObjectID(fcst_data,fcst_lats,fcst_lons,dx=dx,**kw)
                utils.save_pickle(obj=obj,fpath=pk_fpath)

        # QUICK LOOKS
        fcstmin = int(((validutc-initutc).seconds)/60)
        if fcstmin in range(30,210,30) and do_quicklooks and (mem == "m01"):
        #if ((fcstmin % 30) == 0): # and (mem == "m01"):
            ql_fname = "obj_{}_{:%Y%m%d%H}_{}min.png".format(fcst_fmt,initutc,fcstmin)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what)
            else:
                #print("Figure already created")
                pass

        return obj.objects

    def compute_obj_obs(i):
        obs_vrbl, obs_fmt, validutc, caseutc = i

        # Check for object save file for obs/fcst
        fpath = get_object_picklepaths(obs_vrbl,obs_fmt,validutc,caseutc)
        # print(stars,"OBS DEBUG:",obs_fmt,caseutc,validutc)

        if (not os.path.exists(fpath)) or overwrite_pp:
            print("DEBUG:",obs_fmt,caseutc,validutc)
            if "3km" in obs_fmt:
                dx = 3.0
            elif "1km" in obs_fmt:
                dx = 1.0
            else:
                raise Exception

            obs_data, obs_lats, obs_lons = load_obs_dll(validutc,caseutc,
                                obs_vrbl=obs_vrbl,obs_fmt=obs_fmt)
            kw = get_kw(prodfmt=obs_fmt,utc=validutc,fpath=fpath)
            obj = ObjectID(obs_data,obs_lats,obs_lons,dx=dx,**kw)
            utils.save_pickle(obj=obj,fpath=fpath)
            # print("Object instance newly created.")
        else:
            # print("Object instance already created.")
            obj = utils.load_pickle(fpath)

        if validutc.minute == 30 and do_quicklooks:
            ql_fname = "obj_{}_{:%Y%m%d%H%M}.png".format(obs_fmt,validutc)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what)
            else:
                # print("Figure already created")
                pass

        return obj.objects


    fcst_vrbl = "REFL_comp"
    obs_vrbl = "NEXRAD"

    # fcstmins = N.arange(30,210,30)
    # fcstmins = N.arange(0,185,5)
    fcstmins = all_fcstmins

    fcst_fmts = ("d01_3km","d02_1km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)

    # member_names = ['m{:02d}'.format(n) for n in range(2,3)]

    # Append the fcst dataframes for "all object" PCA
    fcst_dfs = []
    for fcst_fmt, obs_fmt in zip(fcst_fmts,obs_fmts):
        print("Now calculating objects for all forecasts on",fcst_fmt)
        fcst_fname = "all_obj_dataframes_{}.pickle".format(fcst_fmt)
        fcst_fpath = os.path.join(objectroot,fcst_fname)

        if (not os.path.exists(fcst_fpath)) or overwrite_pp or overwrite_output:
            # First, check all have been created - don't load if not
            # itr_fcst_1 = loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,member_names,load=False,shuffle=False)
            # if ncpus > 1:
                # with multiprocessing.Pool(ncpus) as pool:
                    # results_fcst = pool.map(compute_obj_fcst,itr_fcst_1)
            # else:
                # for i in itr_fcst_1:
                    # compute_obj_fcst(i)
            # utils.save_pickle(obj=results_fcst,fpath=fcst_fpath)

            # Then, load.
            itr_fcst_2 = loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,member_names,load=True)
            if ncpus > 1:
                with multiprocessing.Pool(ncpus) as pool:
                    results_fcst = pool.map(compute_obj_fcst,itr_fcst_2)
            else:
                for i in itr_fcst_2:
                    compute_obj_fcst(i)
            utils.save_pickle(obj=results_fcst,fpath=fcst_fpath)
        else:
            results_fcst = utils.load_pickle(fcst_fpath)


        # pdb.set_trace()
        fname = "pca_model_{}.pickle".format(fcst_fmt)
        fpath = os.path.join(objectroot,fname)
        if (not os.path.exists(fpath)) or overwrite_pp:
            fcst_df = pandas.concat(results_fcst,ignore_index=True)
            CAT = Catalogue(fcst_df,tempdir=objectroot,ncpus=ncpus)
            pca, PC_df, features, scaler = CAT.do_pca()
            utils.save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,features=features),fpath=fpath)

            fcst_dfs.append(fcst_df)
            del fcst_df

        print("Now calculating objects for all observations on",obs_fmt)
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
            utils.save_pickle(obj=results_obs,fpath=obs_fpath)
        else:
            results_obs = utils.load_pickle(obs_fpath)


        fname = "pca_model_{}.pickle".format(obs_fmt)
        fpath = os.path.join(objectroot,fname)
        if (not os.path.exists(fpath)) or overwrite_pp:
            obs_df = pandas.concat(results_obs,ignore_index=True)
            CAT = Catalogue(obs_df,tempdir=objectroot,ncpus=ncpus)
            pca, PC_df, features, scaler = CAT.do_pca()
            utils.save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,features=features),fpath=fpath)

    # Do combined PCA - JRL TODO FIX!
    do_combined = False
    if do_combined:
        # Get data
        fname = {}
        fpath = {}

        fname['new'] = "pca_all_fcst_objs.pickle"
        fpath['new'] = os.path.join(objectroot,fname['new'])

        all_features = ['area','eccentricity','extent','max_intensity',
                        'mean_intensity','perimeter','longaxis_km',
                        # JRL TODO: more features to discriminate
                        # between the two domains
                        'max_updraught','ud_distance_from_centroid',
                        'mean_updraught',
                        # JRL TODO: az shear! QPF!
                        # Will have to hack the megaframe more
                        ]
        if (not os.path.exists(fpath['new'])) or overwrite_pp:
            # fname['1km'] = "pca_model_d02_1km.pickle"
            fname['1km'] = "all_obj_dataframes_d02_1km.pickle"
            fpath['1km'] = os.path.join(objectroot,fname['1km'])

            fname['3km'] = "all_obj_dataframes_d01_3km.pickle"
            fpath['3km'] = os.path.join(objectroot,fname['3km'])

            results = []
            for k in ('1km','3km'):
                data = utils.load_pickle(fpath[k])
                # pdb.set_trace()
                data2 = pandas.concat(data,ignore_index=True)
                results.append(data2)

            # allobj_df = pandas.concat(fcst_dfs,ignore_index=True)
            all_fcst_obj_df = pandas.concat(results,ignore_index=True)
            CAT = Catalogue(all_fcst_obj_df,tempdir=objectroot,ncpus=ncpus)
            pca, PC_df, features, scaler = CAT.do_pca(all_features)
            utils.save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,features=features),fpath=fpath['new'])

if do_object_pca:
    print(stars,"DOING PCA",stars)
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts) #+ ['all',]
    for fmt in all_fmts:
        if fmt == "all":
            fname = "pca_all_fcst_objs.pickle"
        else:
            fname = "pca_model_{}.pickle".format(fmt)
        fpath = os.path.join(objectroot,fname)
        P = utils.load_pickle(fpath=fpath)

        pca = P['pca']
        features = P['features']
        scaler = P['scaler']
        PC_df = P['data']
        do_pca_plots(pca,PC_df,features,fmt)

        # JRL: the QLCS-ness will be called:
        # Morphology Discrimination Index (MDI)

if do_object_performance:
    print(stars,"DOING OBJ PERFORMANCE",stars)
    # Load one big df with all objects
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)

    CAT = Catalogue(megaframe,tempdir=objectroot,ncpus=ncpus)

    # megaframe = load_megaframe()

    # Match objects between domains (EE3 <-> verif at 3km, etc)
    # Build big contigency table (need time, case, domain, etc)

    # Calculate eFSS at all four UH levels + pc'iles, again for dBZ
    # This enables a probabilistic evaluation despite different thresholds etc
    # Also enables us to evaluate at certain times with a time window
    # This avoids any logic with persistent UH
    # We could also go through each member by time, and ID persistent UH swaths
    # Has the issue of time - it is all lost/bunched together
    # Instead can do eFSS w/ 1, 3, 5 time windows and small-ish spatial windows

    # Also plot performance diagram for same data

    # Also test OFV?

    #######################

    # To start, get all matched object pairs (3-to-1).
    # For each matched pair, try to connect obs to the respective grids
    # Once this is done, we can go through all "paired pairs"

    # Also, go through all fcst objects, paired or not, and match to obs
    # Then do the same for obs objects, and match to fcst
    # Can then build a contingency table - see Skinner 2018 also.

    # pdb.set_trace()

    MATCH = {}
    # Create pickle of matches
    # for dom_code in ('d01_3km','d02_1km'):
    mns = half_member_names
    for member in mns:
        for dom_code in ('d02_1km','d01_3km'):
            if member == mns[0]:
                MATCH[dom_code] = {}
            dom, res = dom_code.split('_')
            obs_prod_code = f"nexrad_{res}_obs"
            #for member in member_names:
            MATCH[dom_code][member] = {}
            print("Now verifying objects for",member,dom_code)
            for initutc, casestr, initstr in get_initutcs_strs():
                MATCH[dom_code][member][casestr] = {}
                MATCH[dom_code][member][casestr][initstr] = {}
                fname = f"{dom}_{member}-verif_{casestr}_{initstr}.pickle"
                fpath = os.path.join(objectroot,"verif_match_tuples",casestr,initstr,fname)
                fpath2 = os.path.join(objectroot,"verif_match_2x2",casestr,initstr,fname)
                if not os.path.exists(fpath):
                    fcst_prod_code = '_'.join((dom_code,member))
                    dictA = {"prod_code":fcst_prod_code,"member":member,
                                    "case_code":casestr,"init_code":initstr}
                    dictB = {"prod_code":obs_prod_code,"case_code":casestr}
                    print("About to parallelise matching for:\n",casestr,initstr)
                    matches, _2x2 = CAT.match_two_groups(
                                        dictA,dictB,do_contingency=True,)
                    print("=== RESULTS FOR",dom_code,member,casestr,
                        initstr,"===")

                    if _2x2 is None:
                        print("No objects - skipping.")
                    else:
                        qquad = '    '
                        bias = _2x2.get("BIAS")
                        pod = _2x2.get("POD")
                        far = _2x2.get("FAR")
                        csi = _2x2.get("CSI")
                        print(f"BIAS: {bias:.2f}",qquad,f"POD: {pod:.2f}",qquad,
                                f"FAR: {far:.2f}",qquad,f"CSI: {csi:.2f}")
                        print(f"a = {_2x2.a}",qquad,f"b = {_2x2.b}",qquad,
                                f"c = {_2x2.c}")
                    utils.save_pickle(obj=matches, fpath=fpath)
                    utils.save_pickle(obj=_2x2, fpath=fpath2)
                    print("Data saved.")
                else:
                    matches = utils.load_pickle(fpath=fpath)
                    _2x2 = utils.load_pickle(fpath=fpath2)
                    print("Data loaded.")

                # Keep in memory
                MATCH[dom_code][member][casestr][initstr]['2x2'] = _2x2
                MATCH[dom_code][member][casestr][initstr]['matches'] = matches

    # Plot performance diagrams for each case - can't separate by lead time...


if do_object_distr:
    print(stars,"DOING OBJ DISTRIBUTIONS",stars)
    # Plot distributions of objects in each domain and obs
    # TODO: run object ID for d02_3km?
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)
    # pdb.set_trace()
    # max_updraught
    # mean_updraught
    # distance_from_centroid
    # angle_from_centroid
    # distance_from_wcentroid
    # updraught_area_km
    attrs = ("max_updraught",)

    if True:
        # Data from objects and updraught data
        df_fcst = megaframe[(megaframe['member'] != "obs")]

        from evac.plot.fourhist import FourHist
        fpath = os.path.join(outroot,"histplots","fourhist_mode.png")
        utils.trycreate(fpath)
        FH = FourHist(fpath)
        FH.plot(df_fcst,xname="resolution",yname="conv_mode",dataname="max_updraught")

        fpath = os.path.join(outroot,"histplots","fourhist_case.png")
        FH = FourHist(fpath)
        FH.plot(df_fcst,xname="resolution",yname="case_code",dataname="max_updraught")

        fpath = os.path.join(outroot,"histplots","fourhist_leadtime.png")
        FH = FourHist(fpath)
        FH.plot(df_fcst,xname="resolution",yname="leadtime_group",dataname="max_updraught")

        fpath = os.path.join(outroot,"histplots","fourhist_reso.png")
        FH = FourHist(fpath)
        FH.plot(df_fcst,xname="resolution",yname="case_code",dataname="qlcsness")


    if True:
        # Subset for only d01/d02
        from evac.plot.pairplot import PairPlot

        for fcstmin in (45,90,135,"ALL"):
            if fcstmin == "ALL":
                miniframe = megaframe[(megaframe['member'] != "obs")]
            else:
                        miniframe = megaframe[(megaframe['member'] != "obs") &
                                    (megaframe['lead_time'] == fcstmin)]

            vars = ["max_updraught","max_intensity","perimeter","ud_distance_from_centroid"]

            miniframe.sort_values(axis=0,by="conv_mode",inplace=True)
            fpath = os.path.join(outroot,"pairplots","pairmode_1_{}min.png".format(fcstmin))
            utils.trycreate(fpath)
            PP = PairPlot(fpath)
            PP.plot(miniframe.sample(frac=0.4),color_name="conv_mode",vars=vars,palette="husl")

            miniframe.sort_values(axis=0,by="resolution",inplace=True)
            fpath2 = os.path.join(outroot,"pairplots","pairdx_1_{}min.png".format(fcstmin))
            PP2 = PairPlot(fpath2)
            PP2.plot(miniframe.sample(frac=0.4),color_name="resolution",vars=vars,palette=RESCOLS)

            vars = ["max_updraught","max_lowrot","longaxis_km","qlcsness"]

            miniframe.sort_values(axis=0,by="conv_mode",inplace=True)
            fpath = os.path.join(outroot,"pairplots","pairmode_2_{}min.png".format(fcstmin))
            utils.trycreate(fpath)
            PP = PairPlot(fpath)
            PP.plot(miniframe.sample(frac=0.4),color_name="conv_mode",vars=vars,palette="Set2")

            miniframe.sort_values(axis=0,by="resolution",inplace=True)
            fpath2 = os.path.join(outroot,"pairplots","pairdx_2_{}min.png".format(fcstmin))
            PP2 = PairPlot(fpath2)
            PP2.plot(miniframe.sample(frac=0.4),color_name="resolution",vars=vars,palette=RESCOLS)

    if True:
        # Plot updraught's distance from centroid against max intensity
        # For d01/d02 separately.
        import seaborn as sns
        sns.set(style='dark',font_scale=0.5)

        fpath = os.path.join(outroot,"kdeplots","ud_distance_v_intensity.png")
        utils.trycreate(fpath)

        f, axes = plt.subplots(2,4,figsize=(10,7),)#sharex=True,sharey=True)

        def ax_gen(axes):
            for n,ax in enumerate(axes.flat):
                yield ax

        axf = ax_gen(axes)

        for dx,res in zip((1.0,3.0),("hi-res","lo-res")):
            print("Plotting the {}km stats".format(int(dx)))
            df_km = megaframe[(megaframe['resolution'] == res) &
                    (megaframe['member'] != 'obs')]
            for n,case_code in enumerate(["20160331","20170501",
                                        "20170502","20170504"]):
                ax = next(axf)
                df_case = df_km[(df_km['case_code'] == case_code)]
                cmap = sns.cubehelix_palette(start=(n/2.5), light=0.8, dark=0.2, as_cmap=True)
                sns.kdeplot(df_case['max_updraught'],
                    df_case['ud_distance_from_centroid'], cmap=cmap,
                    shade=True, ax=ax, aspect=1,
                    # cut=5,
                    )
                ax.set_xlim([0,40])
                ax.set_ylim([0,40])
                ax.set_aspect('equal')
                ax.set_title("{}km for {}".format(int(dx),case_code))
        f.tight_layout()
        f.savefig(fpath)
        print("Figure saved to",fpath)
        plt.close(f)

if do_object_matching:
    print(stars,"DOING OBJ MATCHING/DIFFS",stars)
    # Load one big df with all objects
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe_all = load_megaframe(fmts=all_fmts)
    # For testing
    #megaframe = megaframe_all[(megaframe_all['case_code'] == '20160331') &
    #                            (megaframe_all['lead_time'] > 150)]
    megaframe = megaframe_all
    CAT = Catalogue(megaframe,tempdir=objectroot,ncpus=ncpus)

    # megaframe = load_megaframe()

    # Match objects
    # JRL TODO: need to redo this and loop over member to member?
    fname = "object-matches_d01-d02.pickle"
    fpath = os.path.join(objectroot,fname)
    if (not os.path.exists(fpath)):
        verif_domain = 'nexrad'
        matches = CAT.match_domains(members=member_names,initutcs=get_all_initutcs(),
                        leadtimes=all_fcstmins,domains=dom_names)
        utils.save_pickle(obj=matches, fpath=fpath)
    else:
        matches = utils.load_pickle(fpath=fpath)


    ### DIFFS ###

    # Do differences beteen difference domains/products
    # Positive diffs means d02 is higher.

    def write_row(df,n,key,val):
        df.loc[n,key] = val
        return

    def find_row(df,megaidx):
        row = df[(df['megaframe_idx'] == megaidx)]
        return row

    def do_write_diff(df,n,oldkey,newkey,d02,d01):
        diff = d02.get(oldkey).values[0] - d01.get(oldkey).values[0]
        write_row(df,nm,newkey,diff)
        return

    DTYPES = {
                "d01_id":"i4",
                "d02_id":"i4",

                # diffs
                "area_diff":"f4",
                # "centroid_gap_angle":"f4",
                "eccentricity_diff":"f4",
                "equivalent_diameter_diff":"f4",
                "extent_diff":"f4",
                "max_intensity_diff":"f4",
                "mean_intensity_diff":"f4",
                "perimeter_diff":"f4",
                "ratio_diff":"f4",
                "longaxis_km_diff":"f4",
                "qlcsness_diff":"f4",

                # Updraught props
                "max_updraught_diff":"f4",
                "mean_updraught_diff":"f4",
                "min_updraught_diff":"f4",


                # rotation diffs
                "max_rotation_diff":"f4",
                "mean_rotation_diff":"f4",
                # soemething like updraught ones above
                # "rot_exceed_ID_0":"f4"

                # non-straight-forward diffs
                "centroid_gap_km":"f4",
                # "ud_centroid_diff_km":"f4",
                # JRL TODO: need something for wind rose?
                }

    prop_names = []
    diff_df = utils.do_new_df(DTYPES,len(matches))
    for nm,match in enumerate(matches):
        if match is None:
            continue
        d01_obj_id, d02_obj_id = match
        d01 = find_row(megaframe,d01_obj_id)
        d02 = find_row(megaframe,d02_obj_id)

        # diff_df.loc[nm,'d01_id'] = d01_obj
        write_row(diff_df,nm,"d01_id",d01_obj_id)
        write_row(diff_df,nm,"d02_id",d02_obj_id)

        cd = utils.xs_distance(d02.centroid_lat.values[0],
                    d02.centroid_lon.values[0],
                    d01.centroid_lat.values[0],
                    d01.centroid_lon.values[0])/1000.0
        write_row(diff_df,nm,"centroid_gap_km",cd)

        # Do complicated differences in UH/AWS

        # Persistent 0-2 or 2-5 rotation signals within object bbox

        # Simple differences
        props = ("area","eccentricity","equivalent_diameter","extent",
                        "max_intensity","mean_intensity","perimeter",
                        "ratio","longaxis_km","qlcsness",
                        "max_updraught","mean_updraught",
                        "max_lowrot", "max_midrot",
                        )
        prop_diffs = [p + '_diff' for p in props]
        for prop, prop_diff in zip(props,prop_diffs):
            do_write_diff(diff_df,nm,prop,prop_diff,d02,d01)

    # These is the dataframe with no missing data (Nones)
    diff_df = diff_df[(diff_df['d01_id'] != 0.0)]


    #from evac.plot.ridgeplot import RidgePlot
    #fname = "diffs_ridgeplot.png"
    #fpath = os.path.join(outroot,"ridgeplots",fname)
    #utils.trycreate(fpath)
    #RidgePlot(fpath).plot(diff_df,namelist=prop_diffs,)

    fname = "hist_3x3.png"
    fpath = os.path.join(outroot,"match_diffs",fname)
    utils.trycreate(fpath)
    fig,axes = plt.subplots(nrows=3,ncols=5,figsize=(12,8))

    palette = itertools.cycle(sns.cubehelix_palette(14,light=0.8,dark=0.2))

    axit = axes.flat
    for n,p_d in enumerate(prop_diffs):
        ax = next(axit)
        ax.hist(diff_df[p_d],bins=50,color=next(palette))
        ax.set_title(p_d)
        ax.axvline(0,color='black')

    ax = next(axit)
    ax.hist(diff_df["centroid_gap_km"],bins=50,color='green')
    ax.set_title("Gap in centroids (km)")

    fig.tight_layout()
    fig.savefig(fpath)

    # Joint distributions


if do_object_examples:
    print(stars,"DOING OBJ EXAMPLE PLOT",stars)
    # Plot example storms for each range of qlcs values
    # 6 storms from left to right, increasing in qlcsness
    # Top, plot the bounding box from original data, plus pad
    # Bottom, plot the object pcolormesh.

    intv = 0.25; nstorms = 6
    qlcsmin = round_nearest(qlcs_all.min(),intv,method='floor')
    qlcsmax = round_nearest(qlcs_all.max(),intv,method='ceil')
    qlcsness_vals = N.linspace(qlcsmin,qlcsmax,nstorms)

    intensity_arr = N.zeros([100,40])
    obj_arr = N.zeros_like(intensity_arr)

    # Get subset of d02 only
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)

    miniframe = megaframe[(megaframe['resolution'] == "hi-res") &
                        (megaframe['member'] != 'obs') &
                        (megaframe['qlcsness'] > minq) &
                        (megaframe['qlcsness'] < maxq)
                        ]

    LOOKUP = {0:0,1:0,2:0,3:0,
                4:0,5:0,}

    for nq, q in enumerate(qlcsness_vals):
        minq = q-(intv/3)
        maxq = q+(intv/3)

        # Subset objects in this range
        miniframe = megaframe[(megaframe['resolution'] == "hi-res") &
                        (megaframe['member'] != 'obs') &
                        (megaframe['qlcsness'] > minq) &
                        (megaframe['qlcsness'] < maxq) ]

        # Pick this number
        oo = miniframe.iloc[LOOKUP[nq]]
        obj_id = oo.megaframe_idx
        obj_obj = oo.fpath_save
        obj = utils.load_pickle(obj_obj)
        coords = obj.object_props.loc[obj_id,'coords']

        # Transpose so bottom left is in bottom left of array


        # Shift to the right until not overlapping previous storm



        # "Stamp" the "real" storm, maybe >10 dBZ for clarity


        # "Stamp" the object outline


        # Annotate the QLCS-ness, object number, etc?


    fname = "examples_from_pca.png"
    fpath = os.path.join(outroot,fname)
    fig,axes = subplots(ncols=1,nrows=2)
    for nax,ax in enumerate(axes.flat):
        if nax == 0:
            S = Scales('REFL_comp')
            ax.contourf(intensity_arr,cmap=S.cm,levels=S.clvs)
        else:
            ax.pcolormesh(obj_arr,alpha=0.5,cmap=M.cm.get_cmap("magma",3),
                            #vmin=1,vmax=3,
                            )
    fig.tight_layout()
    fig.savefig(fpath)
    print("Saved image to",fpath)



if do_object_waffle:
    print(stars,"DOING WAFFLEPLOT",stars)
    # Compute PCA for d01/d02 combined objects, but for cellular only
    # Check the leading PCs again - is PC1 the same as previous PC2?

    fname = "pca_all_fcst_objs.pickle"
    fpath = os.path.join(objectroot,fname)
    P = utils.load_pickle(fpath=fpath)
    pca = P['pca']
    features = P['features']
    scaler = P['scaler']
    PC_df = P['data']

    # Then, find which PC corresponds to 'hi-res-ness'

    # Consider looping over all three PCs? Would need to know what
    # each PC represents in the all-fcst-obs PCA.

    # Subsample df (we need ~200-500 objects to make the waffle)
    PC_df_sub = PC_df.sample(n=250)

    # Then order all objects by that PC
    # JRL TODO: might need a new df megaframe hack, addition to Morph Index
    # Could call it the "Resolution Discrimination Index (RDI)"

    # Colour each object by domain, but the colour luminosity is PC value.

    # Do we see that objects are easily 'recognisable' as from which domain?
    # Clustering by domain it was simulated on, or random? We would "want" latter.
    pass

if do_object_cluster:
    print(stars,"DOING OBJ CLUSTERING",stars)
    # Similar to waffle above, but do objects cluster by domain before case?

    # Dendrogram?

    # What about that cool dendrogram heatmap/matrix on seaborn?
    pass

if do_object_brier_uh:
    # Load megaframe and verification matches

    # Below: try for cellular objects only too?

    # for each domain, for each case, for each init time:
    # for each time from 30 min to 150 min, list all observed objects
    # for each object, for each percentile, compute BS of exceeding that pc
    # decompose at the earliest point
    # compute RPS here too
    # Now append to a massive array/list

    # Below: do for each case, and do all cases on same
    # Create waffle plot of all BS scores/components/RPS - ordered by fcst min

    # Brier score: UH for 4 levels for matched objects
    pass
