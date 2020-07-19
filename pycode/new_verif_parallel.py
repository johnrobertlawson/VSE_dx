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
import math

# John hack
from multiprocess import Pool as mpPool

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
from scipy.stats import rv_histogram, gaussian_kde
from scipy.stats import mode as ss_mode
from scipy.ndimage import gaussian_filter
import seaborn as sns

import evac.utils as utils
from evac.datafiles.wrfout import WRFOut
from evac.plot.domains import Domains
from evac.stats.detscores import DetScores
from evac.plot.performance import Performance
# from evac.stats.fi import FI
from fiss import FISS
# from FISS.fiss import FISS
from evac.object.objectid import ObjectID
from evac.object.catalogue import Catalogue
from evac.plot.scales import Scales
from evac.datafiles.narrfile import NARRFile
# from windrose import WindroseAxes


### ARG PARSE ####
parser = argparse.ArgumentParser()

parser.add_argument('-N','--ncpus',dest='ncpus',default=1,type=int)
parser.add_argument('-oo','--overwrite_output',dest='overwrite_output',
                            action='store_true',default=False)
parser.add_argument('-op','--overwrite_pp',dest='overwrite_pp',
                            action='store_true',default=False)
parser.add_argument('-nq','--no_quick',dest="no_quick",
                            action='store_true',default=False)
parser.add_argument('-of','--overwrite_perf',dest='overwrite_perf',
                            action='store_true',default=False)

PA = parser.parse_args()
ncpus = PA.ncpus
overwrite_output = PA.overwrite_output
overwrite_pp = PA.overwrite_pp
overwrite_perf = PA.overwrite_perf
do_quicklooks = not PA.no_quick

### SWITCHES ###
do_plot_quicklooks = False
do_domains = False
do_percentiles = False
_do_performance = False # Broken - to delete?
do_performance = False
do_efss = False # TODO - average for cref over cases, for paper
# From scratch, do object_switch, do_object_pca, then
# delete the object dfs (in subdirs too), and
# and re-run object_switch to do the MDI of each cell
object_switch = False
do_object_pca = False
do_object_performance = False # TODO re-do with new (18) matching/diff vectors?
do_object_distr = False # TODO re-plot after matching new (18) matches
do_object_matching = False # TODO finish 18 members; do info-gain diffs?
do_object_windrose = False # # TODO NOT WORKING - WINDROSE PACKAGE HAS ISSUES
do_object_brier_uh = False # TODO finish - split into first_hour, second_hour etc
do_object_infogain = True #
do_case_outline = False # TODO colorbars, smoothing for SRH+shear, sparse wind barbs
do_one_objectID = False
do_qlcs_verif = False
do_spurious_example = False
do_oID_example = False
do_ign_storm = False

do_uh_infogain = False

# MAYBE DELETE
do_object_examples = False # TODO maybe delete, or do by hand
do_object_waffle = False # TODO maybe delete
do_object_cluster = False # TODO prob delete

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
key_output = "JulyPC"

# R1 - 96pc w/ 144 footprint
# R2 - 96pc w/ 120 footprint
# R3 - 94.5 pc w/ 120 footprint
# FebPC - 96pc w/ 144 footprint - let's go!

#extractroot = '/work/john.lawson/VSE_reso/pp/{}'.format(key_pp)
# extractroot = '/Users/john.lawson/data/{}'.format(key_pp)
lacie_root = '/Volumes/LaCie/VSE_dx'

extractroot = os.path.join(lacie_root,key_pp)

objectroot = os.path.join(lacie_root,'object_instances_PC')
# objectroot = os.path.join(lacie_root,'object_instances_FIXED')

# objectroot = "/Volumes/LaCie/VSE_dx/object_instances"
# outroot = "/home/john.lawson/VSE_dx/pyoutput"
# outroot = "/scratch/john.lawson/VSE_dx/figures"
#outroot = "/work/john.lawson/VSE_dx/figures"
# outroot = "/mnt/jlawson/VSE_dx/figures/"

# When on NSSL laptop
# outroot = "/Users/john.lawson/data/figures_{}".format(key_output)

# When on personal laptop
outroot = "/Users/johnlawson/paper_figures/VSE_dx/{}".format(key_output)

#tempdir = "/Users/john.lawson/data/intermediate_files"
tempdir = "/Volumes/LaCie/VSE_dx/intermediate_files_PC"
# tempdir = "/Volumes/LaCie/VSE_dx/intermediate_files_FIXED"
# narrroot = '/Users/john.lawson/data/AprilFool/NARR'
narrroot = None

##### OTHER STUFF #####
stars = "*"*10
# ncpus = 8
dom_names = ("d01","d02")
domnos = (1,2)
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
half_member_names = ['m{:02d}'.format(n) for n in range(1,19)]
test_member_names = ['m{:02d}'.format(n) for n in range(1,2)]
fifteen_member_names = ['m{:02d}'.format(n) for n in range(1,16)]
ten_member_names = ['m{:02d}'.format(n) for n in range(1,11)]
three_member_names = ['m{:02d}'.format(n) for n in range(1,4)]

# doms = (1,2)
# RAINNC
fcst_vrbls = ("Wmax","UH02","UH25","accum_precip","REFL_comp",
                "u_shear01","v_shear01","SRH03","CAPE_100mb",
                "u_shear06","v_shear06")
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
# CASES = { datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,19,0,0),], }

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
                    # 99, 99.5, 99.9, 99.95
                    "UH02":{
                        "3km": (5.67,9.52,25.97,35.63),
                        "1km": (15.33,24.01,48.65,57.05),
                        },
                    "AWS02":{
                        "3km": (3.6e-3,4.4e-3,6.7e-3,8.2e-3),
                        "1km": (3.7e-3,4.5e-3,6.9e-3,8.3e-3),
                        },
                    "UH25":{
                        "3km": (9.6,17.2,52.4,75.2),
                        "1km": (26.8,50.4,153.2,212.4),
                        },
                    "AWS25":{
                        "3km": (4.2e-3,5.1e-3,7.6e-3,9.0e-3),
                        "1km": (4.2e-3,5.2e-3,7.8e-3,9.2e-3),
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
            return (0.9, 0.99, 0.999,0.9995)
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
    # ("Wmax","UH02","UH25","RAINNC"):
    if (vrbl in fcst_vrbls):# or (vrbl in "MLCAPE","shear"):
        # TODO: are we not just doing 5-min or 1-hr accum_precip?
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        initHHMM = "{:02d}{:02d}".format(initutc.hour, initutc.minute)
        validHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "{}_{}_{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,initHHMM,
                                        validHHMM,mem)
    elif (vrbl in obs_vrbls): # ("AWS02","AWS25","DZ","ST4"):
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

def just_colorbar(fig,fobj,fpath,fontsize=30,cb_xlabel=None,**kw):
    """
    fobj is the output from a matplotlib plotting command.
    """
    # defaults - and for testing
    use_defaults = True
    kw = dict()
    if use_defaults:
        kw['orientation'] = 'horizontal'

    # I think this is important to get font size compatible with OG figure
    FIGSIZE = fig.get_size_inches()

    new_fig,ax = plt.subplots(figsize=FIGSIZE)
    cbar = plt.colorbar(fobj,ax=ax,**kw)
    cbar.ax.tick_params(labelsize=fontsize)
    if cb_xlabel:
        cbar.ax.set_xlabel(cb_xlabel,fontsize=fontsize)
    ax.remove()
    # plt.savefig(fpath)
    new_fig.tight_layout()
    plt.savefig(fpath,bbox_inches='tight')
    return

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
                    proj="lcc",latlon=None):
    bmap = Basemap(
                # width=12000000,height=9000000,
                urcrnrlat=urcrnrlat,urcrnrlon=urcrnrlon,
                llcrnrlat=llcrnrlat,llcrnrlon=llcrnrlon,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection=proj,ax=ax,
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
    if latlon is not None:
        x,y = bmap(latlon[1],latlon[0])
        bmap.scatter(x,y,s=800,color='k',marker='+',zorder=100)
    return bmap

def get_member_names(n):
    return ['m{:02d}'.format(n) for n in range(1,n+1)]

def _fix_df(df):
    # df.reset_index(level=0, inplace=True)
    # df.sort_index(inplace=True)
    return df

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

def loop_efss(caseutc,vrbl,fcstmin,fmt,middle_three=False):
    initutcs = CASES[caseutc]
    if middle_three:
        initutcs = initutcs[1:-1]
        assert len(initutcs) == 3
    else:
        assert len(initutcs) == 5
    for initutc in initutcs:
        validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))
        yield vrbl, caseutc, initutc, fmt, validutc

def all_init_sort():
    for caseutc, initutcs in CASES.items():
        for idx,initutc in enumerate(initutcs):
            yield caseutc, idx, initutc

def get_initutcs_strs():
    for caseutc, initutcs in CASES.items():
        for initutc in CASES[caseutc]:
            casestr = f"{caseutc.year:04d}{caseutc.month:02d}{caseutc.day:02d}"
            initstr = f"{initutc.hour:02d}{initutc.minute:02d}"
            # the_str = "_".join(casestr,initstr)
            yield initutc, casestr, initstr
            # validutc = initutc+datetime.timedelta(seconds=60*int(fcstmin))

def obj_perf_gen():
    for initutc, casestr, initstr in get_initutcs_strs():
        yield initutc, casestr, initstr


def load_both_data(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem):
    fcst_data = load_fcst_dll(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,
                                mem,return_ll=False)

    # These are the corresponding obs formats/data
    obs_vrbl, obs_fmt = fc2ob(fcst_vrbl,fcst_fmt)
    obs_data = load_obs_dll(validutc,caseutc,obs_vrbl=obs_vrbl,
                            obs_fmt=obs_fmt,return_ll=False)
    # pdb.set_trace()

    return fcst_data, obs_data

def normalise(x):
    return (x-N.nanmin(x))/(N.nanmax(x)-N.nanmin(x))

def load_fcst_dll(fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem,return_ll=True):
    if mem == "all":
        the_mems = member_names
    elif mem == "first_half":
        the_mems = get_member_names(18)

    if mem in ("first_half","all"):
        for midx, mem in enumerate(the_mems):
            fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                    validutc=validutc,caseutc=caseutc,initutc=initutc,
                    mem=mem)
            temp_data = N.load(fcst_fpath)
            if midx == 0:
                fcst_data = N.zeros([len(the_mems),*temp_data.shape])
            fcst_data[midx,:,:] = temp_data
    else:
        fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                validutc=validutc,caseutc=caseutc,initutc=initutc,
                mem=mem)
        fcst_data = N.load(fcst_fpath)

    ######### JRL: ALL FCST DATA EDITING GOES ON HERE ##########
    if fcst_vrbl in ("UH02","UH25"):
        # fcst_data = N.negative(fcst_data)

        # small_no = 0.000001

        # fcst_data[N.isnan(fcst_data)] = 0.0
        # fcst_data[N.isnan(fcst_data)] = small_no

        # fcst_data[fcst_data<0.0] = small_no

        fcst_data[fcst_data<0.0] = 0.0
        # fcst_data[fcst_data<0.0] = N.nan
        pass

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
        # obs_data[obs_data<0.0] = N.nan
        obs_data[obs_data<0.0] = 0.0
        pass
    ############################################################

    if return_ll:
        obs_lats, obs_lons = load_latlons(fcst_fmt,caseutc)
        return obs_data, obs_lats, obs_lons
    else:
        return obs_data

def load_timewindow_both_data(vrbl,fmt,validutc,caseutc,
                    initutc,window=1,mem='all'):
    nt = window
    ws = int((nt-1)/2)

    if mem == 'all':
        pass
    elif mem == "first_half":
        pass
    else:
        raise Exception
    # Load central time first
    fcst_c, obs_c = load_both_data(vrbl,fmt,validutc,caseutc,initutc,mem)
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
        fcst_i, obs_i = load_both_data(vrbl,fmt,offsetutc,caseutc,initutc,mem)
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

    # features -> make pretty
    nice_features = []
    for f in features:
        s = copy.copy(f)
        s = s.replace('_',' ')
        s = s.capitalize()
        nice_features.append(s)

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
                label = "MDI (PC1)"
            else:
                label = "PC{}".format(n+1)
            names.append(label)
            scores.append(pc*100)

        bw = 0.2
        pos = N.arange(len(features))
        poss = [pos+(n*bw) for n in (1,0,-1)]

        for n, (pos,label, scorearray) in enumerate(zip(poss,names,scores)):
            ax.barh(pos, scorearray, bw, alpha=0.6, align='center',
                    color=RANCOL[n],label=label,
                    tick_label=nice_features,)
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



def get_kw(prodfmt,utc,threshold,mem=None,initutc=None,fpath=None,
                footprint=144):
    if threshold == 'auto':
        if '3km' in prodfmt:
            # Using 96pc
            threshold = 44.3
            # threshold = 41.5
        elif '1km' in prodfmt:
            threshold = 45.7
            # threshold = 43.0
        else:
            raise Exception
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
    kw['threshold'] = threshold
    kw['footprint'] = footprint


    # print(fcstmin)
    return kw

def load_megaframe(fmts,add_ens=True,add_W=True,add_uh_aws=True,
                    add_resolution=True,add_mode=True,
                    debug_before_uh=False,add_init=True):
    # Overwrite fmts...
    del fmts
    # fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    fcst_fmts = ("d02_1km","d01_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    #all_fmts = list(fcst_fmts) + list(obs_fmts)
    all_fmts = list(obs_fmts) + list(fcst_fmts)

    mega_fname = "MEGAFRAME.megaframe"
    mega_fpath = os.path.join(objectroot,mega_fname)
    # ens will create a similar df to megaframe, but separating
    # member and domain from "prod".

    # w adds on updraught information

    if os.path.exists(mega_fpath):
        print("Megaframe loaded.")
        return utils.load_pickle(mega_fpath)
    else:
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

            # Here, index is set
            df_og['miniframe_idx'] = df_og.index
            # df_og.set_index("miniframe_idx",inplace=True)

            #### HACKS #####
            print("Adding/modifying megaframe...")

            # These also save the .pickle files (like
            # W and ens dataframes below) to speed up
            # creating new dfs to hack into the megaframe
            # pdb.set_trace()
            df_og = add_hax_to_df(df_og,fmt=fmt)
            # pdb.set_trace()

            # Here, index is correct.

            # JRL TODO: put in MDI, RDI, Az.shear, QPF stuff!
            # Add on W
            if add_W and ("d0" in fmt):
                CAT = Catalogue(df_og,ncpus=ncpus,tempdir=objectroot)
                print("Created/updated dataframe Catalogue object.")
                W_lookup = load_lookup((fmt,),vrbl="Wmax",)#fmts)
                W_df = load_W_df(W_lookup,CAT,fmt=fmt)
                # df_og = concat_W_df(df_og,W_df,fmt=fmt)
                # df_og = pandas.concat((df_og,W_df),axis=1)
                df_og = concat_two_dfs(df_og,W_df)
                print("Megaframe hacked: updraught stats added.")

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
                    df_og = concat_two_dfs(df_og,uh_df)
                    print("Megaframe hacked: UH stats added.")

            df_og.drop_duplicates(inplace=True)
            # pdb.set_trace()
            #pdb.set_trace()
            df_list.append(df_og)
        # df_list.append(df_og)


    # Setting object indices!
    # df_og = pandas.concat(df_list,ignore_index=False)
    df_og = pandas.concat(df_list,ignore_index=True)

    df_og['megaframe_idx'] = df_og.index
    # At this point, the megaframe indices are set in stone!
    # In their own column: megaframe_idx
    print("Megaframe created.")

    # pdb.set_trace()
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
        # rot_exceed_vals = PC_Thresh # Pass in the class!
        uh_df = CAT.compute_new_attributes(lookup,do_suite=layer,
                                            rot_exceed_vals=PC_Thresh,
                                            suffix=sfx)
        # pdb.set_trace()
        uh_df.set_index("miniframe_idx",inplace=True)

        utils.save_pickle(obj=uh_df,fpath=fpath)
        print("UH DF Saved to",fpath)
    else:
        uh_df = utils.load_pickle(fpath=fpath)
        print("UH metadata DataFrame loaded from",fpath)
    return uh_df

def concat_two_dfs(df_o,df_w,lsuffix=None):
    kw = {'lsuffix':lsuffix}
    new_df = df_o.join(df_w,**kw)
    return new_df

def load_W_df(W_lookup,CAT,fmt):
    fname = f"W_df_{fmt}.pickle"
    fpath = os.path.join(objectroot,fname)
    if not os.path.exists(fpath):
        sfx = f"_W_{fmt}"
        W_df = CAT.compute_new_attributes(W_lookup,do_suite="W",suffix=sfx)
        W_df.set_index("miniframe_idx",inplace=True)

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
                    # fcmns = (0,)
                else:
                    obs = False
                    fcst = True
                    # FCST
                    mns = member_names
                    # fcmns = all_fcstmins
                fcmns = all_fcstmins
                # pdb.set_trace()
                for mem in mns:
                    for validmin in fcmns:
                        validutc = initutc+datetime.timedelta(seconds=60*int(validmin))
                        path_to_pickle = get_extraction_fpaths(vrbl=fcst_vrbl,
                                    fmt=fcst_fmt,validutc=validutc,
                                    caseutc=caseutc,initutc=initutc,mem=mem)
                        if obs:
                            validmin = 0
                        # pdb.set_trace()
                        yield dict(fcst_vrbl=fcst_vrbl, valid_time=validutc,
                                fcst_min=validmin, member=mem,
                                path_to_pickle=path_to_pickle,fcst_fmt=fcst_fmt)

def create_lookup(fcst_fmts,vrbl):
    itr = list(loop_ens_data(fcst_vrbl=vrbl,fcst_fmts=fcst_fmts))
    nobjs = len(itr)

    def lookup_series(d):
        DTYPES = {
            # "index":"object",
            "fcst_vrbl":"object",
            "valid_time":"object",
            "fcst_min":"object",
            # "prod_code":"object",
            "path_to_pickle":"object",
            "fcst_fmt":"object",
            "member":"object",
            }

        new_df = utils.do_new_df(DTYPES,1)
        for key in DTYPES.keys():
            new_df.loc[0,key] = d[key]
        return new_df

    print(f"Creating {vrbl} lookup.")

    if ncpus > 1:
        with mpPool(ncpus) as pool:
            # results = pool.map(compute_df_hax,gg)
            results = pool.map(lookup_series,itr)
    else:
        results = []
        for i in itr:
            results.append(lookup_series(i))

    df_lookup = pandas.concat(results,ignore_index=False)
    # This index is nothing to do with megaframe index
    # df_lookup.reset_index(level=0, inplace=True)
    # pdb.set_trace()
    return df_lookup

def add_hax_to_df(df_og,fmt,pca=None):
    def gen(df):
        for o in df.itertuples():
            yield o

    def compute_df_hax(o,pca=True):
        DTYPES = {
                    "miniframe_idx":"object",
                    #"resolution":"object",
                    #"conv_mode":"object",
                    #"case_code:":"object",
                    #"member":"object",
                    #"domain":"object",
                    #"leadtime_group":"object",
                    #"init_code":"object",
                }
        df = utils.do_new_df(DTYPES,1)

        # Indexing
        # idx = o.index
        idx = 0 # just the first entry of this df
        # This is the actual object index
        df.loc[idx,"miniframe_idx"] = o.miniframe_idx

        # Resolution
        res = "EE1km" if (int(o.nlats) > 200) else "EE3km"
        df.loc[idx,"resolution"] = res

        # Member, domain
        dom, dxkm, mem = o.prod_code.split("_")
        df.loc[idx,"member"] = mem
        df.loc[idx,"domain"] = dom

        # Compute QLCS-ness (MDI)
        pca_fmt = f"{dom}_{dxkm}"
        #if pca is not None:
        pca_fname = "pca_model_{}.pickle".format(pca_fmt)
        pca_fpath = os.path.join(objectroot,pca_fname)
        P = utils.load_pickle(fpath=pca_fpath)

        # series = df.loc[0,P['features']]
        # qlcsness = P['pca'].transform(P['scaler'].transform(series))[0][0]
        x = N.zeros([1,len(P['features'])])
        #for ns,s in enumerate(P['features']):
        #    x[ns] = o.s

        # Hard coded because I don't understand Pandas
        x[0,0] = o.area
        x[0,1] = o.eccentricity
        x[0,2] = o.extent
        x[0,3] = o.max_intensity
        x[0,4] = o.mean_intensity
        x[0,5] = o.perimeter
        x[0,6] = o.longaxis_km

        qlcsness = P['pca'].transform(P['scaler'].transform(x))[0][0]
        df.loc[idx,'qlcsness'] = qlcsness

        # Mode
        # pdb.set_trace()
        if qlcsness < -0.5:
            conv_mode = "cellular"
        elif qlcsness > 0.5:
            conv_mode = "linear"
        else:
            conv_mode = "ambiguous"
        df.loc[idx,"conv_mode"] = conv_mode

        # Case code
        utc = o.time
        if utc < datetime.datetime(2016,4,1,12,0,0):
            case_code = "20160331"
        elif utc < datetime.datetime(2017,5,2,12,0,0):
            case_code = "20170501"
        elif utc < datetime.datetime(2017,5,3,12,0,0):
            case_code = "20170502"
        elif utc < datetime.datetime(2017,5,5,12,0,0):
            case_code = "20170504"
        else:
            raise Exception
        df.loc[idx,"case_code"] = case_code

        # Lead-time group
        lt = o.lead_time
        if int(lt) < 62:
            gr = "first_hour"
        elif int(lt) < 122:
            gr = "second_hour"
        elif int(lt) < 182:
            gr = "third_hour"
        else:
            raise Exception
        df.loc[idx,"leadtime_group"] = gr

        # Init-time group
        if mem == "obs":
            ic = None
        else:
            dt = utc - datetime.timedelta(seconds=60*lt)
            ic = f"{dt.hour:02d}{dt.minute:02d}"
        df.loc[idx,"init_code"] = ic


        # pdb.set_trace()
        return df

    # First - check if this has been computed already
    fpath = os.path.join(objectroot,f"hax_df_{fmt}.pickle")

    if not os.path.exists(fpath):

        t0 = time.time()
        # Dataset is too big for memory limitations
        # Split into 10
        #df_hax_list = []
        #batches = 5
        #nrows = df_og.shape[0]
        #chunk_size = int(nrows/batches)
        #count = 0
        #for start in range(0,nrows,chunk_size):
        #    count += 1
        #    print(f"Parallelising for chunk #{count}.")
        #    df_subset = df_og.iloc[start:start + chunk_size]



        gg = gen(df_og)
        #    gg = gen(df_subset)
            # cs = math.ceil(chunk_size/ncpus)
        #cs = int(math.ceil(df_og.shape[0]/ncpus))

        if ncpus > 1:
            with mpPool(ncpus) as pool:
                # results = pool.map(compute_df_hax,gg)
                results = pool.map(compute_df_hax,gg)#, chunksize=cs)
        else:
            results = []
            for o in gg:
                results.append(compute_df_hax(o))

        #print("Done this chunk; now joining this batch of dataframes.")
        #df_hax_list.append(pandas.concat(results,ignore_index=False))
        # pdb.set_trace()
        pass

        print("Done all parallelisation; now joining dataframes.")
        # df_hax = pandas.concat(df_hax_list,ignore_index=False)
        df_hax = pandas.concat(results,ignore_index=False)
        df_hax.set_index("miniframe_idx",inplace=True)

        # pdb.set_trace()
        utils.save_pickle(obj=df_hax,fpath=fpath)
        print("Saved to",fpath)

        t1 = time.time()
        dt = t1-t0
        dtm = int(dt//60)
        dts = int(dt%60)

        print(f"DF creation took {dtm:d} min  {dts:d} sec.")
    else:
        df_hax = utils.load_pickle(fpath=fpath)
        print("Resolution metadata DataFrame loaded from",fpath)


    # Set index for df_hax
    # pdb.set_trace()
    # This should merge new attributes but overwrite old ones
    df_og = concat_two_dfs(df_og,df_hax,lsuffix='_left')
    # df_og.update(df_hax)
    # df_og.combine_first(df_hax)
    # pdb.set_trace()
    print("Megaframe hacked: resolution added")
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

def get_MATCH(mns,return_megaframe=False,modes=('cellular',),
            custom_megaframe=None):
    if custom_megaframe is not None:
        megaframe = custom_megaframe
    else:
        # Load one big df with all objects
        fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
        obs_fmts = ("nexrad_3km","nexrad_1km")
        all_fmts = list(fcst_fmts) + list(obs_fmts)
        megaframe = load_megaframe(fmts=all_fmts)
        # pdb.set_trace()
    CAT = Catalogue(megaframe,tempdir=objectroot,ncpus=ncpus)

    # pdb.set_trace()

    MATCH = {}
    # Create pickle of matches
    # for dom_code in ('d01_3km','d02_1km'):
    # modes = ('cellular',)#'linear')
    for member in mns:
        for dom_code, mode in itertools.product(('d02_1km','d01_3km'),modes):
            if dom_code not in MATCH.keys():
                MATCH[dom_code] = {}
            dom, res = dom_code.split('_')
            obs_prod_code = f"nexrad_{res}_obs"
            #for member in member_names:
            if member not in MATCH[dom_code].keys():
                MATCH[dom_code][member] = {}
            print(f"Now verifying {mode} objects for",member,dom_code)
            for initutc, casestr, initstr in get_initutcs_strs():
                if casestr not in MATCH[dom_code][member].keys():
                    MATCH[dom_code][member][casestr] = {}
                if initstr not in MATCH[dom_code][member][casestr].keys():
                    MATCH[dom_code][member][casestr][initstr] = {
                            m:{} for m in modes}
                fname = f"{dom}_{member}-verif_{casestr}_{initstr}.pickle"
                fpath = os.path.join(objectroot,"verif_match_tuples",
                                        mode,casestr,initstr,fname)
                fpath2 = os.path.join(objectroot,
                                "verif_match_2x2",mode,casestr,initstr,fname)
                fpath3 = os.path.join(objectroot,"verif_match_sorted",
                                        mode,casestr,initstr,fname)
                if (not os.path.exists(fpath)) or (not os.path.exists(fpath2)):
                    # pdb.set_trace()

                    fcst_prod_code = '_'.join((dom_code,member))
                    dictA = {"prod_code":fcst_prod_code,#"member":member,
                                    "case_code":casestr,"init_code":initstr,
                                    # "conv_mode":"cellular"}
                                    "conv_mode":mode}
                    dictB = {"prod_code":obs_prod_code,"case_code":casestr,
                                    # "conv_mode":"cellular"}
                                    "conv_mode":mode}
                    print("About to parallelise matching for:\n",casestr,initstr)
                    matches, _2x2, sorted_pairs = CAT.match_two_groups(
                                        dictA,dictB,do_contingency=True,td_max=21.0)
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
                    utils.save_pickle(obj=sorted_pairs,fpath=fpath3)
                    print("Data saved.")
                else:
                    matches = utils.load_pickle(fpath=fpath)
                    _2x2 = utils.load_pickle(fpath=fpath2)
                    sorted_pairs = utils.load_pickle(fpath=fpath3)
                    # pdb.set_trace()
                    print("Data loaded.")

                # Keep in memory
                MATCH[dom_code][member][casestr][initstr][mode]['2x2'] = _2x2
                MATCH[dom_code][member][casestr][initstr][mode]['matches'] = matches
                MATCH[dom_code][member][casestr][initstr][mode]['sorted'] = sorted_pairs
            pass
        pass

    if return_megaframe:
        return MATCH, megaframe
    return MATCH

def get_dom_match(mns,return_megaframe=False,modes=None):
    if modes is None:
        # modes = ('linear',)
        modes = ('cellular',)
    # Load one big df with all objects
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)
    # pdb.set_trace()
    CAT = Catalogue(megaframe,tempdir=objectroot,ncpus=ncpus)

    MATCH = {}
    # Create pickle of matches
    # for dom_code in ('d01_3km','d02_1km'):
    for member in mns:
        MATCH[member] = {}
        for mode in modes:
            MATCH[member][mode] = {}
            for initutc, casestr, initstr in get_initutcs_strs():
                if casestr not in MATCH[member][mode].keys():
                    MATCH[member][mode][casestr] = {}
                if initstr not in MATCH[member][mode][casestr].keys():
                    MATCH[member][mode][casestr][initstr] = {}

                fname_match = f"{mode}_{member}-dom-match_{casestr}_{initstr}.pickle"
                fname_2x2 = fname_match.replace("match","2x2")
                fname_sorted = fname_match.replace("match","sortedpairs")
                mdir = os.path.join(objectroot,"dom_match_pickles",
                                        mode,casestr,initstr)
                fpath_match = os.path.join(mdir,fname_match)
                fpath_2x2 = os.path.join(mdir,fname_2x2)
                fpath_sorted = os.path.join(mdir,fname_sorted)
                if (not os.path.exists(fpath_match)):
                    dictA = {"member":member,"conv_mode":mode,
                                "case_code":casestr,"init_code":initstr,
                                "domain":"d01"}
                    dictB = {"member":member,"conv_mode":mode,
                                "case_code":casestr,"init_code":initstr,
                                "domain":"d02"}
                    print("About to parallelise dom-matching for:\n",
                                stars,casestr,initstr,member,stars)
                    # pdb.set_trace()
                    matches, _2x2, sorted_pairs = CAT.match_two_groups(
                                        dictA,dictB,do_contingency=True,td_max=21.0)
                    print("=== RESULTS FOR",member,casestr,
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
                    utils.save_pickle(obj=matches, fpath=fpath_match)
                    utils.save_pickle(obj=_2x2, fpath=fpath_2x2)
                    utils.save_pickle(obj=sorted_pairs, fpath=fpath_sorted)
                    print("Data saved.")
                else:
                    matches = utils.load_pickle(fpath=fpath_match)
                    _2x2 = utils.load_pickle(fpath=fpath_2x2)
                    sorted_pairs = utils.load_pickle(fpath=fpath_sorted)
                    # pdb.set_trace()
                    print("Data loaded.")

                # Keep in memory
                MATCH[member][mode][casestr][initstr]['2x2'] = _2x2
                MATCH[member][mode][casestr][initstr]['matches'] = matches
                MATCH[member][mode][casestr][initstr]['sorted'] = sorted_pairs
            pass
        pass

    if return_megaframe:
        return MATCH, megaframe
    return MATCH

def constrain(frac,minmax=(0.01,0.99)):
    assert 0.0 <= frac <= 1.0
    assert minmax[1] > minmax[0]
    if frac > minmax[1]:
        return minmax[1]
    elif frac < minmax[0]:
        return minmax[0]
    else:
        return frac


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
        # "UH02":(0.0,100.0),
        "UH02":(0.0,70.0),
        "AWS25":(0.0,0.02),
        "AWS02":(0.0,0.02),
        }


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

                D.add_domain(name=name,label=label,lats=lats,lons=lons,
                                color=COLORS[dname])
        D.plot_domains()
    print("Domains plot saved to",fpath)

if do_percentiles:

    nbinn = 1000

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

        # Test UH problems
        # In "load_fcst_dll", data is multiplied by -1; NaNs removed; negatives removed

        # hist = N.histogram(data,bins=nbinn,range=MINMAX[v],)
        # data[data < 0.0] = N.nan
        data = data[~N.isnan(data)]
        hist = N.histogram(data,bins=nbinn,range=MINMAX[v],)[0]
        #if N.percentile(data,99.9) < 1:
        #    pdb.set_trace()
        return hist

    def plot_quantiles(RVH,ax,quantiles,mul_units,pad=0.08,vrbl=None):
        multiplier, units = mul_units

        for q in quantiles:
            col = "red" # if q in red_pc else "black"
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
    _vrbls = ("UH02","UH25","AWS02","AWS25")
    # _vrbls = ("AWS25","UH25",)
    #_vrbls = ["NEXRAD","REFL_comp"]
    # _vrbls = ["NEXRAD_cut","REFL_comp_cut"]

    for _vrbl in _vrbls:
        for kmstr in ('3km','1km'):
            pcroot = os.path.join(outroot,"pc_distr",_vrbl,kmstr)

            fname_npy = "pc_distr_{}_{}.npy".format(_vrbl,kmstr)
            fpath_npy = os.path.join(outroot,tempdir,"pc_distr",fname_npy)
            # pdb.set_trace()
            if not os.path.exists(fpath_npy) or overwrite_pp:
                if ncpus > 1:
                    with multiprocessing.Pool(ncpus) as pool:
                        results = pool.map(pc_func,gen_pc_loop(_vrbl,kmstr))
                else:
                    results = []
                    for i in gen_pc_loop(_vrbl,kmstr):
                        results.append(pc_func(i))
                # Do merged histogram
                results = N.array(results)
                utils.trycreate(fpath_npy,isdir=False)
                N.save(file=fpath_npy,arr=results)
            else:
                results = N.load(fpath_npy,allow_pickle=True)

            # hist = N.sum(results[:,0],axis=0)
            hist = N.sum(results,axis=0)
            lin_bins = N.linspace(*MINMAX[_vrbl],nbinn+1)
            # RVH = rv_histogram((hist,old_bins))
            RVH = rv_histogram((hist,lin_bins))

            print(_vrbl,kmstr,N.sum(hist),"points counted/sorted.")

            # bins = N.linspace(old_bins.min(),old_bins.max(),num=hist.shape[0])
            bins = lin_bins




            #q1 = N.arange(0.1,1.0,0.1)
            #q2 = N.array([0.95,0.99,0.995,0.999,0.9999])
            #quantiles = N.concatenate([q1,q2])
            quantiles = (
                            # 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
                            0.9,0.95,0.99,0.995,0.999,0.9995,0.9999)
            # RVH.cdf(x=quantiles)


            b_hist = N.cumsum(hist)/N.sum(hist) # * 100
            bin_vals = []
            for q in quantiles:
                bin_no = len(b_hist[b_hist <= q])
                bin_vals.append(bins[bin_no])

                print(f"{q:.4f}: {bins[bin_no]:.4f}")
            # pdb.set_trace()


            if False:
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

            if True:
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

            if False:
                fname_fig = "pc_scatter_{}_{}.png".format(_vrbl,kmstr)
                fpath_fig3 = os.path.join(pcroot,fname_fig)

                fig,ax = plt.subplots(1)
                ax.scatter(bins,RVH.cdf(bins))

                fig.savefig(fpath_fig)
                ax.set_xlim(LIMS[_vrbl])
                # ax.set_ylim([RVH.ppf(50)])
                fig.savefig(fpath_fig2)
                # pdb.set_trace()

if _do_performance:
    fcst_fmts = ("d01_3km","d02_1km")
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

    _vs = ("REFL_comp",)
    _ths = (15,30,40,50)
    #_fms = (60,120,180)
    _fms = (30,60,90,120,150,180)

    switch_25 = 0
    # overwrite_perf is set in lines 60-70 from python args

    # allall is just for 40 dBZ - want to plot that for all
    fpath_diffs = "./trad_perf_diffs_array.pickle"
    fpath_allpod = './allall_pod.pickle'
    fpath_allfar = './allall_pod.pickle'

    if os.path.exists(fpath) and os.path.exists(allall_fpath) and (not overwrite_perf):
        diffs_arr = utils.load_pickle(fpath)
        allall = utils.load_pickle(allall_fpath)
    else:

        clist = []
        for case in CASES.keys():
            clist.append(utils.string_from_time('dir',case,strlen='day'))
        allall = {p:{f:{c:[] for c in clist} for f in fcst_fmts} for p in ("pod","far")}

        # Create a "master" figure for the paper
        fname = "perfdiag_REFL_comp_30_multi.png"
        fpath = os.path.join(outroot,fname)
        PD0 = Performance(fpath=fpath,legendkwargs=None,legend=False)

        # This are continually overwritten i think...dimensions are silly
        far_vec_diff = N.zeros([4,len(_vs_),len(_fms),len(_ths),len(member_names),5])
        pod_vec_diff = N.zeros_like(far_vec_diff)

        # dimensions: (vrbl, case, fcst_min, threshold )
        # hm_arr = N.zeros([len(_vs),4,len(_fms),len(_ths)])
        hm_arr = N.zeros_like(far_vec_diff)

        ZZZ = {}
        YYY = {}
        for (nthresh,thresh), (nvrbl,vrbl), (nfm,fcstmin) in itertools.product(
                enumerate(_ths),enumerate(_vs,),enumerate(_fms)):
            for fmt in ("d01_3km","d02_1km"): # in indices, [1] - [0] for figs
                itr = loop_perf(vrbl=vrbl,thresh=thresh,fcstmin=fcstmin,fcst_fmt=fmt)

                with multiprocessing.Pool(ncpus) as pool:
                    results = pool.map(compute_perf,itr)

                pdb.set_trace()
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
            for ncase,casestr in enumerate(casestrs):
                fname1 = "perfdiag_{}_{}th_{}min_{}".format(vrbl,thresh,fcstmin,casestr)
                fpath1 = os.path.join(outroot,"perfdiag",casestr,fname1)
                PD1 = Performance(fpath=fpath1,legendkwargs=None,legend=True)

                fname2 = "perfdiag_{}_{}th_{}min_{}_eachens".format(vrbl,thresh,fcstmin,casestr)
                fpath2 = os.path.join(outroot,"perfdiag","eachens",casestr,fname2)
                PD2 = Performance(fpath=fpath2,legendkwargs=None,legend=False)

                pod_all = []
                far_all = []
                pod_mem = N.zeros([2,len(member_names),5])
                far_mem = N.zeros_like(pod_mem)
                #pod_mem = {f:{m:N.zeros([2,len(member_names)]) for m in member_names} for f in fcst_fmts}
                #far_mem = {f:{m:[] for m in member_names} for f in fcst_fmts}
                for nfmt, fmt in enumerate(fcst_fmts):
                    POD = ZZZ[fmt]["POD"]
                    FAR = ZZZ[fmt]["FAR"]

                    pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                                    'alpha':ALPHAS[fmt]}
                    lstr = get_nice(fmt)
                    print("Plotting",fmt,"to",fpath1)
                    # pdb.set_trace()
                    PD1.plot_data(pod=N.mean(POD[casestr]),
                            far=N.mean(FAR[casestr]),plotkwargs=pk,label=lstr)

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
                    for nmem, mem in enumerate(member_names):
                        _pod_ = YYY[fmt]["POD"][casestr][mem]
                        pod_all.append(_pod_)
                        _far_ = YYY[fmt]["FAR"][casestr][mem]
                        far_all.append(_far_)
                        if (thresh == 30):
                            allall['pod'][fmt][casestr].append(_pod_)
                            allall['far'][fmt][casestr].append(_far_)

                        # pod_mem[fmt][mem].append(_pod_)
                        # far_mem[fmt][mem].append(_far_)
                        pod_mem[nfmt,nmem,:] = _pod_
                        far_mem[nfmt,nmem,:] = _far_

                # One last dump (heh heh)

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

                # Get differences between POD and FAR for vector computation!
                for nmem, mem in enumerate(member_names):
                    pod_vec_diff[ncase,nvrbl,nfm,nthresh,nmem,:] = (pod_mem[1,nmem,:] - pod_mem[0,nmem,:])
                    # far_vec_diff[ncase,nmem,:] = (far_mem[1,nmem,:] - far_mem[0,nmem,:])
                    far_vec_diff[ncase,nvrbl,nfm,nthresh,nmem,:] = (far_mem[1,nmem,:] - far_mem[0,nmem,:])
                # pdb.set_trace()

                # Do wind rose of performance-diagram differences (i.e. which way the difference
                # points, with optimal being 1.0 in the "northeast" direction.

                # dfar = far_vec_diff[ncase,1,:,:] - far_vec_diff[ncase,0,:,:]
                # dpod = pod_vec_diff[ncase,1,:,:] - pod_vec_diff[ncase,0,:,:]
                # dfar = far_vec_diff[ncase,:,:].flatten()
                # dpod = pod_vec_diff[ncase,:,:].flatten()

                # Mean for this vrbl, fcstmin, case, threshold.
                dfar = far_vec_diff[ncase,nvrbl,nfm,nthresh,:,:].flatten()
                dpod = pod_vec_diff[ncase,nvrbl,nfm,nthresh,:,:].flatten()

                # For each in the array, compute speed and direction from x/y difference
                wspd, wdir = utils.combine_wind_components(dfar,dpod)

                # thresh, vrbl, fcstmin, casestr
                fname = f"perf_diag_diffs_{vrbl}_{fcstmin}min_{casestr}_{thresh}th.png"
                fpath = os.path.join(outroot,"perfdiag_diffs",fname)
                utils.trycreate(fpath)

                mag_list = []
                fig,ax = plt.subplots(1,figsize=(6,8))

                # Work out magnitude - if EE1km did better, make it positive.
                mag = N.sqrt(dpod**2 + dfar**2)
                EE3_best = {}
                # angles = (135,315)
                angles = (225,45)
                EE3_best[0] = N.where(wdir < max(angles))
                EE3_best[1] = N.where(wdir > min(angles))
                for n, idx in EE3_best.items():
                    mag[EE3_best[n]] *= -1

                _dfar = dfar *-1
                # Plot
                _sc = ax.scatter(dpod,_dfar,alpha=0.4,c=mag,
                                    # cmap=M.cm.cividis,
                                    vmin=-0.3,vmax=0.3,zorder=100,
                                    cmap='vse_diverg')
                ax.axhline(0,color='grey')
                ax.axvline(0,color='grey')
                ax.plot([-1,1],[-1,1],color='red',linestyle='dotted',zorder=0,alpha=0.5)
                ax.plot([-1,1],[1,-1],color='lightgrey',zorder=0,alpha=0.5)
                ax.grid(True,color='lightgrey')
                plt.colorbar(_sc,ax=ax,orientation='horizontal',shrink=0.7,)

                mean_dpod = N.mean(dpod)
                mean_dfar = N.mean(dfar)
                mean__dfar = N.mean(_dfar)
                # wspd, wdir = utils.combine_wind_components(mean_dfar,mean_dpod)

                # Work out magnitude - if EE1km did better, make it positive.
                mag, wdir = utils.combine_wind_components(mean_dfar,mean_dpod)
                # mag = N.sqrt(mean_dpod**2 + mean_dfar**2)
                angles = (225,45)
                Q = N.zeros([2],dtype=bool)
                Q[0] = wdir < max(angles)
                Q[1] = wdir > min(angles)
                if N.all(Q):
                    # d02 did better (well, while I think the order is right!)
                    # mag *= -1
                    col = "#D41159"
                    # N.zeros([4,len(member_names),5])
                    # JRL TODO: here, add this time/init/case to a
                    mag_list.append(-1 * mag)
                else:
                    col = "#1A85FF"
                    mag_list.append(mag)

                # if "perf_diag_diffs_REFL_comp_120min_20160331_15th" in fpath:
                #     pdb.set_trace()
                # Plot mean!
                ax.scatter(mean_dpod,mean__dfar,marker='X',c=col,
                                edgecolors='k',# markeredgewidth=3.0,
                                s=210,alpha=0.99,zorder=2000)

                ax.set_xlim([-0.3,0.3])
                ax.set_ylim([-0.3,0.3])
                ax.set_aspect('equal')
                ax.set_xlabel("EE1km-EE3km differences in Strike Rate")
                ax.set_ylabel("EE3km-EE1km differences in Prob. of Detection")

                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)
                # pdb.set_trace()
                pass
                hm_arr[nvrbl,ncase,nfm,nthresh] = N.array(mag_list)
                #pdb.set_trace()
                pass
            # pdb.set_trace()
            pass

            # PD.ax.set_xlim([0,0.3])
            fname3 = "perfdiag_{}_{}th_{}min_mean_ens".format(vrbl,thresh,fcstmin)
            fpath3 = os.path.join(outroot,"perfdiag","eachens",fname3)
            PD3 = Performance(fpath=fpath3,legendkwargs=None,legend=False)
            _podmms = []
            _farmms = []
            for nfmt, fmt in enumerate(fcst_fmts):
                for nmem, mem in enumerate(member_names):
                    annostr = "".format(mem)
                    pod_mem_mean = N.nanmean(pod_mem[nfmt,nmem,:])
                    far_mem_mean = N.nanmean(far_mem[nfmt,nmem,:])

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

            for nfmt, fmt in enumerate(fcst_fmts):
                for nmem, mem in enumerate(member_names):
                    annostr = "{}".format(mem)
                    pod_mem_mean = N.nanmean(pod_mem[nfmt,nmem,:])
                    far_mem_mean = N.nanmean(far_mem[nfmt,nmem,:])
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

        utils.save_pickle(obj=hm_arr,fpath=fpath)
        utils.save_pickle(obj=allall,fpath=allall_fpath)


    # Do summary plot of whether means were red or blue.
    # Use eFSS method of plotting diffs
    # dimensions: (vrbl, case, fcst_min, threshold )
    # averaged for vrbl over: (members, init times)

    # Loop image creation for vrbl and cases
    # y-axis is threshold (absolute)
    # x-axis is fcstmin
    # Just need to plot the diff values themselves ...

    kw = dict(vmin=-0.15,vmax=0.15)
    for (ncase, case), (nvrbl, vrbl) in itertools.product(
                    enumerate(CASES.keys()),enumerate(_vs)):
        casestr = utils.string_from_time('dir',case,strlen='day')
        fname = f"trad_perfdiag_diffs_{casestr}_{vrbl}.png"
        fpath = os.path.join(outroot,fname)
        fig,ax = plt.subplots(1)
        data = hm_arr[nvrbl,ncase,:,:].T
        im = ax.imshow(data,cmap='vse_diverg', **kw)

        nx,ny = data.shape
        ax.set_yticks(N.arange(nx))
        ax.set_xticks(N.arange(ny))

        ax.set_xticklabels(_fms)
        ax.set_xlabel("Forecast minute")

        ax.set_yticklabels(_ths)
        ax.set_ylabel("dBZ threshold")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        for i,j in itertools.product(range(nx),range(ny)):
            t = "{:.2f}".format(data[i,j])
            text = ax.text(j,i,t,ha="center",va="center",color="k",size=14,
                                fontweight="bold")
        plt.colorbar(im,ax=ax)
        fig.tight_layout()

        fig.savefig(fpath)
        print(f"Finished plotting differences for {vrbl} on {case}.")

    # Finally, the summary
    pod_all = allall['pod']
    far_all = allall['far']
    # pdb.set_trace()



    #  allall['pod'][fmt][casestr].append(pod_all)
    fname = f"trad_perfdiag_summary_30dBZ_alltimes.png"
    fpath = os.path.join(outroot,fname)
    PDx = Performance(fpath=fpath,legendkwargs=None,legend=True)
    switch_622 = 1
    # For each case, plot a marker and annotate that case
    letters = ["A","B","C","D"]

    PADS = {
        0:[-0.15,0.065],
        1:[-0.08,0.03],
        2:[-0.08,0.03],
        3:[-0.15,0.03],
        }

    for ncase,caseutc in enumerate(CASES.keys()):

        casestr = utils.string_from_time('dir',caseutc,strlen='day')

        for fmt in fcst_fmts:
            if switch_622:
                lstr = get_nice(fmt)
            else:
                lstr = None

            if casestr[:4] == "2016":
                ypad = 0.05
            else:
                ypad = 0.03

                # D and A need to be way left

            pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                        'alpha':ALPHAS[fmt]}
            podpt = N.nanmean(pod_all[fmt][casestr])
            farpt = N.nanmean(far_all[fmt][casestr])
            PDx.plot_data(pod=podpt,far=farpt,
                            plotkwargs=pk,label=lstr)
            # Label this case for d01  (should see which is which?)
            if fmt == "d01_3km":
                annostr = "{}-{}".format(letters[ncase],casestr)
                PDx.ax.annotate(annostr,xy=((1-farpt)+PADS[ncase][0],N.mean(podpt)+PADS[ncase][1]),
                                xycoords='data',fontsize=10,color='black',
                                fontweight='bold',annotation_clip=False)
        switch_622 = 0
    PDx.save()

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
                        temporal_window=temporal_window,
                        skip_fiss=True)
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

    temporal_windows = (3,)
    # temporal_windows = (5,)


    fcstmins = (30,90,150)
    vrbls = ("UH25","UH02","REFL_comp")

    for fcstmin, temporal_window,caseutc,vrbl in itertools.product(
                fcstmins,temporal_windows,CASES.keys(),vrbls):
        casestr = utils.string_from_time('dir',caseutc,strlen='day')
        fssdir = os.path.join(tempdir,"efss_fiss",vrbl)
        utils.trycreate(fssdir,isdir=True)
        obs_qts = PC_Thresh.get_quantiles(vrbl)

        print(f"Doing {fcstmin} min, {temporal_window} tw, for case {casestr}, "
                        f"for {vrbl}.")
        efss_data = {}
        fiss_data = {}
        e_npy0_f = "d01_3km_efss_{}_{}tw_{}min.npy".format(
                                    casestr,temporal_window,fcstmin)
        e_npy1_f = "d02_1km_efss_{}_{}tw_{}min.npy".format(
                                    casestr,temporal_window,fcstmin)
        f_npy0_f = "d01_3km_fiss_{}_{}tw_{}min.npy".format(
                                    casestr,temporal_window,fcstmin)
        f_npy1_f = "d02_1km_fiss_{}_{}tw_{}min.npy".format(
                                    casestr,temporal_window,fcstmin)

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
                t0 = time.time()
                threshs_fcst = PC_Thresh.get_threshs(vrbl,fmt)
                threshs_obs = PC_Thresh.get_threshs(vrbl,fmt)
                itr = loop_efss(caseutc,vrbl=vrbl,fcstmin=fcstmin,fmt=fmt,
                                middle_three=False)
                                # middle_three can be turned on if evaluating all
                                # otherwise we use all init times

                efss = []
                for i in list(itr):
                    efss.append(compute_efss(i,shuffled_copy(spatial_windows[fmt]),
                                    temporal_window))

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
                    efss_data[fmt][nhidx,thidx] = N.nanmean(efss_load)
                    fiss_data[fmt][nhidx,thidx] = N.nanmean(fiss_load)

            N.save(file=e_npy0,arr=efss_data['d01_3km'])
            N.save(file=e_npy1,arr=efss_data['d02_1km'])
            N.save(file=f_npy0,arr=fiss_data['d01_3km'])
            N.save(file=f_npy1,arr=fiss_data['d02_1km'])

            t1 = time.time()
            dt = t1-t0
            dtm = int(dt//60)
            dts = int(dt%60)
            print(f"FISS+eFSS calculation for all times took {dtm:d} min  {dts:d} sec.")

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
            fname = "{}-diffs_{}_{}min_{}tw.png".format(
                                score,casestr,fcstmin,temporal_window)
            fpath = os.path.join(outroot,"{}-diffs".format(score),vrbl,fname)
            utils.trycreate(fpath,isdir=False)

            fig,ax = plt.subplots(1)

            # y-axis is obs_qts
            # x-axis is neighbourhoods

            if score == "efss":
                _diffs = efss_data['d02_1km'][1:,:] - efss_data['d01_3km']
                kw = dict(vmin=-0.15,vmax=0.15)
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
                text = ax.text(j,i,t,ha="center",va="center",color="k",size=14,
                                    fontweight="bold")
            plt.colorbar(im,ax=ax)
            fig.tight_layout()

            fig.savefig(fpath)
            # pdb.set_trace()
        pass
    pass
    # pdb.set_trace()



if object_switch:
    print(stars,"DOING OBJECT COMPUTATIONS",stars)
    # Plot type
    #plot_what = 'ratio'; plot_dir = 'check_ratio'
    # plot_what = 'qlcs'; plot_dir = 'qlcs_v_cell'
    # plot_what = 'ecc'; plot_dir = "compare_ecc_ratio"
    # plot_what = 'extent'; plot_dir ='check_extent'
    # plot_what = "4-panel"; plot_dir = "four-panel"
    plot_what = "pca"; plot_dir = "pca_check"


    def plot_object_quicklook(i):
        fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem, load = i

        # Check for object save file for obs/fcst
        pk_fpath = get_object_picklepaths(fcst_vrbl,fcst_fmt,validutc,caseutc,
                                        initutc=initutc,mem=mem)
        # print(stars,"FCST OBJ QUICK PLOT DEBUG:",fcst_fmt,caseutc,initutc,validutc,mem)

        obj = utils.load_pickle(pk_fpath)
        if obj == "ERROR":
            pdb.set_trace()

        # QUICK LOOKS
        fcstmin = int(((validutc-initutc).seconds)/60)
        if fcstmin in range(30,210,30) and do_quicklooks and (mem == "m01"):
            ql_fname = "obj_{}_{:%Y%m%d%H}_{}min.png".format(
                        fcst_fmt,initutc,fcstmin)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what)
                print("Figure plotted to",outdir)
            else:
                print("Figure already created")
                pass
            # pdb.set_trace()
        return

    def compute_obj_fcst(i):
        fcst_vrbl,fcst_fmt,validutc,caseutc,initutc,mem, load = i

        #if validutc == initutc:
            # Forecast is just zeros.
        #    return None

        # Check for object save file for obs/fcst
        pk_fpath = get_object_picklepaths(fcst_vrbl,fcst_fmt,validutc,caseutc,
                                        initutc=initutc,mem=mem)
        print(stars,"FCST DEBUG:",fcst_fmt,caseutc,initutc,validutc,mem)

        if (not os.path.exists(pk_fpath)) or overwrite_pp:

            fcst_data, fcst_lats, fcst_lons = load_fcst_dll(fcst_vrbl,fcst_fmt,
                                    validutc,caseutc,initutc,mem)

            kw = get_kw(prodfmt=fcst_fmt,utc=validutc,mem=mem,
                        initutc=initutc,fpath=pk_fpath,threshold='auto')
            # pdb.set_trace()

            if "3km" in fcst_fmt:
                dx = 3.0
                # kw['threshold'] = 0
            elif "1km" in fcst_fmt:
                dx = 1.0
                # kw['threshold'] = 0
            else:
                assert True == False

            kw['dx'] = dx
            # 144 is default
            obj = ObjectID(fcst_data,fcst_lats,fcst_lons,**kw)
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
                pdb.set_trace()
                print("Overwriting",pk_fpath)
                fcst_data, fcst_lats, fcst_lons = load_fcst_dll(
                                        fcst_vrbl,fcst_fmt,
                                        validutc,caseutc,initutc,mem)

                kw = get_kw(prodfmt=fcst_fmt,utc=validutc,threshold='auto',
                            mem=mem,initutc=initutc,
                            fpath=pk_fpath)
                obj = ObjectID(fcst_data,fcst_lats,fcst_lons,dx=dx,**kw)
                utils.save_pickle(obj=obj,fpath=pk_fpath)

        # QUICK LOOKS
        fcstmin = int(((validutc-initutc).seconds)/60)
        if fcstmin in range(30,210,30) and do_quicklooks and (mem == "m01"):
        #if ((fcstmin % 30) == 0): # and (mem == "m01"):
            ql_fname = "obj_{}_{:%Y%m%d%H}_{}min.png".format(
                    fcst_fmt,initutc,fcstmin)
            outdir = os.path.join(outroot,"object_quicklooks",plot_dir)
            ql_fpath = os.path.join(outdir,ql_fname)
            if (not os.path.exists(ql_fpath)) or overwrite_output:
                obj.plot_quicklook(outdir=outdir,fname=ql_fname,what=plot_what)
            else:
                print("Figure already created")
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
            kw = get_kw(prodfmt=obs_fmt,utc=validutc,
                            threshold=40.5,fpath=fpath) # 96th pc
                            # threshold = 38.2, fpath=fpath)
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
            itr_fcst_2 = loop_obj_fcst(fcst_vrbl,fcstmins,
                            fcst_fmt,member_names,load=True)
                            # fcst_fmt,half_member_names,load=True)

            if ncpus > 1:
                with multiprocessing.Pool(ncpus) as pool:
                    results_fcst = pool.map(compute_obj_fcst,itr_fcst_2)
            else:
                for i in itr_fcst_2:
                    compute_obj_fcst(i)
            utils.save_pickle(obj=results_fcst,fpath=fcst_fpath)
        else:
            results_fcst = utils.load_pickle(fcst_fpath)
            itr_fcst_2 = loop_obj_fcst(fcst_vrbl,fcstmins=N.arange(60,210,30),
                                    fcst_fmt=fcst_fmt,members=['m01',],load=True)

            if ncpus > 1:
                with multiprocessing.Pool(ncpus) as pool:
                    pool.map(plot_object_quicklook,itr_fcst_2)
            else:
                for i in itr_fcst_2:
                    plot_object_quicklook(i)

        # pdb.set_trace()
        fname = "pca_model_{}.pickle".format(fcst_fmt)
        fpath = os.path.join(objectroot,fname)
        if (not os.path.exists(fpath)) or overwrite_pp:
            # TODO JRL: is this the right concat, or use the concat_two_dfs()?
            fcst_df = pandas.concat(results_fcst,ignore_index=True)
            CAT = Catalogue(fcst_df,tempdir=objectroot,ncpus=ncpus)
            pca, PC_df, features, scaler = CAT.do_pca()
            utils.save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,
                            features=features),fpath=fpath)

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
            utils.save_pickle(obj=dict(data=PC_df,pca=pca,scaler=scaler,
                                features=features),fpath=fpath['new'])

if do_object_pca:
    print(stars,"DOING PCA",stars)
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts) # + ['all',]
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



    # Plot performance diagrams for each case - can't separate by lead time...
    # Histogram of object by hour - is this fair? Hist shows broadly similar distr's
    # pdb.set_trace()
    # dom_codes = (d02_1km, d01_3km)
    # member = (m01, m02...)
    # casestr = (20160331...)
    # initstr = (2300 etc - depends on case)

    mns = member_names
    # modes = ("linear",)
    modes = ('cellular','linear')

    # if modes[0] == "linear":
    #    pickle_fname = "linear_match.pickle"
    #    pickle_fpath = os.path.join(objectroot,pickle_fname)
    #    MATCH = utils.load_pickle(pickle_fpath)
    #else:
        # mns = fifteen_member_names
        # mns = test_member_names
        #MATCH = get_MATCH(mns)
        # 4x5 (cases x init times) for mean performance for case


    plot_these = [
            ("20160331","1900"),
            ("20160331","2100"),
            ("20160331","2300"),

            ("20170501","1900"),
            ("20170501","2100"),
            ("20170501","2300"),

            ("20170502","2300"),
            ("20170502","0100"),
            ("20170502","0300"),

            ("20170504","2200"),
            ("20170504","0000"),
            ("20170504","0200"),
    ]

    def plot_huge_objperf(mode, do_anno_member=False):
        MATCH = get_MATCH(mns,modes=(mode,))
        fname = f"obj_perfdiag_allcases_{mode}.png"
        fpath = os.path.join(outroot,fname)

        fname2 = f"obj_perfdiag_allcases_{mode}_pub.png"
        fpath2 = os.path.join(outroot,fname2)

        fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(18,15))
        fig2,_axes2 =  plt.subplots(nrows=4,ncols=3,figsize=(12,15))
        axes2 = _axes2.flat
        for ax, (initutc, casestr, initstr) in zip(axes.flat,obj_perf_gen()):
            if (casestr,initstr) in plot_these:
                do_pub_fig = True
                PD2 = Performance(fpath=fpath2,legendkwargs=None,legend=False,
                                    fig=fig2,ax=next(axes2))

            else:
                do_pub_fig = False
            print(f"Plotting performance for {casestr} {initstr} {mode}")
            # mns set above, so we can sub-sample members
            PD = Performance(fpath=fpath,legendkwargs=None,legend=False,fig=fig,ax=ax)
            for dom_code, mode in itertools.product(('d02_1km','d01_3km'),
                                (mode,)):
                pod_all = []
                far_all = []
                for member in mns:
                    _2x2 = MATCH[dom_code][member][casestr][initstr][mode]['2x2']
                    fmt = dom_code # I think
                    pod = _2x2.get("POD")
                    far = _2x2.get("FAR")
                    pod_all.append(pod)
                    far_all.append(far)

                    annostr = "{}".format(member)
                    pad = 0.04
                    apad = 0.02
                    color_phys = get_color(fmt=fmt,mem=member)
                    pk = {'marker':MARKERS[fmt],'c':color_phys,'s':3.5*SIZES[fmt],
                                    'alpha':0.2}
                    lstr = get_nice(fmt)
                    PD.plot_data(pod=pod,far=far,plotkwargs=pk,label=lstr)
                    if do_pub_fig:
                        PD2.plot_data(pod=pod,far=far,plotkwargs=pk,label=lstr)
                    if do_anno_member:
                        PD.ax.annotate(annostr,xy=(1-far-apad,pod+apad),
                                    xycoords='data',fontsize='6',color='black',
                                    fontweight='medium')
                    if do_pub_fig and do_anno_member:
                        PD2.ax.annotate(annostr,xy=(1-far-apad,pod+apad),
                                        xycoords='data',fontsize='6',color='black',
                                        fontweight='medium')
                    # pdb.set_trace()

                    #pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                    #                'alpha':ALPHAS[fmt]}
                    # PD.save()
                pk_final = {'marker':'X','c':color_phys,'s':15*SIZES[fmt],
                                        'alpha':0.85,'edgecolors':('black',)}
                pod_mean = N.nanmean(N.array(pod_all))
                far_mean = N.nanmean(N.array(far_all))
                PD.plot_data(pod=pod_mean,far=far_mean,plotkwargs=pk_final)
                if do_pub_fig:
                    PD2.plot_data(pod=pod_mean,far=far_mean,plotkwargs=pk_final)
                #PD.ax.annotate("mean",xy=(1-far_mean-apad,pod_mean+apad),
                #                xycoords='data',fontsize='8',color=color_phys,
                #                    fontweight='bold')
                if do_pub_fig:
                    #PD2.ax.annotate("mean",xy=(1-far_mean-apad,pod_mean+apad),
                    #                xycoords='data',fontsize='8',color=color_phys,
                    #                    fontweight='bold')
                    PD2.ax.set_title(f"{casestr}:  {initstr} UTC\n")

            ax.set_title(f"{casestr}:  {initstr} UTC")
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

        fig2.tight_layout()
        fig2.savefig(fpath2)
        plt.close(fig2)

        return

    for mode in modes:
        plot_huge_objperf(mode)

if do_object_distr:
    print(stars,"DOING OBJ DISTRIBUTIONS",stars)
    # Plot distributions of objects in each domain and obs
    # TODO: run object ID for d02_3km?
    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)
    pdb.set_trace()
    # max_updraught
    # mean_updraught
    # distance_from_centroid
    # angle_from_centroid
    # distance_from_wcentroid
    # updraught_area_km
    attrs = ("max_updraught",)
    # pdb.set_trace()

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
    mode = "cellular"
    # mode = 'linear'
    mns = get_member_names(36)
    match_dict, megaframe = get_dom_match(mns, return_megaframe=True,
                                modes=(mode,))

    ### DIFFS ###

    # Do differences beteen difference domains/products
    # Positive diffs means d02 is higher.

    def write_row(df,n,key,val):
        df.loc[n,key] = val
        return

    def find_row(df,megaidx):
        row = df[(df['megaframe_idx'] == megaidx)]
        assert len(row) == 1
        return row

    def do_write_diff(df,n,oldkey,newkey,d02,d01):
        diff = d02.get(oldkey).values[0] - d01.get(oldkey).values[0]
        write_row(df,nm,newkey,diff)
        return

    DTYPES = {
                "d01_id":"i4",
                "d02_id":"i4",

                }

    prop_names = []

    # match_dict[member][mode][casestr][initstr][{'2x2','matches'}]

    megalist = []
    # loop over all matches, and join dicts together - alert if any clashes in keys
    for member in mns:
        for initutc, casestr, initstr in get_initutcs_strs():
            this_dict = match_dict[member][mode][casestr][initstr]['matches']

            for k,v in this_dict.items():
                if v is not None:
                    megalist.append((k,v[0]))

    # JRL TODO: something with BIAS, CSI etc information?
    # could also do that RDI with a PCA between the two domain cellular objects

    # pdb.set_trace()

    diff_df = utils.do_new_df(DTYPES,len(megalist))

    for nm,match in enumerate(megalist):
        if match is None:
            continue
        d02_obj_id, d01_obj_id = sorted(match)
        d01 = find_row(megaframe,d01_obj_id)
        d02 = find_row(megaframe,d02_obj_id)

        # pdb.set_trace()

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
        _all_props = ("area","eccentricity","equivalent_diameter","extent",
                        "max_intensity","mean_intensity","perimeter",
                        "ratio","longaxis_km","qlcsness",
                        "max_updraught","mean_updraught",
                        "max_lowrot", "max_midrot",
                        )
        props = ('area','extent','longaxis_km','eccentricity',
                    'max_intensity','max_updraught','max_lowrot','max_midrot')
        prop_diffs = [p + '_diff' for p in props]
        for prop, prop_diff in zip(props,prop_diffs):
            do_write_diff(diff_df,nm,prop,prop_diff,d02,d01)

    # These is the dataframe with no missing data (Nones)
    diff_df = diff_df[(diff_df['d01_id'] != 0.0)]
    # pdb.set_trace()

    LIMS = {
        "area_diff":[-300,300],
        "extent_diff":[-0.5,0.3],
        "longaxis_km_diff":[-15,20],
        # "qlcsness_diff":[-3,3],
        'eccentricity_diff':[-0.5,0.5],
        "max_intensity_diff":[-10,25],
        "max_updraught_diff":[-15,35],
        "max_lowrot_diff":[-20,100],
        "max_midrot_diff":[-50,350],
    }

    BINS = {
        "area_diff":90,
        "extent_diff":32,
        "longaxis_km_diff":55,
        "eccentricity_diff":40,
        "max_intensity_diff":35,
        "max_updraught_diff":35,
        "max_lowrot_diff":50,
        "max_midrot_diff":60,
    }

    NICESTR = {
        "area_diff":"Object area ($km^2$)",
        "extent_diff":"Extent (fraction)",
        "longaxis_km_diff":"Longest axis length ($km$)",
        # "qlcsness_diff":[-3,3],
        'eccentricity_diff':"Eccentricity",
        "max_intensity_diff":"Maximum comp. reflectivity ($dBZ$)",
        "max_updraught_diff":"Maximum updraft ($m\,s^{-1}$)",
        "max_lowrot_diff":"Maximum 0-2 km UH ($m^2\,s^{-2}$)",
        "max_midrot_diff":"Maximum 2-5 km UH ($m^2\,s^{-2}$)",
    }

    #from evac.plot.ridgeplot import RidgePlot
    #fname = "diffs_ridgeplot.png"
    #fpath = os.path.join(outroot,"ridgeplots",fname)
    #utils.trycreate(fpath)
    #RidgePlot(fpath).plot(diff_df,namelist=prop_diffs,)

    fname = f"hist_3x3_{mode}.png"
    fpath = os.path.join(outroot,"match_diffs",fname)
    utils.trycreate(fpath)
    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(11,11))
    loc_frac = (0.85,0.85)

    # palette = itertools.cycle(sns.cubehelix_palette(9,light=0.7,dark=0.3))
    lll = utils.generate_letters_loop(9)
    axit = axes.flat
    for n,p_d in enumerate(prop_diffs):
        ax = next(axit)
        ax.hist(diff_df[p_d],bins=BINS[p_d],color='grey')
        # ax.hist(diff_df[p_d],color=next(palette))
        ax.set_title(NICESTR[p_d])
        ax.set_facecolor('lightgray')
        x_locs = list(N.nanpercentile(diff_df[p_d],(5,25,50,75,95)).flatten())
        x_locs +=  [N.nanmean(diff_df[p_d]),]
        for n,x_loc in enumerate(x_locs):
            if n in (1,3):
                c = 'orange'
            elif n in (2,):
                # c = 'red'
                c = 'magenta'
            elif n in (0,4):
                continue
                c = 'yellow'
                continue
            elif n == 5:
                continue
                # c = 'magenta'
            ax.axvline(x_loc,color=c,linestyle='--',linewidth=2)
        ax.axvline(0,color='black',linewidth=3)
        ax.set_xlim(LIMS[p_d])
        utils.make_subplot_label(ax,f"({next(lll)})",loc_frac=loc_frac)

    ax = next(axit)
    ax.set_facecolor('lightgray')
    ax.hist(diff_df["centroid_gap_km"],bins=50,color='green')
    ax.set_xlim([0,50])
    ax.set_title("Gap in centroids (km)")
    utils.make_subplot_label(ax,f"({next(lll)})",loc_frac=(0.9,0.9))


    fig.tight_layout()
    fig.savefig(fpath)

    # Joint distributions

if do_object_windrose:
    # Do polar coordinate scatter plot for location of max updraft from centroid,
    # for cell objects only, assuming eccentricity = circle

    #fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    #obs_fmts = ("nexrad_3km","nexrad_1km")
    #all_fmts = list(fcst_fmts) + list(obs_fmts)
    #mf = load_megaframe(fmts=all_fmts)

    def cart2pol(x, y):
        rho = N.sqrt(x**2 + y**2)
        phi = N.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        # radius is rho
        # angle is phi
        x = rho * N.cos(phi)
        y = rho * N.sin(phi)
        return(x, y)

    if False:
        # EE to obs
        mns = get_member_names(16)
        MATCH, mf = get_MATCH(mns, return_megaframe=True)
    elif True:
        # EE3 to EE1
        mns = get_member_names(8)
        MATCH, mf = get_dom_match(mns,return_megaframe=True)
        # pdb.set_trace()

    # This is for d01-d02

    angles_all = {n:[] for n in (0,1)}
    distances_all = {n:[] for n in (0,1)}

    x_all = {n:[] for n in (0,1)}
    y_all = {n:[] for n in (0,1)}

    # for ID_A, ID_B in MATCH:
    for member in mns:
        for initutc, casestr, initstr in get_initutcs_strs():
            # matches = MATCH[member]['cellular'][casestr][initstr]['sorted']
            matches = MATCH[member]['linear'][casestr][initstr]['sorted']
            for ID_A, ID_B in matches:
                objA = mf[mf['megaframe_idx'] == ID_A]
                objB = mf[mf['megaframe_idx'] == ID_B]
                assert len(objA) == 1
                assert len(objB) == 1
                # ud_angle_from_centroid
                # ud_distance_from_centroid
                vals = N.zeros([2,2])
                for n,o in enumerate([objA,objB]):
                    v0 = o.ud_angle_from_centroid.values[0]
                    vals[n,0] = v0
                    v1 = o.ud_distance_from_centroid.values[0]
                    vals[n,1] = v1

                    x,y = pol2cart(v0, v1)

                    angles_all[n].append(v0)
                    distances_all[n].append(v1)

                    x_all[n].append(x)
                    y_all[n].append(y)

                # diffs = N.zeros([2])
                # for n in (0,1):
                    # diffs[n] = vals[1,n] - vals[0,n]


                # pdb.set_trace()


    # pdb.set_trace()
    for n,km in enumerate(["3km","1km"]):
        x = N.array(x_all[n]).flatten()
        y = N.array(y_all[n]).flatten()

        fname = f"ud_centroid_diffs_{km}.png"
        fpath = os.path.join(outroot,"polar",fname)
        fig,ax = plt.subplots()

        # Calculate the point density
        xy = N.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        xx, yy, zz = x[idx], y[idx], z[idx]

        ax.scatter(xx, yy, c=zz, s=50, edgecolor='')

        fig.tight_layout()
        utils.trycreate(fpath)
        fig.savefig(fpath)
        print(f"Saved {km} updraft--centroid diffs")

    # For matched objects only: what is the "mean vector" difference between EE3/1


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
        obj_id = oo.index
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
    # BROKEN
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


    # modes = ('cellular',)
    # modes = ('linear',)
    modes = ('cellular','linear')
    # JRL Sept 2019
    verif_meth = "IGN"
    # verif_meth = "BS"


    print("ß"*20,"READY TO WAFFLE!","ø"*20)
    fcst_doms = ('d02_1km','d01_3km')

    obs_IDs = {}
    # Loop first to build unique observed object set

    # Create new df with object ID, all props, and % chance of being forecast
    # % assigned is out of len(mns)
    # This assumes no matched object in a member is a "fail"

    # concat d01/d02 obs objects in big list

    bs_props = ["midrot_exceed_ID_0","midrot_exceed_ID_1",
    "midrot_exceed_ID_2","midrot_exceed_ID_3",
    "lowrot_exceed_ID_0","lowrot_exceed_ID_1",
    "lowrot_exceed_ID_2","lowrot_exceed_ID_3",
    ]
    # new_props = ["cell_yesno","line_yesno"]
    # all_props = bs_props + new_props
    all_props = bs_props

    all_bs = {p:{k:[] for k in ("1km","3km")} for p in bs_props}
    all_lt = {p:{k:[] for k in ("1km","3km")} for p in bs_props}
    overwrite_BS_pickles = True

    # Just those cells that are rotating
    all_meso_bs =  {p:{k:[] for k in ("1km","3km")} for p in bs_props}
    all_meso_lt =  {p:{k:[] for k in ("1km","3km")} for p in bs_props}

    for mode in modes:

        # mns = fifteen_member_names
        # if mode == "linear":
        #    mns = get_member_names(18)
        #    pickle_fname = "linear_match.pickle"
        #    pickle_fpath = os.path.join(objectroot,pickle_fname)
        #    MATCH = utils.load_pickle(pickle_fpath)

        #    mf_fname = "linear_mf.pickle"
        #    mf_fpath = os.path.join(objectroot,mf_fname)
        #    mf = utils.load_pickle(mf_fpath)
        #else:
        #    mns = get_member_names(18)
        #    MATCH, mf = get_MATCH(mns, return_megaframe=True)

        mns = get_member_names(36)
        MATCH, mf = get_MATCH(mns, return_megaframe=True,modes=(mode,))

        # parallelise if too slow?
        ### THIS IS FOR CREATING DATAFRAME OF OBSERVED OBJECTS PER RUN
        for initutc, casestr, initstr in obj_perf_gen():
            # This is a given initialisation (all 180 min, len(mns) members)
            for fcst_dom_code in fcst_doms:
                # d01 or d02
                dom, km = fcst_dom_code.split('_')

                # For mode in modes was here
                # Indent now reduced...

                # Save output at this level
                fname = f"{verif_meth}_prop_{casestr}_{initstr}_{mode}_{km}.pickle"
                fpath = os.path.join(objectroot,f"{verif_meth}_pickles",fname)

                if os.path.exists(fpath) and (not overwrite_BS_pickles):
                    bs = utils.load_pickle(fpath=fpath)
                else:
                # if True:
                    # Linear or cell
                    obs_objs = []
                    obs_IDs = set()
                    df_this_init = []

                    for member in mns:
                        # JRL TODO: this is where we convert to sorted_pairs
                        matches = MATCH[fcst_dom_code][member][casestr][
                                                initstr][mode]['matches']
                        # for fcst_ID, v in matches.items():
                        for obs_ID, v in matches.items():
                            # obs_ID is the "real" observed object - build pdf for each
                            if obs_ID not in obs_IDs:
                                the_row = mf[mf['megaframe_idx']==obs_ID]
                                assert the_row.shape[0] == 1
                                # pdb.set_trace()
                                # assert the_row.member == 'obs'
                                if the_row.member.values[0] == 'obs':
                                    obs_objs.append(the_row)
                                    obs_IDs.add(obs_ID)

                    # DataFrame with all observed objects that have been matched
                    if len(obs_objs)>0:
                        obs_obj_df = pandas.concat(obs_objs,ignore_index=True)
                    else:
                        print("No objects for this time - skipping")
                        continue

                    bs = {o:{} for o in obs_IDs}
                    # For a given observed object within this run....
                    for idx,oo in enumerate(obs_obj_df.itertuples()):
                        if (oo is None):# or (oo.member == "obs"):
                            continue
                        obj_ID = oo.megaframe_idx
                        # print(obj_ID)
                        fcst_bools = {p:[] for p in bs_props}
                        fcst_bool = {}

                        # Probs generated in this loop block
                        for member in mns:
                            # Get object that matched
                            matches = MATCH[fcst_dom_code][member][casestr][
                                            initstr][mode]['matches']

                            for obs_ID, v in matches.items():
                                if obs_ID == obj_ID:
                                    # fcst_bool =
                                    if v is None:
                                        # No object matched
                                        for prop in bs_props:
                                            fcst_bool[prop] = 0
                                    else:
                                        fcst_ID, _TI = v
                                        of = mf[mf['megaframe_idx']==fcst_ID]
                                        assert of.shape[0] == 1
                                        for prop in bs_props:
                                            #fcst_bool = of.get(prop).values[0]
                                            fcst_bool[prop] = int(
                                                    of.get(prop).values[0])

                                            #if not N.isnan(fcst_bool):
                                            #    fcst_bool = int(fcst_bool)
                                            # pdb.set_trace()
                                    for prop in bs_props:
                                        fcst_bools[prop].append(fcst_bool[prop])
                                        # print(len(fcst_bools[prop]))
                        # Assume all members have been checked, and missing = fail
                        obs_bools = {p:[] for p in bs_props}
                        oo_df = mf[mf['megaframe_idx']==obs_ID]

                        for prop in bs_props:
                            new_prop = f"{prop}_BS"
                            # obs_bools[prop].append(obs_bool)

                            # obs_val = 1 if obs_bools[prop] is True else 0

                            # obs_bools[prop] = int(oo_df.get(prop).values[0])
                            # obs_val = obs_bools[prop]

                            obs_val = oo_df.get(prop).values[0]
                            if not N.isnan(obs_val):
                                obs_val = int(obs_val)

                                isnan = N.isnan(fcst_bools[prop])
                                # number of non-nan members
                                total_nans = N.sum(isnan)
                                n_valid_mem = len(mns) - total_nans
                                prob_frac_mns = N.nansum(fcst_bools[prop])/n_valid_mem
                                # prob_frac_mns = len(mns)
                                debugg = False
                                if prop == "midrot_exceed_ID_0" and debugg:
                                    print(obj_ID,fcst_bools[prop],
                                            n_valid_mem,prob_frac_mns,
                                            obs_val)
                                    pdb.set_trace()
                                assert 0 <= prob_frac_mns <= 1

                                # Sept 2019 info edits

                                # bs_val = (prob_frac_mns - obs_val)**2
                                if prob_frac_mns == 1:
                                    prob_frac_mns = 0.99
                                elif prob_frac_mns == 0:
                                    prob_frac_mns = 0.01
                                # bs_val = -N.log2(prob_frac_mns)
                                c1 = (obs_val*N.log2(obs_val/prob_frac_mns))
                                c2 = ((1-obs_val)*N.log2(
                                        (1-obs_val)/(1-prob_frac_mns)))
                                if N.isnan(c1):
                                    bs_val = -1*(c2)
                                elif N.isnan(c2):
                                    bs_val = -1*(c1)
                                else:
                                    bs_val = -1*(c1+c2)
                                # pdb.set_trace()
                                assert not N.isnan(bs_val)
                                # ([0,1] - {1,0}) **2
                                # Perfect score: 100% and it is observed (0**2)
                                # Also, 0% and it is not observed.
                            else:
                                bs_val = N.nan

                            bs[obj_ID][prop] = bs_val

                            # The nans are related to missing mrms data
                            # if N.isnan(bs_val):

                            lt_val = ((oo.time-initutc).total_seconds())/60
                            bs[obj_ID]["leadtime"] = lt_val

                            all_bs[prop][km].append(bs_val)
                            all_lt[prop][km].append(lt_val)

                            if obs_val == 1:
                                all_meso_bs[prop][km].append(bs_val)
                                all_meso_lt[prop][km].append(lt_val)

                            #if prop == "midrot_exceed_ID_0":
                            #    pdb.set_trace()
                            # obs_obj_df.iloc[idx,new_prop]
                        # bs is a dictionary - each prop key accesses the BS for
                        # that object in this run. Get average fcst time?
                        # pdb.set_trace()
                        pass
                    pass
                    # pdb.set_trace()

                    # BS values need dumping in table with valid time

                    utils.save_pickle(fpath=fpath,obj=bs)
                    print("Saved to",fpath)
                pass
                # Dump objects into a big array
                # [lead_time,BS]
                # lead_times.append()

        fname = f"all_leadtimes_{verif_meth}_objs_{mode}.pickle"
        fpath = os.path.join(objectroot,f"{verif_meth}_pickles",fname)
        if os.path.exists(fpath) and (not overwrite_BS_pickles):
            all_lt = utils.load_pickle(fpath)
        else:
            utils.save_pickle(fpath=fpath,obj=all_lt)

        fname = f"all_brierscores_{verif_meth}_objs_{mode}.pickle"
        fpath = os.path.join(objectroot,f"{verif_meth}_pickles",fname)
        if os.path.exists(fpath) and (not overwrite_BS_pickles):
            all_bs = utils.load_pickle(fpath)
        else:
            utils.save_pickle(fpath=fpath,obj=all_bs)

        fname = f"all_leadtimes_{verif_meth}_mesos_{mode}.pickle"
        fpath = os.path.join(objectroot,f"{verif_meth}_pickles",fname)
        if os.path.exists(fpath) and (not overwrite_BS_pickles):
            all_meso_lt = utils.load_pickle(fpath)
        else:
            utils.save_pickle(fpath=fpath,obj=all_meso_lt)

        fname = f"all_brierscores_{verif_meth}_mesos_{mode}.pickle"
        fpath = os.path.join(objectroot,f"{verif_meth}_pickles",fname)
        if os.path.exists(fpath) and (not overwrite_BS_pickles):
            all_meso_bs = utils.load_pickle(fpath)
        else:
            utils.save_pickle(fpath=fpath,obj=all_meso_bs)

        # pdb.set_trace()
        # For domain, for init time:
        # Load all obs' BS scores
        # Sort by lead time
        # Do waffle plot ordered by lead time, coloured by BS from 0-1

        # Do eight props as left-to-right increasing lead time
        # One fig for 3km, another for 1km

        for meso_sw in ("meso-only",):#"all"):
            # meso_sw = "meso-only"
            # meso_sw = "all"

            # JRL TODO: pick the largest sbs_nice_n to have most sample size for
            # low percentile - and maybe do 10 nrows.
            # Going to test with just a few members, and run more for larger sample
            if meso_sw == "meso-only":
                # all_bs = all_meso_bs
                # all_lt = all_meso_lt

                if mode == "linear":
                    sbs_nice_n = 1500
                    nrows = 25

                else:
                    # sbs_nice_n = 450
                    # sbs_nice_n = 2000
                    sbs_nice_n = 1500
                    nrows = 25

            # from pywaffle import Waffle
            for km in ("1km","3km"):
                for prop in bs_props:
                    fname = f"waffle_{prop}_{km}_{meso_sw}_{mode}.png"
                    fpath = os.path.join(outroot,"waffle",mode,fname)
                    arr = N.array([all_lt[prop][km],all_bs[prop][km]])

                    df = pandas.DataFrame(data=arr.T,
                                columns=('lead_time','brier_score'))

                    # Do histograms for the total per member #



                    #arr_sort = N.sort(arr,axis=0)[0,:]
                    # Each array entry (axis 1) is now BS for that object
                    #assert len(N.where(arr_sort > 1)) == 0
                    #assert len(N.where(arr_sort < 1)) == 0

                    # pdb.set_trace()

                    # Subsample?
                    # total_raw = df_sort.shape[0]
                    # sbs_frac = 0.2
                    # sbs_n = int(sbs_frac * total_raw)

                    # nrows = 5
                    # Make right for reshaping - multiple of nrows

                    # sbs_nice_n = sbs_n - (sbs_n%nrows)
                    # sbs_nice_n = 125

                    # JRL Sept 2019: This seems very important regarding base rates?
                    # JRL: nah - just removing observed objects with a AWS val of NaN

                    # df.dropna(inplace=True)

                    # JRL Sept 2019: can we sub-sample at a higher value? We are ignoring
                    # the top pc level anyway
                    # Explain in manuscript why this was done
                    # pdb.set_trace()
                    nn0 = df.shape[0]
                    print(fname,"is",nn0)
                    #pdb.set_trace()
                    try:
                        df = df.sample(n=sbs_nice_n)
                    except ValueError: # not enough samples
                        print("Skipping this figure - not enough data")
                        continue

                    df = df.sort_values(by="brier_score")
                    nn1 = df.shape[0]

                    # Reshape for nrows (these rows should increase in time left to right)
                    data_arr = df.brier_score.values
                    # pdb.set_trace()

                    ncols = data_arr.size//nrows

                    # JRL: which order?!
                    data_arr = N.reshape(data_arr,[nrows,ncols],order="F")
                    # arr_waffle = N.reshape(arr_sort,(20,-1))

                    # Mask "misses", marked as 9999
                    data_arr = N.ma.masked_equal(data_arr,9999)

                    cm = M.cm.Reds_r if km=="1km" else M.cm.Blues_r
                    cm.set_bad('grey')

                    fig,ax = plt.subplots(figsize=(8,6))
                    # data_arr_mi = N.ma.masked_invalid(data_arr)
                    pcm = ax.pcolormesh(
                                            # data_arr_mi,
                                            data_arr,
                                            edgecolors="white",
                                            snap=True,
                                            cmap=cm,
                                            alpha=0.8,
                                            vmin=0.0,
                                            vmax=4.0,
                                            )
                    plt.colorbar(pcm,ax=ax,orientation='horizontal',)
                    #ax.colorbar.set_xlim([0,4])
                    ax.grid(False)
                    # draw thick black line to show x=0.5, and thinner for 0.25 and 0.75
                    plt.axvline(x=int(sbs_nice_n/(2*nrows)),color='k',lw=5)
                    plt.axvline(x=int(sbs_nice_n/(4*nrows)),color='k',lw=2.5)
                    plt.axvline(x=int((3*sbs_nice_n)/(4*nrows)),color='k',lw=2.5)

                    # ax.set_title(f"n_0: {nn0}; n_s: {nn1}",fontstyle='italic')
                    ax.axis("off")
                    ax.set_aspect("equal")
                    utils.trycreate(fpath,isdir=False)
                    fig.tight_layout()
                    fig.savefig(fpath)
                    print("Saved to",fpath)
                    # pdb.set_trace()
                    pass
# End of loop over both modes.

if do_case_outline:
    CASES_narr = collections.OrderedDict()
    CASES_narr[datetime.datetime(2016,3,31,0,0,0)] = datetime.datetime(2016,3,31,12,0,0)
    CASES_narr[datetime.datetime(2017,5,1,0,0,0)] = datetime.datetime(2017,5,1,12,0,0)
    CASES_narr[datetime.datetime(2017,5,2,0,0,0)] = datetime.datetime(2017,5,2,12,0,0)
    CASES_narr[datetime.datetime(2017,5,4,0,0,0)] = datetime.datetime(2017,5,4,12,0,0)

    CASES_EE3 = collections.OrderedDict()
    CASES_EE3[datetime.datetime(2016,3,31,0,0,0)] = datetime.datetime(2016,3,31,19,0,0)
    CASES_EE3[datetime.datetime(2017,5,1,0,0,0)] = datetime.datetime(2017,5,1,19,0,0)
    CASES_EE3[datetime.datetime(2017,5,2,0,0,0)] = datetime.datetime(2017,5,2,23,0,0)
    CASES_EE3[datetime.datetime(2017,5,4,0,0,0)] = datetime.datetime(2017,5,4,22,0,0)

    # representative times (from tor reports etc)
    RTs = collections.OrderedDict()
    RTs[datetime.datetime(2016,3,31,0,0,0)] = datetime.datetime(2016,4,1,0,15,0)
    RTs[datetime.datetime(2017,5,1,0,0,0)] = datetime.datetime(2017,5,1,19,30,0)
    RTs[datetime.datetime(2017,5,2,0,0,0)] = datetime.datetime(2017,5,3,3,15,0)
    RTs[datetime.datetime(2017,5,4,0,0,0)] = datetime.datetime(2017,5,5,1,30,0)

    # And the init that each representative time comes from
    RTs_init = collections.OrderedDict()
    RTs_init[datetime.datetime(2016,3,31,0,0,0)] = datetime.datetime(2016,3,31,23,0,0)
    RTs_init[datetime.datetime(2017,5,1,0,0,0)] = datetime.datetime(2017,5,1,19,0,0)
    RTs_init[datetime.datetime(2017,5,2,0,0,0)] = datetime.datetime(2017,5,3,3,0,0)
    RTs_init[datetime.datetime(2017,5,4,0,0,0)] = datetime.datetime(2017,5,5,1,0,0)

    # Load tornado report CSV to plot lat/lon


    # Load severe hail reports, or draw a swathe of SVR reports with Python?
    # Could plot hail reports as proxy for mesocyclonic activity.

    fig,_axes = plt.subplots(ncols=4,nrows=4,figsize=(12,12),tight_layout=True)
    fig_fname = "case_outline.png"
    fig_fpath = os.path.join(outroot,fig_fname)
    axes = _axes.flat

    fcmin = 10
    import sklearn.preprocessing as sklp
    letters = ['A','B','C','D']

    count = 0
    # For each case...
    for (caseutc, plotutc_NR),(_caseutc, initutc_EE) in zip(
                CASES_narr.items(),CASES_EE3.items()):
        casestr = utils.string_from_time('dir',plotutc_NR,strlen='hour')
        narr_fname = f'merged_AWIP32.{casestr}.3D'
        narr_fpath = os.path.join(narrroot,narr_fname)

        plotutc_EE = initutc_EE + datetime.timedelta(seconds=60*fcmin)
        # radarutc = initutc_EE + datetime.timedelta(seconds=60*180)
        radarutc = RTs[caseutc]
        initutc_radar = RTs_init[caseutc]

        # Load NARR data
        # G = GribFile(fpath)
        G = NARRFile(narr_fpath)
        data_925 = G.get("HGT",level="level 925")[0,0,:,:]
        data_500 = G.get("HGT",level="level 500")[0,0,:,:]

        # lats = G.lats
        # lons = G.lons

        # Plot 500 hPa height, 925 hPa height from NARR
        ax = next(axes)
        casecode = f"{letters[count]}-{casestr[:8]}"
        ax.set_title(f"1200 UTC (NARR) \n {casecode}",weight='bold')
        m0 = create_bmap(45.0,-65.0,25.0,-115.0,ax=ax)
        x,y = m0(G.lons,G.lats)
        m0.drawcoastlines()
        m0.drawmapboundary(fill_color='gray')
        m0.fillcontinents(color="lightgray",lake_color="gray")
        m0.drawstates()
        m0.drawcountries()
        count += 1

        kws = {'linewidths':0.7}

        # mediumorchid
        # purple
        m0.contour(x,y,data_500,colors='k',levels=N.arange(4000,6000,60),**kws)
        m0.contour(x,y,data_925,colors='mediumorchid',levels=N.arange(0,2000,40),**kws)

        # plot d01 domain for reference
        lats = N.load(os.path.join(extractroot,"d01_raw",casestr[:8],"lats.npy"))
        lons = N.load(os.path.join(extractroot,"d01_raw",casestr[:8],"lons.npy"))
        x,y = m0(lons,lats)
        color = "#1A85FF"
        m0.plot(x[0,:],y[0,:],color,lw=2)
        m0.plot(x[:,0],y[:,0],color,lw=2)
        m0.plot(x[len(y)-1,:],y[len(y)-1,:],color,lw=2)
        m0.plot(x[:,len(x)-1],y[:,len(x)-1],color,lw=2)

        # Plot shear and CAPE from EE3km
        from sklearn.preprocessing import MinMaxScaler

        # Load data
        kw = {'fcst_fmt':"d01_3km",'validutc':plotutc_EE,'initutc':initutc_EE,
                'caseutc':caseutc,'mem':"first_half"}

        DATA = {}
        NORM = {}
        # array size is (n_members,nlats,nlons)
        vrbls = ('u_shear01','v_shear01','CAPE_100mb',"SRH03","REFL_comp",
                    'u_shear06','v_shear06')
        for vrbl in vrbls:
            fcst_data, fcst_lats, fcst_lons = load_fcst_dll(vrbl,**kw)
            if vrbl == "REFL_comp":
                fcst_data[fcst_data < 0.0] = 0.0
            elif vrbl == "SRH03":
                fcst_data[fcst_data < 0.0] = 0.0

            fcst_data_norm = N.zeros_like(fcst_data)
            # Loop and normalise
            for nmem,mem in enumerate(get_member_names(18)):
                # norm_data = sklp.normalize(fcst_data[nmem,:,:])
                # norm_data = MinMaxScaler().fit_transform(fcst_data[nmem,...])
                # norm_data = normalise(fcst_data[nmem,:,:])
                norm_data = fcst_data[nmem,:,:]
                fcst_data_norm[nmem,...] = norm_data


            # z-scores, right?
            NORM[vrbl] = fcst_data_norm
            DATA[vrbl] = fcst_data

        # DATA['shear_u_all'], lats, lons = load_fcst_dll("u_shear01",**kw)
        # DATA['shear_v_all'], *_ = load_fcst_dll("v_shear01",**kw)
        # DATA['CAPE_all'], *_ = load_fcst_dll("CAPE_100mb",**kw)
        # DATA['SRH03_all'], *_ = load_fcst_dll("SRH03",**kw)
        # DATA['cref_all'], *_ = load_fcst_dll("REFL_comp",**kw)

        # Closest member to mean - pick three for each variable in descending order
        # Find member that is representative of all three variables
        MIs = N.zeros([len(vrbls),18])
        best_members = []
        for nv,vrbl in enumerate(vrbls):
            print(vrbl)
            for nmem,mem in enumerate(get_member_names(18)):
                # member data
                mem_slice = NORM[vrbl][nmem,:,:]
                # Mean for each grid point
                mean_slice = N.nanmean(NORM[vrbl],axis=0)

                # do mutual information for each member w/ mean_val
                hist_2d, x_edges, y_edges = N.histogram2d(mem_slice.ravel(),
                                                mean_slice.ravel(),bins=20)

                # convert bin count to fraction
                pxy = hist_2d / N.sum(hist_2d)
                # marginal for x over y, and y over x
                px = N.sum(pxy,axis=1)
                py = N.sum(pxy,axis=0)

                method = 2
                if method == 1:
                    import sklearn.metrics as skm
                    MI = skm.adjusted_mutual_info_score(px,py)
                elif method == 2:
                    # Multiply marginals by broadcast
                    px_py = px[:,None] * py[None,:]
                    # Only want non-zero values.
                    nzs = pxy > 0
                    MI = N.sum(pxy[nzs] * N.log(pxy[nzs]/px_py[nzs]))
                    # assert 0 <= MI <= 1
                MIs[nv,nmem] = MI

            max_MI = MIs[nv,:].max()
            min_MI = MIs[nv,:].min()
            print("Min MI is",min_MI)
            print("Max MI is",max_MI)
            idx = N.argmax(MIs[nv,:])
            best_member = get_member_names(18)[idx]
            print("Member with max is member",best_member)
            best_members.append(idx)

        # JRL: TODO use total mutual interest - just sum
        # Then plot the percentile of that member as a bar chart for the variables used
        # Output separately so i can use as inset
        # Also output colorbar separately (use the just_one_colorbar method?)
        # This shows the reader it is a representative member.

        mode_method = 2
        if mode_method == 1:
            moderesult = ss_mode(best_members)
            # (mode index, count for each mode index)
            modeidx, modecount = moderesult
            if modeidx.size == 1:
                overall_best_idx = modeidx[0]

            overall_best_member = get_member_names(18)[overall_best_idx]
            print("Using the mode, the best member overall is", overall_best_member)

        elif mode_method == 2:
            # total_MI = N.sum(MIs,axis=0)
            # overall_best_idx = N.argmax(total_MI)
            average_MI_bits = N.mean(MIs,axis=0)
            overall_best_idx = N.argmax(average_MI_bits)
            overall_best_member = get_member_names(18)[overall_best_idx]
            print("Using total MI, the best member overall is", overall_best_member)

        RMs = dict()
        for nv,vrbl in enumerate(vrbls):
            RMs[vrbl] = DATA[vrbl][overall_best_idx,:,:]

        # MIs is distribution of MI for this case/time - can be used to show
        # clustering over time for each variable. MI is unitless, so can be
        # averaged over mulitple variables to give the "closest member"

        ax = next(axes)
        phr = plotutc_EE.hour
        ax.set_title(f"{phr:02d}{fcmin} UTC (EE3km): {overall_best_member}")


        # second column: shear, cape, helicity from 3-km:
        # shear_01 as wind barbs
        # cape_100mb as contour of certain levels, or a shading that's alpha/etc
        # SRH03 as contour w/ hatching? [-200,1000+]
        # (>2 in?) svr hail reports labelled with warn time and observed time - separate?
        # Do higher in threshold to 'thin' reports, as their purpose here is as
        # a proxy for supercell rotation

        # m1 = create_bmap(50.0,-55.0,25.0,-130.0,ax=ax)

        m1 = create_bmap(fcst_lats.max(),fcst_lons.max(),
                        fcst_lats.min(),fcst_lons.min(),ax=ax,
                        proj='merc')
        x,y = m1(fcst_lons,fcst_lats)
        m1.drawstates(color='grey')
        m1.drawcountries(color='grey')
        cape = m1.contourf(x,y,RMs["CAPE_100mb"],
                        cmap=M.cm.Reds, levels=N.arange(200,2100,100),alpha=0.5)
        if count == 1:
            just_colorbar(fig,cape,os.path.join(outroot,"case_CAPE_cb.png"),
                                cb_xlabel="CAPE (J/kg)")

        srh_lvs = N.arange(100,2000,100)
        srh = m1.contour(x,y,gaussian_filter(RMs['SRH03'],sigma=2.5,mode='nearest'),
                        colors='grey',levels=srh_lvs)
        ax.clabel(srh, srh_lvs[::2], inline=1, fontsize=10,fmt='%d')
        fmt='%1.1f'

        nn = 26 # higher nn = fewer bars
        n = int(nn/2)
        bl = 7
        # Mask all under a certain amount? For both shears?
        m1.barbs(x[n::nn,n::nn],y[n::nn,n::nn],RMs['u_shear01'][n::nn,n::nn],
                    RMs['v_shear01'][n::nn,n::nn],length=bl,color='black')
        m1.barbs(x[::nn,::nn],y[::nn,::nn],RMs['u_shear06'][::nn,::nn],
                    RMs['v_shear06'][::nn,::nn],length=bl,color='blue')

        # Get severe reports csv
        report_csv_fname = "observed_tornadoes.csv"
        report_csv_fpath = os.path.join(extractroot,"storm_reports",report_csv_fname)
        tor_df = pandas.read_csv(report_csv_fpath)
        # pdb.set_trace()
        # Get only cases for this day.
        case_df = tor_df[tor_df['Case'] == int(casestr[:8])]

        # third column: cref (one time, with bunkers storm motion?) with reports
        # cref as contourf, as it's most important
        # tor reports labelled with warn time and observed time - separate?

        def annotate_naders(case_df,ax,m):
            for o in case_df.itertuples():
                x_pt, y_pt = m(o.Lon,o.Lat)
                # warn = int(o.get("Warning time").values[0])
                warn = int(o._7)
                # pdb.set_trace()
                method = 3
                if method == 1:
                    ax.annotate(warn,xy=(x_pt,y_pt),xycoords="data",
                        annotation_clip=False,zorder=1000,
                        fontsize=8,# fontstyle="italic",
                        fontweight='bold',color="red",)
                elif method == 2:
                    fontdict = {"color":"white","fontweight":"bold"}
                    # fontdict = None
                    bbox = dict(facecolor='red', alpha=0.6, edgecolor='k',
                                    linewidth=1)
                    ax.text(x_pt,y_pt,warn,fontdict=fontdict,bbox=bbox)
                else:
                    pass
            return

        rhr = radarutc.hour
        rmn = radarutc.minute
        fm = int((radarutc - initutc_radar).total_seconds()/60)

        ax = next(axes)
        ax.set_title(f"{rhr:02d}{rmn:02d} UTC (EE3km): {overall_best_member} ({fm} min)")

        kw = {'fcst_fmt':"d01_3km",'validutc':radarutc,'initutc':initutc_radar,
                        'caseutc':caseutc,'mem':overall_best_member}
        EE3_data, fcst_lats, fcst_lons = load_fcst_dll("REFL_comp",**kw)
        if vrbl == "REFL_comp":
            EE3_data[EE3_data < 0.0] = 0.0

        # m2 = create_bmap(50.0,-55.0,25.0,-130.0,ax=ax)
        m2 = create_bmap(fcst_lats.max(),fcst_lons.max(),
                        fcst_lats.min(),fcst_lons.min(),ax=ax,
                        proj='merc')
        x,y = m2(fcst_lons,fcst_lats)
        m2.drawstates(color='grey')
        m2.drawcountries(color='grey')
        S = Scales('cref')
        # m2.contourf(x,y,EE3_data,cmap=S.cm,levels=S.clvs,alpha=0.8)
        pcm = m2.pcolormesh(x,y,EE3_data,cmap=S.cm,vmin=S.clvs[0],
                            vmax=S.clvs[-1],alpha=0.5,
                            antialiased=True,)
        pcm.cmap.set_under('white')

        annotate_naders(case_df,ax,m2)

        # Load data for EE1km to compare
        ax = next(axes)
        ax.set_title(f"{rhr:02d}{rmn:02d} UTC (EE1km): {overall_best_member} ({fm} min)")

        kw = {'fcst_fmt':"d02_1km",'validutc':radarutc,'initutc':initutc_radar,
                'caseutc':caseutc,'mem':overall_best_member}
        EE1_data, fcst_lats, fcst_lons = load_fcst_dll("REFL_comp",**kw)
        if vrbl == "REFL_comp":
            EE1_data[EE1_data < 5.0] = 5.0

        # m2 = create_bmap(50.0,-55.0,25.0,-130.0,ax=ax)
        m3 = create_bmap(fcst_lats.max(),fcst_lons.max(),
                        fcst_lats.min(),fcst_lons.min(),ax=ax,
                        proj='merc')
        x,y = m3(fcst_lons,fcst_lats)
        m3.drawstates(color='grey')
        m3.drawcountries(color='grey')
        S = Scales('cref')
        cm2 = S.cm
        # m3.contourf(x,y,EE1_data,cmap=S.cm,levels=S.clvs,alpha=0.8)
        pcm = m3.pcolormesh(x,y,EE1_data,cmap=cm2,vmin=S.clvs[0],
                            vmax=S.clvs[-1],alpha=0.5,
                            antialiased=True,)
        pcm.cmap.set_under('white')

        annotate_naders(case_df,ax,m3)

        if count == 1:
            just_colorbar(fig,pcm,os.path.join(outroot,"case_cref_cb.png"),
                            cb_xlabel="Composite reflectivity (dBZ)")

        # fig.savefig(fig_fpath)
        # pdb.set_trace()

    # fig.tight_layout()
    fig.savefig(fig_fpath)#,bbox_inches='tight')
    plt.close(fig)

if do_object_infogain:
    # halfhours = N.arange(30,210,30)
    # tperiods = ['first_hour','second_hour','third_hour']
    # fcstmins = [60,120,180]
    modes = ('cellular','linear')

    uh_props = ["midrot_exceed_ID_0","midrot_exceed_ID_1",
        "midrot_exceed_ID_2","midrot_exceed_ID_3",
        "lowrot_exceed_ID_0","lowrot_exceed_ID_1",
        "lowrot_exceed_ID_2","lowrot_exceed_ID_3",]

    def n_p_gen(naive_probs, means, medians):
        for n_p in naive_probs:
            yield n_p, means, medians

    def plot(ax,EE3,EE1,fpath,bounds,vrbl='cref',H_all=None,H_noclearsky=None,
                Ho_all=None,Ho_ncs=None):
        worst_score, best_score = bounds
        # sns.despine(bottom=True,left=True)
        cols = ["#1A85FF","#D41159",]
        dstrs = ["3km","1km"]
        fig,ax = plt.subplots(1,figsize=(7,5))
        for n,data in enumerate((EE3,EE1)):
            # sns.set(rc ={'figure.figsize':(8,8)})

            #sns.stripplot(x=data,y=n,color=cols[n],
            #            dodge=True,jitter=True,
            #            alpha=0.25,zorder=1,)
            # pdb.set_trace()

            sns_plot = True; line_plot = False

            #if vrbl == "UH":
            #    dataset = data.flatten()[::1]
            #else:
            dataset = data.flatten()
            if sns_plot:
                # if vrbl == "UH": pdb.set_trace()
                sns.kdeplot(dataset,shade=True,
                color=cols[n],alpha=0.3,
                label=dstrs[n],ax=ax,
                # kernel='gau',bw='scott',
                # kernel='gau',bw='silverman',
                kernel='cos', bw=0.15,
                # kernel = 'cos', bw='scott',
                legend=True)
                # pdb.set_trace()

            elif line_plot:
        #if True:
            # Also do histogram
                bbins = [N.log2((n/36)/0.2) for n in range(0,37)]
                bbins[0] = N.log2(0.01/0.2)
                bbins[-1] = N.log2(0.99/0.2)


            # all_min = min(EE3.min(),EE1.min())
            # all_max = max(EE3.max(),EE1.max())
            # logbins = N.logspace(all_min,all_max,50)
            # logbins = N.logspace(-4,4,50)
            # logbins = N.concatenate([N.logspace(-1,0,25),N.logspace(0,1,25)])
            # logbins = 50
            # Need to fit a log-friendly KDE? Scatter?
            # pdb.set_trace()
            if False:
                ax.hist(data,bins=bbins,alpha=0.7,label=["3km","1km"],
            # ax.hist([EE3,EE1],alpha=0.7,label=["3km","1km"],
                    color=["#1A85FF","#D41159"],
                    histtype='stepfilled',
                #range=[N.floor(all_min),N.ceil(all_max)],
            #range=[-4,4],
                )
            if False:
                xlocs, counts = N.unique(data,return_counts=1)
                # vals,edges = N.histogram(data,bins=bbins)

                # ax.plot(edges,vals,label=dstrs[n],color=cols[n])
                ax.plot(xlocs,counts,label=dstrs[n],color=cols[n])

        #if True:
            # ax.set_xscale("symlog",)#linthreshx=1)

        ax.axvline(0,color='k',linewidth=1.5)
        ax.axvline(N.nanmean(EE3),color='#1A85FF',linewidth=1.3,linestyle='--')
        ax.axvline(N.nanmean(EE1),color='#D41159',linewidth=1.3,linestyle='--')
        #ax.axvline(N.median(EE3),color='#1A85FF',
        #                    linewidth=1.3,linestyle='--')
        #ax.axvline(N.median(EE1),color='#D41159',
        #                    linewidth=1.3,linestyle='--')
        # pdb.set_trace()
        print(f"Medians are {N.nanmedian(EE3)} and {N.nanmedian(EE1)}.")
        print(f"Means are {N.nanmean(EE3)} and {N.nanmean(EE1)}.")
        if vrbl != "UH":
            ax.axvline(worst_score,color='brown',linewidth=2)
        ax.axvline(best_score,color='green',linewidth=2)
        ax.set_ylabel("Density")
        # ax.legend()
        if vrbl == "UH":
            ax.set_xlim([min(N.nanmin(EE1),N.nanmin(EE3)),0.1])
            # ax.set_xlim([min(N.nanmin(EE1),N.nanmin(EE3)),0.5])

            ax.set_xlabel("Remaining UH entropy per object (bits)")

            if H_all is not None:
                # Annotate text with H for all and no-clear-sky

                s1 = f"Removing <0.2 probs, 1km gains {H_noclearsky:.1f} bits"
                s2 = f" (for all probs, 1km gains {H_all:.1f} bits)"
                ss = s1+s2
                anno = False
            elif Ho_all is not None:

                s3 = f"No <20% forecasts of obs=no: 1km gains {Ho_ncs:.3f} bits per object"
                s4 = f" (for all probs, 1km gained {Ho_all:.3f} bits)"
                ss = s3+s4
                anno = False
            else:
                anno = False
            if anno is True:
                fontkw = {
                        #'family':'serif',
                        'color':'darkred',
                        #'weight':'normal',
                        #'size':16,
                        }
                # ax.text(-6, -0.15, ss, fontdict=fontkw)
                ax.text(0.5, -0.13, ss, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes,
                    fontdict=fontkw)
            # Do table with sample size also?

        else:
            ax.set_xlim([worst_score-0.5,best_score+0.5])
            ax.set_xlabel("Object information gain (bits)")
        # ax.set_ylim([0,1.2])
        # pdb.set_trace()
        fig.tight_layout()
        fig.savefig(fpath)
        print("Saved to",fpath)
        return

    def get_H(arr3km,arr1km):
        arr3km = arr3km[~N.isnan(arr3km)]
        arr1km = arr1km[~N.isnan(arr1km)]
        uq_3 = N.unique(arr3km,return_counts=True)
        uq_1 = N.unique(arr1km,return_counts=True)
        H_3km = uq_3[0] * uq_3[1]
        H_1km = uq_1[0] * uq_1[1]
        H3_total = N.sum(H_3km)
        H1_total = N.sum(H_1km)
        # return H3_total - H1_total
        H_gain = (H1_total - H3_total)

        num_obj_3 = N.sum(uq_3[1])
        num_obj_1 = N.sum(uq_1[1])
        Ho_gain = (H1_total/num_obj_1) - (H3_total/num_obj_3)

        # pdb.set_trace()
        return H_gain, Ho_gain

    def compute_OSIG(i):
        naive_prob, means, medians = i
        # mns = get_member_names(18)
        # naive_prob = 0.2
        mns = member_names
        # Approximates 1/36 - could be set to even smaller values to greater
        # punish certainty - but that doesn't make sense with a model with
        # known underdispersion.
        constrain_minmax = (0.01,0.99)
        # Compute the prior ignorance (assuming a blanket SPC-style probability)
        # Could make this a manual field, drawn like a forecast!
        # Then do differences between the two
        # Plot as EE3 v EE1 distributions of info gain/loss of all objects
        # Compute stat.sig? Could do for two distributions!

        # 4x5 (cases x init times) for mean performance for case

        # modes = ('cellular',)#'linear')
        # modes = ("linear",)
        # fcst_doms = ('d01_3km',)

        #if modes[0] == "linear":
        #    pickle_fname = "linear_match.pickle"
        #    pickle_fpath = os.path.join(objectroot,pickle_fname)

        #    mf_fname = "linear_mf.pickle"
        #    mf_fpath = os.path.join(objectroot,mf_fname)

        #    MATCH = utils.load_pickle(pickle_fpath)
        #    mf = utils.load_pickle(mf_fpath)
        #else:
        #    MATCH, mf = get_MATCH(mns, return_megaframe=True)

        # tperiods = list(mf.leadtime_group.unique())
        fcst_doms = ('d02_1km','d01_3km')
        tperiods = ["first_hour","second_hour","third_hour"]
        modes = ('cellular','linear')
        # modes = ("cellular",)
        # All information gain by dom/mode
        all_IGs = {m:{fdc:N.zeros([2,0]) for fdc in fcst_doms} for m in modes}
        # Info gain per hour, and by dom/mode
        IGs_fname = "IGs_by_time.pickle"
        UHs_fname = "UHs_by_time.pickle"
        mUHs_fname = "mUHs_by_time.pickle"
        IGs_fpath = os.path.join(objectroot,IGs_fname)
        UHs_fpath = os.path.join(objectroot,UHs_fname)
        mUHs_fpath = os.path.join(objectroot,mUHs_fname)

        IG_debug = False
        if os.path.exists(IGs_fpath) and os.path.exists(mUHs_fpath) and (IG_debug is False):
            IGs_by_time = utils.load_pickle(IGs_fpath)
            UHs_by_time = utils.load_pickle(UHs_fpath)
            # This one ignores DKL ~ 0 for not-obs, not-fcst
            mUHs_by_time = utils.load_pickle(mUHs_fpath)
        else:

            IGs_by_time = {m:{fdc:{tp:[] for tp in tperiods} for fdc in fcst_doms} for m in modes}
            UHs_by_time = {m:{fdc:{tp:{p:[] for p in uh_props}
                for tp in tperiods} for fdc in fcst_doms} for m in modes}
            mUHs_by_time = {m:{fdc:{tp:{p:[] for p in uh_props}
                for tp in tperiods} for fdc in fcst_doms} for m in modes}

            for mode, fcst_dom_code in itertools.product(modes,fcst_doms):
                MATCH, mf = get_MATCH(mns, return_megaframe=True,modes=(mode,))
                for initutc, casestr, initstr in obj_perf_gen():
                    #if (initstr != "0200"):
                    #    continue
                    for hour in tperiods:
                        #if hour != "first_hour":
                        #    continue
                        # This is a given initialisation (all 180 min, len(mns) members)
                        dom, km = fcst_dom_code.split('_')

                        print("Doing IG for: ",
                            mode,fcst_dom_code,casestr,initstr)

                        prod_code = f"nexrad_{km}_obs"
                        earliest_utc = initutc - datetime.timedelta(seconds=60*10)
                        latest_utc = initutc +  datetime.timedelta(seconds=60*190)
                        obs_objs_IDs = mf[
                                (mf['prod_code'] == prod_code) &
                                (mf['conv_mode'] == mode) &
                                (mf['case_code'] == casestr) &
                                (mf['time'] >= earliest_utc) &
                                (mf['time'] <= latest_utc)
                                # (mf['leadtime_group'] == f"{hour}")
                                ].megaframe_idx

                        if False:
                            ootest = mf[
                                (mf['prod_code'] == prod_code) &
                                (mf['conv_mode'] == mode) &
                                (mf['case_code'] == casestr) &
                                (mf['time'] == datetime.datetime(2017,5,3,2,30,0)) &
                                (mf['member'] == "obs")
                                ]


                        #if hour != "first_hour":
                        #    pdb.set_trace()
                        # Check to see all unique
                        unique_IDs = obs_objs_IDs.unique()
                        assert len(unique_IDs) == len(obs_objs_IDs)
                        # IGs[mode][fcst_dom_code] = N.zeros([2,len(unique_IDs)])
                        IGs = N.zeros([2,len(unique_IDs)])

                        for noidx, obs_ID in enumerate(unique_IDs):
                            #if obs_ID == 1133:
                            #    pdb.set_trace()
                            eligible_uh = []
                            match_bools = N.zeros([len(mns)],dtype=bool)
                            for nm, member in enumerate(mns):
                                matches = MATCH[fcst_dom_code][member][casestr][
                                initstr][mode]['matches']
                                # for fcst_ID, v in matches.items():
                                # pdb.set_trace()
                                for o, v in matches.items():
                                    if v is None:
                                        # No match
                                        # JRL - IMPORTANT:
                                        # Note that matching the "wrong" mode will give False
                                        # should consider redoing with above/below MDI = 0.0
                                        # which is a bit of fuzzy logic.
                                        # Will be OK will cell/line split as it was done
                                        # to both obs/fcst fields so we hope error is random
                                        # match_bools[nm] = False
                                        pass
                                    elif o == obs_ID:
                                        # This is the object we're checking for
                                        # else, there's a match!
                                        match_bools[nm] = True
                                        eligible_uh.append(v)
                                        continue

                                        # If there's no match at all, maybe there are no objects?
                                        assert isinstance(match_bools[nm],N.bool_)

                            fcst_prob = N.sum(match_bools.astype(int))/len(mns)
                            fcst_prob = constrain(fcst_prob,minmax=constrain_minmax)
                            # For all observed objects for a domain in a given init (/20):
                            # What is the % of a match?

                            # naive_ign = -N.log2(naive_prob)
                            # fcst_ign = -N.log2(fcst_prob)
                            # info_gain = fcst_ign - naive_ign
                            info_gain = N.log2(fcst_prob/naive_prob)
                            #if (obs_ID == 582):
                            #    print(fcst_dom_code,info_gain)
                                # pdb.set_trace()
                            assert not N.isnan(info_gain)

                            # Here, IGs is every object's info gain

                            # IGs[mode][fcst_dom_code][0,noidx] = obs_ID
                            # IGs[mode][fcst_dom_code][1,noidx] = info_gain
                            IGs[0,noidx] = int(obs_ID)
                            IGs[1,noidx] = info_gain

                            # Now to append this into the big list
                            the_obj = mf[mf['megaframe_idx'] == obs_ID]
                            assert len(the_obj) == 1


                            # Compute UH exceedence here
                            # Loop over props (list of lists -> array later)
                            obj_utc = the_obj.time.values[0]
                            # determine which time window this is in
                            compare_utc = initutc + datetime.timedelta(
                                            # seconds=int(halfhour)*60)
                                            seconds=int(tperiods.index(hour))*60)

                            obj_utc = utils.dither_one_value(obj_utc)
                            dt = (compare_utc-obj_utc).total_seconds()
                            tt = int(abs(dt)/60)
                            halfhours = [0,60,120]
                            idx = N.searchsorted(halfhours,tt) - 1
                            hh = tperiods[idx]
                            for prop in uh_props:
                                obs_yesno = the_obj[prop].values[0]

                                # Find the fcst objects that matched
                                fcst_yesnos_raw = []
                                if len(eligible_uh) == 0:
                                    fprob = 0.01
                                else:
                                    for oidx in eligible_uh:
                                        fcst_obj = mf[mf['megaframe_idx'] == oidx[0]]
                                        fcst_yesnos_raw.append(fcst_obj[prop].values[0])
                                    # Loop, and count those that exceed this pc
                                    fcst_yesnos_arr = N.array(fcst_yesnos_raw)
                                    fprob = N.sum(fcst_yesnos_arr)/36
                                    fprob = min(0.99,fprob)
                                    fprob = max(0.01,fprob)
                                # compute DKL
                                c1 = ((1-obs_yesno)*N.log2((1-obs_yesno)/(1-fprob)))
                                c2 = (obs_yesno * N.log2(obs_yesno/fprob))
                                #c1 = ((1-fprob)*N.log2((1-fprob)/(1-obs_yesno)))
                                #c2 = (fprob*N.log2(fprob/obs_yesno))
                                # pdb.set_trace()


                                if N.isinf(c1):
                                    c1 = 0
                                elif N.isinf(c2):
                                    c2 = 0

                                if N.isnan(c1) and not N.isnan(c2):
                                    DKL = -1*c2
                                elif N.isnan(c2) and not N.isnan(c1):
                                    DKL = -1*c1
                                elif N.isnan(c1) and N.isnan(c2):
                                    DKL = N.nan
                                else:
                                    DKL = -(c1+c2)
                                # assert not N.isnan(DKL)
                                UHs_by_time[mode][fcst_dom_code][hh][prop].append(
                                    DKL)

                                if N.isnan(fprob) or N.isnan(obs_yesno):
                                    clearsky = True

                                elif (float(fprob) < 0.2) & (
                                            int(obs_yesno) == 0):
                                    clearsky = True
                                else:
                                    clearsky = False

                                if not clearsky:
                                    mUHs_by_time[mode][fcst_dom_code][
                                                    hh][prop].append(DKL)


                            #if the_obj.leadtime_group.values[0] != "first_hour":
                            #    pdb.set_trace()
                            # pdb.set_trace()

                            all_IGs[mode][fcst_dom_code] = N.concatenate(
                                [all_IGs[mode][fcst_dom_code],IGs],
                                axis=1,)


                            # Now, let's put it into the right hour
                            #if True:
                            #    hour = the_obj.leadtime_group.values[0]

                                # pdb.set_trace()

                            #if hour != "first_hour":
                            #    pdb.set_trace()
                            #pdb.set_trace()
                            # IGs_by_time[mode][fcst_dom_code][hour].append(IGs[1,:])
                            # IGs_by_time[mode][fcst_dom_code][hour] =
                            IGs_by_time[mode][fcst_dom_code][hh].append(info_gain)
                            # pdb.set_trace()
                            # 0 is 30 min or earlier.
                            # Then work from there.

            # Save to disk
            utils.save_pickle(obj=IGs_by_time, fpath=IGs_fpath)
            utils.save_pickle(obj=UHs_by_time, fpath=UHs_fpath)
            utils.save_pickle(obj=mUHs_by_time,fpath=mUHs_fpath)

        # pdb.set_trace()
        # 1/4 done
        # assert 1==0
        worst_score = N.log2(constrain_minmax[0]/naive_prob)
        best_score = N.log2(constrain_minmax[1]/naive_prob)
        print(f"Worst IG score is {worst_score:.2f} bits")

        # mode = 'cellular'
        # mode = 'linear'
        for mode in modes:

            naive = int(naive_prob*100)
            figsize = (6,4)
            fig,ax = plt.subplots(1,figsize=figsize)
            fname = f"obj_infogain_distrs_{mode}_allhours_{naive}pc.png"
            fpath = os.path.join(outroot,"info_gain",fname)
            utils.trycreate(fpath)

            EE3_IGs = all_IGs[mode]["d01_3km"][1,:].flatten()
            EE1_IGs = all_IGs[mode]["d02_1km"][1,:].flatten()

            # fig,ax = plt.subplots(1)
            # plot(ax,EE3_IGs,EE1_IGs,fpath,(worst_score,best_score))
            #fig.savefig(fpath)
            #pdb.set_trace()

            for nh in range(3):
                # hours = ("first_hour","second_hour","third_hour")
                hour = tperiods[nh]
                fig,ax = plt.subplots(1,figsize=figsize)
                ig_fname = f"obj_infogain_distrs_{mode}_{hour}_{naive}pc.png"
                ig_fpath = os.path.join(outroot,"info_gain",ig_fname)

                #EE3_IGs = N.array(IGs_by_time[mode]["d01_3km"][hour]).flatten()
                #EE1_IGs = N.array(IGs_by_time[mode]["d02_1km"][hour]).flatten()

                #EE3_IGs = N.array(IGs_by_time[mode]["d01_3km"][hour]).flatten()
                #EE1_IGs = N.array(IGs_by_time[mode]["d02_1km"][hour]).flatten()


                #aa = N.array(IGs_by_time[mode]["d01_3km"][hour])
                #pdb.set_trace()

                # plot(ax,EE3_IGs,EE1_IGs,fpath,(worst_score,best_score))
                plot(ax,N.array(IGs_by_time[mode]["d01_3km"][hour]),
                        N.array(IGs_by_time[mode]["d02_1km"][hour]),
                        ig_fpath,(worst_score,best_score),
                        )

                # Now UH
                for prop in uh_props:
                    fig,ax = plt.subplots(1,figsize=figsize)
                    uh_fname = f"uh_infogain_distrs_{mode}_{hour}_{prop}.png"
                    uh_fpath = os.path.join(outroot,"info_gain",uh_fname)

                    arr3km_a = N.array(UHs_by_time[mode]["d01_3km"][hour][prop])
                    arr1km_a = N.array(UHs_by_time[mode]["d02_1km"][hour][prop])
                    plot(ax,arr3km_a,arr1km_a,uh_fpath,(worst_score,best_score),
                        vrbl="UH",)

                    H_total, Ho_total = get_H(arr3km_a,arr1km_a)
                    H_total_all = H_total


                # for prop in uh_props:
                    fig,ax = plt.subplots(1,figsize=figsize)
                    uh_fname = f"muh_infogain_distrs_{mode}_{hour}_{prop}.png"
                    uh_fpath = os.path.join(outroot,"info_gain",uh_fname)

                    arr3km_b = N.array(
                            mUHs_by_time[mode]["d01_3km"][hour][prop])
                    arr1km_b = N.array(
                            mUHs_by_time[mode]["d02_1km"][hour][prop])

                    H_total_noclearsky, Ho_ncs = get_H(arr3km_b,arr1km_b)
                    plot(ax,arr3km_b,arr1km_b,uh_fpath,(worst_score,best_score),
                        vrbl="UH",
                        # H_all=H_total_all,H_noclearsky=H_total_noclearsky)
                        Ho_all=Ho_total,Ho_ncs=Ho_ncs)

                    #print("Info gain by 1km is", H_total_noclearsky,
                    #    "bits for",prop,"and removing clear sky")

                    # Annotate both the "all" and "clear sky" bits gain
                    # H_total_clearsky and H_total_all
                    # ax.text()



                # Putthe mean/median into array for bar chart
                means[0,nh] = N.nanmean(EE3_IGs)
                means[1,nh] = N.nanmean(EE1_IGs)

                medians[0,nh] = N.nanmedian(EE3_IGs)
                medians[1,nh] = N.nanmedian(EE1_IGs)

                # Note we use three bins for times, which isn't great, but
                # sample size probably too small for e.g. 30 min bins, also
                # not fair that 180 min of forecasts can match with 220 min of obs?
                # fig.savefig(fpath)
        return

    # naive_probs = N.arange(0.1,1.0,0.1)
    naive_probs = (0.1,)
    # arrays for means for each [dx,tperiod,]

    means = utils.generate_shared_arr([2,3,len(naive_probs)],dtype=float)
    medians = utils.generate_shared_arr([2,3,len(naive_probs)],dtype=float)

    # means = N.zeros([2,3,len(naive_probs)])]

    # Probabilistic object-information-gain!

    # naive_prob% of a user at a point being affected by a (linear, cellular) object,
    # with the usual object-ID tolerances and assumptions

    fpath_means = "./OSIG_means.npy"
    fpath_medians = './OSIG_medians.npy'

    # loaded = False
    gg = n_p_gen(naive_probs,means,medians)

    if os.path.exists(fpath_means) and os.path.exists(fpath_medians):
        means = N.load(file=fpath_means)
        medians = N.load(file=fpath_medians)
    else:
        if ncpus > 1:
            with multiprocessing.Pool(ncpus) as pool:
                pool.map(compute_OSIG,gg)
        else:
            for g in gg:
                compute_OSIG(g)




    # N.save(file=fpath_means,arr=means)
    # N.save(file=fpath_medians,arr=medians)

    # CR-IG between the two?

    # MAYBE need to plot histogram w/o smoothing.

    # Bar chart of (x-axis) hour 1/2/3 and (y) median/mean OSIG
    fig,ax = plt.subplots(1)
    # pdb.set_trace()

    # fit a curve once we have the bars  - log? find ref from chaos lit

    # asymptotic curve?


if do_performance:
    fcst_fmts = ("d01_3km","d02_1km")
    mns = get_member_names(18)

    def compute_perf(i):
        caseutc, initutcs, fmt, vrbl, fcstmin, thresh, member = i
        # The 5 init times
        POD = N.zeros([5])
        FAR = N.zeros_like(POD)

        for ninit,initutc in enumerate(initutcs):
            validutc = initutc + datetime.timedelta(seconds=60*int(fcstmin))
            fcst_data, obs_data = load_both_data(fcst_vrbl=vrbl,fcst_fmt=fmt,
                            validutc=validutc,caseutc=caseutc,initutc=initutc,
                            mem=member)
            DS = DetScores(fcst_arr=fcst_data,obs_arr=obs_data,thresh=thresh,
                            overunder='over')
            POD[ninit] = DS.get("POD")
            FAR[ninit] = DS.get("FAR")
        # print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return (POD,FAR,caseutc,fmt,vrbl,fcstmin,thresh,member)

    def loop_perf():
        for caseutc, initutcs in CASES.items():
            for fmt, vrbl, fcstmin, thresh, member in itertools.product(
                    fcst_fmts,vrbls,fcstmins,threshs,mns):
                yield caseutc, initutcs, fmt, vrbl, fcstmin, thresh, member


    vrbls = ("REFL_comp",)
    threshs = (15,30,40,50)
    fcstmins = (30,60,90,120,150,180)

    nvrbls = len(vrbls)
    nthreshs = len(threshs)
    nfms = len(fcstmins)
    nmems = len(mns)

    # POD and FAR saved here
    fpath_allpod = './allall_pod.npy'
    fpath_allfar = './allall_far.npy'

    if os.path.exists(fpath_allpod) and (not overwrite_perf):
        allpod = N.load(fpath_allpod)
        allfar = N.load(fpath_allfar)
    else:
        # [format, case, vrbl, fcstmin, thresh, member, init-time.]
        allpod = N.zeros([2,4,nvrbls,nfms,nthreshs,nmems,5])
        allfar = N.zeros_like(allpod)

        with multiprocessing.Pool(ncpus) as pool:
            results = pool.map(compute_perf,loop_perf())

        for r in results:
            POD,FAR,caseutc,fmt,vrbl,fcstmin,thresh,member = r
            nfmt = fcst_fmts.index(fmt)
            ncase = list(CASES.keys()).index(caseutc)
            nvrbl = vrbls.index(vrbl)
            nfm = fcstmins.index(fcstmin)
            nthresh = threshs.index(thresh)
            nmem = mns.index(member)

            allpod[nfmt,ncase,nvrbl,nfm,nthresh,nmem,:] = POD
            allfar[nfmt,ncase,nvrbl,nfm,nthresh,nmem,:] = FAR

        N.save(file=fpath_allpod,arr=allpod)
        N.save(file=fpath_allfar,arr=allfar)

    # vecmags,vecdirs = utils.combine_wind_components(dfar,dpod)

    PADS = {
    0:[-0.15,0.065],
    1:[-0.08,0.03],
    2:[-0.08,0.03],
    3:[-0.15,0.03],
        }
    letters = ["A","B","C","D"]
    # For plotting basic perf diags for d01, d02.
    switch_a = 1
    skip_tier1 = 0
    for vrbl, fcstmin, thresh in itertools.product(
            vrbls, fcstmins, threshs):
        if skip_tier1: continue
        fname = f"perf_diag_{vrbl}_{fcstmin}min_{thresh}th.png"
        fpath = os.path.join(outroot,"perfdiag_basic",fname)
        PDx = Performance(fpath=fpath,legendkwargs=None,legend=True)
        for ncase, caseutc in enumerate(CASES.keys()):
            casestr = utils.string_from_time('dir',caseutc,strlen='day')

            nfmt = fcst_fmts.index(fmt)
            nvrbl = vrbls.index(vrbl)
            nfm = fcstmins.index(fcstmin)
            nthresh = threshs.index(thresh)

            # These are 3D?
            pod_slice = allpod[:,ncase,nvrbl,nfm,nthresh,:,:]
            far_slice = allfar[:,ncase,nvrbl,nfm,nthresh,:,:]

            assert pod_slice.ndim == 3

            # Do diffs?

            # Average over initutcs
            pod_slice_aveinit = N.nanmean(pod_slice, axis=2)
            far_slice_aveinit = N.nanmean(far_slice, axis=2)

            # Now 2D (formats and members)
            # Average over members
            pod_slice_avemem = N.nanmean(pod_slice_aveinit, axis=1)
            far_slice_avemem = N.nanmean(far_slice_aveinit, axis=1)

            for nfmt, fmt in enumerate(fcst_fmts):
                lstr = get_nice(fmt) if not ncase else None

                pk = {'marker':MARKERS[fmt],'c':COLORS[fmt],'s':SIZES[fmt],
                        'alpha':ALPHAS[fmt]}
                farpt = far_slice_avemem[nfmt]
                podpt = pod_slice_avemem[nfmt]
                PDx.plot_data(pod=podpt,far=farpt,
                                    plotkwargs=pk,label=lstr)

                if fmt == "d01_3km":
                    annostr = "{}-{}".format(letters[ncase],casestr)
                    PDx.ax.annotate(annostr,xy=((1-farpt)+PADS[ncase][0],N.mean(podpt)+PADS[ncase][1]),
                                    xycoords='data',fontsize=10,color='black',
                                    fontweight='bold',annotation_clip=False)
        PDx.save()
        print("Saved to",fpath)

    # For plotting difference heatmaps
    dpod = allpod[1,...] - allpod[0,...]
    dfar = allfar[1,...] - allfar[0,...]

    vecmags,vecdirs = utils.combine_wind_components(dfar,dpod)

    vecmags[vecdirs > 45] *= -1
    vecmags[vecdirs < 225] *= -1

    kw = dict(vmin=-0.15,vmax=0.15)
    for (nvrbl,vrbl), (ncase,caseutc), in itertools.product(
            enumerate(vrbls),enumerate(CASES.keys()),):
        casestr = utils.string_from_time('dir',caseutc,strlen='day')
        fname = f"perfdiag_diffs_{vrbl}_{casestr}.png"
        fpath = os.path.join(outroot,"perfdiag_diff-heatmaps",fname)
        utils.trycreate(fpath)

        mag_slice = vecmags[ncase,nvrbl,:,:,:,:]
        hm_arr = N.mean(mag_slice,axis=(2,3))

        fig,ax = plt.subplots(1)
        data = hm_arr.T
        im = ax.imshow(data,cmap='vse_diverg', **kw)

        nx,ny = data.shape
        ax.set_yticks(N.arange(nx))
        ax.set_xticks(N.arange(ny))

        ax.set_xticklabels(fcstmins)
        ax.set_xlabel("Forecast minute")

        ax.set_yticklabels(threshs)
        ax.set_ylabel("dBZ threshold")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        for i,j in itertools.product(range(nx),range(ny)):
            t = "{:.2f}".format(data[i,j])
            text = ax.text(j,i,t,ha="center",va="center",color="k",size=14,
                                fontweight="bold")
        plt.colorbar(im,ax=ax)
        fig.tight_layout()

        fig.savefig(fpath)
        print("Saved to",fpath)


if do_one_objectID:
    initutcs = get_all_initutcs()
    objectroot = "/Volumes/LaCie/VSE_dx/object_instances"

    fms = N.arange(30,210,30)

    def do_plot(i):
        validutc,caseutc,casestr,initutc,initstr = i
        fm = int((validutc-initutc).total_seconds()/60)
        obj_fpath = get_object_picklepaths(vrbl="REFL_comp",
                        fmt="d02_1km",
                        validutc=validutc,caseutc=caseutc,
                        initutc=initutc,mem="m01")
        obj = utils.load_pickle(obj_fpath)
        outdir = os.path.join(outroot,"object_example")
        fname = f"obj_d02_1km_{casestr}_{initstr}_{fm}min.png"
        plot_what = 'pca'
        # pdb.set_trace()
        obj.plot_quicklook(outdir=outdir,fname=fname,what=plot_what)
        return

    def gen_ex_loop():
        for caseutc, *_, initutc in all_init_sort():
            casestr = utils.string_from_time('dir',caseutc,strlen='day')
            initstr = f"{initutc.hour:02d}{initutc.minute:02d}"
            # print("Doing examples for {casestr}    {initstr}")
            for fm in fms:
                validutc = initutc + datetime.timedelta(seconds=int(fm)*60)
                yield validutc,caseutc,casestr,initutc,initstr

    # Now, loop and plot
    if ncpus > 1:
        with multiprocessing.Pool(ncpus) as pool:
            pool.map(do_plot,gen_ex_loop())
    else:
        for g in gen_ex_loop():
            do_plot(g)
    print("Done.")

if do_qlcs_verif:
    # Get all linear/complex objects (fcst, obs)
    # Subset by eccentricity for QLCS only
    # Get UH exceedance binaries for object in obs from megaframe
    # Then look in each ensemble member within x km and x min of that max
    # Assumed that supercell objects won't contaminate this with small x km.

    fcst_fmts = ("d01_3km","d02_1km",)#"d02_3km")
    obs_fmts = ("nexrad_3km","nexrad_1km")
    all_fmts = list(fcst_fmts) + list(obs_fmts)
    megaframe = load_megaframe(fmts=all_fmts)

    if False:
        obsframe = megaframe[(megaframe['member'] == 'obs') &
                            (megaframe['qlcsness'] > 0.5) &
                            (megaframe['eccentricity'] > 0.85)
                            ]

        fcstframe = megaframe[(megaframe['member'] != 'obs') &
                            (megaframe['qlcsness'] > 0.5) &
                            (megaframe['eccentricity'] > 0.85)
                            ]

    linear_frame = megaframe[
                        (megaframe['qlcsness'] > 0.5) &
                        (megaframe['eccentricity'] > 0.85)
                        ]

    # pdb.set_trace()


    # Match these groups, iterating over ensemble members?

    mns = get_member_names(18)
    MATCH, mf = get_MATCH(mns, return_megaframe=True, modes=('linear',),
                            custom_megaframe=linear_frame)

    pickle_fname = "linear_match.pickle"
    pickle_fpath = os.path.join(objectroot,pickle_fname)

    mf_fname = "linear_mf.pickle"
    mf_fpath = os.path.join(objectroot,mf_fname)

    if not os.path.exists(pickle_fpath):
        utils.save_pickle(obj=MATCH,fpath=pickle_fpath)
    if not os.path.exists(mf_fpath):
        utils.save_pickle(obj=mf,fpath=mf_fpath)

    modes = ("linear",)

if do_spurious_example:
    # Get fcst/obs data
    fcst_data_3, fcst_lats_3, fcst_lons_3 = load_fcst_dll(
            "REFL_comp","d01_3km",
            datetime.datetime(2016,3,31,19,30,0),
            datetime.datetime(2016,3,31,0,0,0),
            datetime.datetime(2016,3,31,19,0,0),
            "m01")

    fcst_data_1, fcst_lats_1, fcst_lons_1 = load_fcst_dll(
            "REFL_comp","d02_1km",
            datetime.datetime(2016,3,31,19,30,0),
            datetime.datetime(2016,3,31,0,0,0),
            datetime.datetime(2016,3,31,19,0,0),
            "m01")

    obs_data, obs_lats, obs_lons = load_obs_dll(
        datetime.datetime(2016,3,31,19,30,0),
        datetime.datetime(2016,3,31,0,0,0),
        fcst_vrbl = "REFL_comp",
        fcst_fmt = "d02_1km",)

    """
    validutc,caseutc,obs_vrbl=None,fcst_vrbl=None,obs_fmt=None,
                    fcst_fmt=None,return_ll=True):

    fcst_data_3, _obs_data = load_both_data("REFL_comp","3km",
                            datetime.datetime(2016,3,31,19,30,0),
                            datetime.datetime(2016,3,31,0,0,0),
                            datetime.datetime(2016,3,31,19,0,0),
                            "m01")

    fcst_data_1, obs_data = load_both_data("REFL_comp","1km",
                            datetime.datetime(2016,3,31,19,30,0),
                            datetime.datetime(2016,3,31,0,0,0),
                            datetime.datetime(2016,3,31,19,0,0),
                            "m01")
    """
    fig,axes = plt.subplots(ncols=3,figsize=(15,8))

    # OBS
    ax = axes.flat[0]
    m0 = create_bmap(obs_lats.max(),obs_lons.max(),
                    obs_lats.min(),obs_lons.min(),ax=ax,
                    proj='merc')
    x,y = m0(obs_lons,obs_lats)
    m0.drawstates(color='grey')
    m0.drawcountries(color='grey')
    S = Scales('cref')
    # m2.contourf(x,y,EE3_data,cmap=S.cm,levels=S.clvs,alpha=0.8)
    pcm = m0.pcolormesh(x,y,obs_data,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')

    # EE3
    ax = axes.flat[1]
    m1 = create_bmap(fcst_lats_3.max(),fcst_lons_3.max(),
                    fcst_lats_3.min(),fcst_lons_3.min(),ax=ax,
                    proj='merc')
    x,y = m1(fcst_lons_3,fcst_lats_3)
    m1.drawstates(color='grey')
    m1.drawcountries(color='grey')
    S = Scales('cref')
    # m2.contourf(x,y,EE3_data,cmap=S.cm,levels=S.clvs,alpha=0.8)
    pcm = m1.pcolormesh(x,y,fcst_data_3,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')

    # EE1
    ax = axes.flat[2]
    m2 = create_bmap(fcst_lats_1.max(),fcst_lons_1.max(),
                    fcst_lats_1.min(),fcst_lons_1.min(),ax=ax,
                    proj='merc')
    x,y = m2(fcst_lons_1,fcst_lats_1)
    m2.drawstates(color='grey')
    m2.drawcountries(color='grey')
    S = Scales('cref')
    # m2.contourf(x,y,EE3_data,cmap=S.cm,levels=S.clvs,alpha=0.8)
    pcm = m2.pcolormesh(x,y,fcst_data_1,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')

    fig.tight_layout()
    spurious_fpath = os.path.join(outroot,"spurious_compare.png")
    fig.savefig(spurious_fpath)

if do_oID_example:
    data_folder = os.path.join(objectroot,"20160331")
    data_fname = "objects_REFL_comp_d02_1km_20160331_1900_1930_m01.pickle"
    # data_fname = "objects_REFL_comp_d01_3km_20160331_1900_1930_m01.pickle"
    data_fpath = os.path.join(data_folder,data_fname)
    print(data_fpath)

    obj = utils.load_pickle(data_fpath)
    # pdb.set_trace()

    mf = load_megaframe(fmts=None)

    def do_label_pca(bmap,ax,discrim_vals=(-0.5,0.5)):
        # JRL: change to lookup in megaframe.

        locs = dict()
        obj_discrim = N.zeros_like(obj.OKidxarray)
        for o in obj.objects.itertuples():
            lab = "{:0.2f}".format(o.qlcsness)
            locs[lab] = (o.centroid_lat,o.centroid_lon)
            id = int(o.label)


            # TODO: look up object ID in megaframe and determine MDI
            # what is obj_idx?
            # Find object in DataFrame
            obj_df = mf[
                    (mf['centroid_row'] == o.centroid_row) &
                    (mf['centroid_col'] == o.centroid_col) &
                    (mf['fpath_save'] == data_fpath)
                    ].megaframe_idx

            # pdb.set_trace()
            oidx = obj_df.iloc[0]

            qlcsness = mf.iloc[oidx].qlcsness

            if qlcsness < discrim_vals[0]:
                discrim = 1
            elif qlcsness > discrim_vals[1]:
                discrim = 3
            else:
                discrim = 2
            obj_discrim = N.where(obj.idxarray == id, discrim, obj_discrim)
            bmap.contour(x,y,obj_discrim,colors=['k',],linewidths=0.5)

        marr = N.ma.masked_where(obj_discrim < 1, obj_discrim)
        mraw = N.ma.masked_where(obj.raw_data < 1, obj_discrim)

        pcm = bmap.pcolormesh(data=marr,x=x,y=y,alpha=0.5,vmin=1,vmax=3,
                                cmap=M.cm.get_cmap("magma",3),)


    def label_objs(bmap,ax,locs):
        for k,v in locs.items():
            xpt, ypt = bmap(v[1],v[0])
            # bbox_style = {'boxstyle':'square','fc':'white','alpha':0.5}
            # bmap.plot(xpt,ypt,'ko',)#markersize=3,zorder=100)
            # ax.text(xpt,ypt,k,ha='left',fontsize=15)
            ax.annotate(k,xy=(xpt,ypt),xycoords="data",zorder=1000,
                            fontsize=7,fontweight='bold')
            # pdb.set_trace()
        return


    fig,axes = plt.subplots(1,3,figsize=(9,4))
    fcst_lats = obj.lats
    fcst_lons = obj.lons

    for n,ax in enumerate(axes.flat):
        # if n!=0:
        #     continue
        print("Plotting subplot #",n+1)
        # bmap = create_bmap(ax=ax)
        bmap = create_bmap(fcst_lats.max(),fcst_lons.max(),
                        fcst_lats.min(),fcst_lons.min(),ax=ax,
                        proj='merc')
        x,y = bmap(fcst_lons,fcst_lats)
        S = Scales(vrbl='REFL_comp')

        if n == 0:
            pcm = bmap.pcolormesh(x,y,obj.raw_data,cmap=S.cm,vmin=S.clvs[0],
                                vmax=S.clvs[-1],alpha=0.5,
                                antialiased=True,)
            pcm.cmap.set_under('white')
            ax.set_title("Raw data")

        elif n == 1:
            locs = dict()
            for o in obj.objects.itertuples():
                locs[str(int(o.label))] = (o.weighted_centroid_lat,
                    o.weighted_centroid_lon)
            pcm = bmap.pcolormesh(x,y,obj.object_field,cmap=S.cm,vmin=S.clvs[0],
                                vmax=S.clvs[-1],alpha=0.5,
                                antialiased=True,)
            # 45.7 is the 96th percentile
            bmap.contour(x,y,obj.raw_data,levels=[45.7],color='k',linewidths=0.6)
            pcm.cmap.set_under('white')

            # label_objs(bmap,ax,locs)
            ax.set_title("Intensity array")


        elif n==2:
            do_label_pca(bmap,ax)

            mode_names = ["","Cellular","Ambiguous","Linear/Complex"]
            def format_func(x,y):
                return mode_names[x]

            # This function formatter will replace integers with target names
            formatter = plt.FuncFormatter(lambda val, loc: mode_names[val])

            # We must be sure to specify the ticks matching our target names
            cax = fig.add_axes([0.75, 0.1, 0.2, 0.04])
            #cax.set_xlim(0.5, 3.5)
            fig.colorbar(pcm,cax=cax,ticks=(1,2,3), format=formatter,
                            orientation='horizontal')

            # Set the clim so that labels are centered on each block

            # label_objs(bmap,ax,locs)
            #plt.colorbar(pcm,cax=cax)
            ax.set_title("MDI value")
        bmap.drawstates(color='grey')
        bmap.drawcountries(color='grey')


    out_fname = "obj_ex.png"
    out_fpath = os.path.join(outroot,out_fname)
    utils.trycreate(out_fpath)
    fig.tight_layout()
    fig.savefig(out_fpath)
    print("Saved figure to",out_fpath)
    plt.close(fig)


if do_ign_storm:
    # Show verif
    # Plot 3km raw and objects by ignorance
    # Same for 1 km objects

    # Obs obj centroid location
    # -88.24355617959804
    # 33.48270273742653
    # 1133

    # latlon = (33.46,-88.28)
    # latlon = (41.6,-78.72)
    # latlon = (33.79,-84.24)
    # latlon = (36.6,-101.0)

    # latlon = (32.9,-88.6)


    latlon = (32.65,-89.2)

    # latpt, lonpt = latlon

    # latmax, latmin, lonmax, lonmin = (34.5,33.0,-87.5,-89.5)
    # pad = 0.8

    #lon_pad = 1.0
    #lat_pad = 0.8

    lon_pad = 0.6
    lat_pad = 0.55

    latmax = latlon[0] + lat_pad
    latmin = latlon[0] - lat_pad
    lonmax = latlon[1] + lon_pad
    lonmin = latlon[1] - lon_pad


    vrbl = "UH25"

    latlon = (0,0)

    # Western OK/TX cell: obs_1km: 1430, (36.384018,-101.334395)
    # Western OK/TX cell: obs_3km: 581,

    # d02_1km 1.91
    # d01_3km 2.02


    # Eastern OK/TX cell: obs_1km: 1431, (36.594944,-100.882999)
    # Eastern OK/TX cell: obs_1km: 582,

    # d02_1km 2.23
    # d01_3km 1.91
    mem = 'm02'

    mf = load_megaframe(fmts=None)

    valid_time = datetime.datetime(2016,3,31,21,30,0)
    valid = f"{valid_time.hour:02d}{valid_time.minute:02d}"

    init_time =  datetime.datetime(2016,3,31,20,0,0)
    init = f"{init_time.hour:02d}{init_time.minute:02d}"

    case_date = datetime.datetime(2016,3,31,0,0,0)
    case = f"{case_date.year:04d}{case_date.month:02d}{case_date.day:02d}"

    def do_plot_obj(bmap,ax,obj,contour_only=False,lw=(1.0,),anno_ign=False):
        locs = dict()
        obj_discrim = N.zeros_like(obj.OKidxarray)
        for o in obj.objects.itertuples():
            lab = "{:0.2f}".format(o.qlcsness)
            locs[lab] = (o.centroid_lat,o.centroid_lon)
            id = int(o.label)
            discrim = 1
            obj_discrim = N.where(obj.idxarray == id, discrim, obj_discrim)
            bmap.contour(x,y,obj_discrim,colors=['k',],levels=[0,1],linewidths=lw)

            # OBS megaframe idx:
            # TOP LEFT = 2518
            # BOTTOM LEFT = 2516
            # BOTTOM RIGHT = 2515
            if anno_ign:

                # obs_objs = mf[
                #             (mf['prod_code'] == )
                # ]
                obj_df = mf[
                        (mf['centroid_row'] == o.centroid_row) &
                        (mf['centroid_col'] == o.centroid_col) &
                        # (mf['fpath_save'] == data_fpath)
                        (mf['time'] == valid_time)
                        ].megaframe_idx
                mega_idx = obj_df.values[0]
                mini_idx = mf[mf['megaframe_idx'] == mega_idx
                                ].miniframe_idx.values[0]
                # pdb.set_trace()
                anno_str = f"{mega_idx}\n{mini_idx}"

                x_pt, y_pt = bmap(o.centroid_lon,o.centroid_lat)

                if (lonmin < o.centroid_lon < lonmax) & (
                        latmin < o.centroid_lat < latmax):
                    ax.annotate(anno_str,xy=(x_pt,y_pt),xycoords="data",
                        annotation_clip=False,zorder=1000,
                        fontsize=11,# fontstyle="italic",
                        fontweight='bold',color="blue",)
                    pdb.set_trace()

        if not contour_only:
            marr = N.ma.masked_where(obj_discrim < 1, obj_discrim)
            mraw = N.ma.masked_where(obj.raw_data < 1, obj_discrim)

            pcm = bmap.pcolormesh(data=marr,x=x,y=y,alpha=0.5,vmin=1,vmax=3,
                                    cmap=M.cm.get_cmap("magma",3),)
        return


    data_folder = os.path.join(objectroot,case)
    data_fname_1 = f"objects_REFL_comp_d02_1km_{case}_{init}_{valid}_{mem}.pickle"
    data_fname_3 = f"objects_REFL_comp_d01_3km_{case}_{init}_{valid}_{mem}.pickle"
    data_fname_o = f"objects_NEXRAD_nexrad_1km_{case}_{valid}.pickle"

    data_fpath_1 = os.path.join(data_folder,data_fname_1)
    data_fpath_3 = os.path.join(data_folder,data_fname_3)
    data_fpath_o = os.path.join(data_folder,data_fname_o)


    obj_3 = utils.load_pickle(data_fpath_3)
    obj_1 = utils.load_pickle(data_fpath_1)
    obj_o = utils.load_pickle(data_fpath_o)


    # Verif - UH
    uhcm = M.cm.Reds
    # uh_minmax = (0.001,0.012)
    #uh_minmax_3 = (0.2,1.5)
    #uh_minmax_1 = (0.5,4.2)
    uh_minmax_3 = (0.3,3.2)
    uh_minmax_1 = (0.8,9.9)

    # aws_minmax = (0.001,0.009)
    # aws_minmax = (0.007,0.010)
    aws_minmax = (0.002,0.0044)





    # Get fcst/obs data - REFL_comp
    fcst_data_3, fcst_lats_3, fcst_lons_3 = load_fcst_dll(
            "REFL_comp","d01_3km",
            valid_time,case_date,init_time,mem)

    fcst_data_1, fcst_lats_1, fcst_lons_1 = load_fcst_dll(
            "REFL_comp","d02_1km",
            valid_time,case_date,init_time,mem)

    obs_data, obs_lats, obs_lons = load_obs_dll(
        valid_time, case_date,
        fcst_vrbl = "REFL_comp",
        fcst_fmt = "d02_1km",)



    # Get ignorance, maybe objects

    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(10,7))
    fname = "ML_AS_ex_cb.png"
    fpath = os.path.join(outroot,fname)
    S = Scales('cref')

    # Verif
    ax = axes.flat[0]
    m = create_bmap(
                    #obs_lats.max(),obs_lons.max(),
                    #obs_lats.min(),obs_lons.min(),
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(obs_lons,obs_lats)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,obs_data,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')

    # EE3km
    ax = axes.flat[1]
    m = create_bmap(
                    # fcst_lats_3.max(),fcst_lons_3.max(),
                    # fcst_lats_3.min(),fcst_lons_3.min(),ax=ax,
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(fcst_lons_3,fcst_lats_3)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,fcst_data_3,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')

    # EE1km
    ax = axes.flat[2]
    m = create_bmap(
                    # fcst_lats_1.max(),fcst_lons_1.max(),
                    # fcst_lats_1.min(),fcst_lons_1.min(),ax=ax,
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(fcst_lons_1,fcst_lats_1)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,fcst_data_1,cmap=S.cm,vmin=S.clvs[0],
                        vmax=S.clvs[-1],alpha=0.5,
                        antialiased=True,)
    pcm.cmap.set_under('white')
    # plt.colorbar(pcm,ax=ax)



    # Get fcst/obs data - UH and AWS
    fcst_data_3, fcst_lats_3, fcst_lons_3 = load_fcst_dll(
            "UH25","d01_3km",
            valid_time,case_date,init_time,mem)

    fcst_data_1, fcst_lats_1, fcst_lons_1 = load_fcst_dll(
            "UH25","d02_1km",
            valid_time,case_date,init_time,mem)

    obs_data, obs_lats, obs_lons = load_obs_dll(
        valid_time,case_date,
        fcst_vrbl = "UH25",
        fcst_fmt = "d02_1km",)



    ax = axes.flat[3]
    m = create_bmap(
                    # obs_lats.max(),obs_lons.max(),
                    # obs_lats.min(),obs_lons.min(),ax=ax,
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(obs_lons,obs_lats)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,obs_data,cmap=uhcm,
                        vmin=aws_minmax[0],vmax=aws_minmax[1],
                        alpha=0.5,antialiased=True,)
    pcm.cmap.set_under('white')
    do_plot_obj(m,ax,obj_o,contour_only=True,anno_ign=True,lw=(1.0,))
    # plt.colorbar(pcm,ax=ax)


    # EE3km
    ax = axes.flat[4]
    m = create_bmap(
                    # fcst_lats_3.max(),fcst_lons_3.max(),
                    # fcst_lats_3.min(),fcst_lons_3.min(),ax=ax,
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(fcst_lons_3,fcst_lats_3)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,fcst_data_3,cmap=uhcm,
                        vmin=uh_minmax_3[0],vmax=uh_minmax_3[1],
                        alpha=0.5,antialiased=True,)
    pcm.cmap.set_under('white')
    do_plot_obj(m,ax,obj_3,contour_only=True,lw=(1.0,),anno_ign=True)


    # EE1km
    ax = axes.flat[5]
    m = create_bmap(
                    # fcst_lats_1.max(),fcst_lons_1.max(),
                    # fcst_lats_1.min(),fcst_lons_1.min(),ax=ax,
                    latmax,lonmax,latmin,lonmin,ax=ax,
                    proj='merc',latlon=latlon)
    x,y = m(fcst_lons_1,fcst_lats_1)
    m.drawstates(color='grey')
    m.drawcountries(color='grey')
    pcm = m.pcolormesh(x,y,fcst_data_1,cmap=uhcm,
                        vmin=uh_minmax_1[0],vmax=uh_minmax_1[1],
                        alpha=0.5,antialiased=True,)
    pcm.cmap.set_under('white')
    do_plot_obj(m,ax,obj_1,contour_only=True,lw=(1.0,),anno_ign=True)

    if False:

        ax = axes.flat[6]
        m = create_bmap(
                        #obs_lats.max(),obs_lons.max(),
                        #obs_lats.min(),obs_lons.min(),ax=ax,
                        latmax,lonmax,latmin,lonmin,ax=ax,
                        proj='merc',latlon=latlon)
        x,y = m(obs_lons,obs_lats)
        m.drawstates(color='grey')
        m.drawcountries(color='grey')
        do_plot_obj(m,ax,obj_o)

        ax = axes.flat[7]
        m = create_bmap(
                        #fcst_lats_3.max(),fcst_lons_3.max(),
                        #fcst_lats_3.min(),fcst_lons_3.min(),ax=ax,
                        latmax,lonmax,latmin,lonmin,ax=ax,
                        proj='merc',latlon=latlon)
        x,y = m(fcst_lons_3,fcst_lats_3)
        m.drawstates(color='grey')
        m.drawcountries(color='grey')
        do_plot_obj(m,ax,obj_3)

        ax = axes.flat[8]
        m = create_bmap(
                        # fcst_lats_1.max(),fcst_lons_1.max(),
                        # fcst_lats_1.min(),fcst_lons_1.min(),ax=ax,
                        latmax,lonmax,latmin,lonmin,ax=ax,
                        proj='merc',latlon=latlon)
        x,y = m(fcst_lons_1,fcst_lats_1)
        m.drawstates(color='grey')
        m.drawcountries(color='grey')
        do_plot_obj(m,ax,obj_1)

    # Print matches for each obs object

    # OBS megaframe idx:
    # TOP LEFT = 2518
    # BOTTOM LEFT = 2516
    # BOTTOM RIGHT = 2515

    print("2518 (top left):")
    MATCH = get_MATCH(member_names,modes=('cellular',))
    # MATCH[dom_code][member][casestr][initstr][mode]['2x2']

    # for mem in member_names:
    for mem in ('m02',):
        print(mem)


        prod_code = f"nexrad_3km_obs"
        obs_objs_IDs = mf[
                (mf['prod_code'] == prod_code) &
                (mf['conv_mode'] == 'cellular') &
                (mf['case_code'] == case) # &
                # (mf['time'] >= earliest_utc) &
                # (mf['time'] <= latest_utc)
                # (mf['leadtime_group'] == f"{hour}")
                ].megaframe_idx

        matches = MATCH['d01_3km'][mem][case][init]['cellular']['matches']
        matches_sort = MATCH['d01_3km'][mem][case][init]['cellular']['sorted']

        print(matches_sort)
        print(sorted(matches))

        print('=+'*20)

        prod_code = f"nexrad_1km_obs"
        obs_objs_IDs = mf[
                (mf['prod_code'] == prod_code) &
                (mf['conv_mode'] == 'cellular') &
                (mf['case_code'] == case) # &
                # (mf['time'] >= earliest_utc) &
                # (mf['time'] <= latest_utc)
                # (mf['leadtime_group'] == f"{hour}")
                ].megaframe_idx

        matches = MATCH['d02_1km'][mem][case][init]['cellular']['matches']
        matches_sort = MATCH['d02_1km'][mem][case][init]['cellular']['sorted']

        print(matches_sort)
        print(sorted(matches))




        # pdb.set_trace()

    # pdb.set_trace()

    fig.tight_layout()
    fig.savefig(fpath)

if do_uh_infogain:

    def compute_UH_DKL(i):
        """ .
        """
        dom, mode, oo, MATCH = i

        fcst_obj_mf = mf[
                    (mf['member']!='obs') &
                    (mf['conv_mode']==mode) &
                    (mf['resolution']==dom)
                    # mf['leadtime_group'==hr]
                    ]
        if dom == "EE3km":
            dc = "d01_3km"
        elif dom == "EE1km":
            dc = "d02_1km"
        else:
            assert 1==0


        for propno, prop in enumerate(uh_props):
            # Did it exceed the threshold?
            # Need to access this - don't loop over tuples but Series?
            obs_yesno = oo[prop]
            obs_ID = oo['megaframe_idx']
            casestr = oo['case_code']
            caseutc = datetime.datetime.strptime(casestr,"%Y%m%d")
            obs_utc = oo['time']

            # Get cases from CASES
            # For each init time, would it capture the object?
            iss = []
            for initutc in CASES[caseutc]:
                # compare datetimes

                # Allow for windowing of objects
                startutc = initutc + datetime.timedelta(seconds=(20*60))
                endutc = initutc + datetime.timedelta(
                                            seconds=((3*60*60)-(20*60)))
                if (obs_utc > startutc) and (obs_utc < endutc):
                #dt = obs_utc - initutc
                    iss.append(initutc)
            # Now load each ensemble member match

            # No match is a 0.01
            # All match is a 0.99

            # Get window of object timing and form list of
            # all eligible forecasts (so we avoid punishing
            # erroneous 0% forecasts)

            # Need to loop over all forecast inits and try to match
            DKL_list = []
            for initutc in iss:
                initstr = datetime.datetime.strftime(initutc,"%H%M")
                # Check to see if this object can even be matched in
                # the time of the object

                # Get hour period for this forecast object
                # Return array of [DKL,fcsthr]
                dt = obs_utc - initutc
                hr_dec = dt.seconds/3600

                if hr_dec < 1:
                    leadtime_group = 0
                elif hr_dec < 2:
                    leadtime_group = 1
                elif hr_dec < 3:
                    leadtime_group = 2
                else:
                    assert 1==0

                mem_exceeds = []
                for mem in member_names:
                    matches = MATCH[dc][mem][casestr][initstr][
                                mode]['sorted']
                    if len(matches) == 0:
                        mem_yesno = False
                    else:
                        mem_match_yn = obs_ID in N.array(matches)[:,0]
                        if mem_match_yn:
                            mem_yesno = False
                        else:
                            for match in matches:
                                if match[0] == obs_ID:
                                    idx = match[1]
                                    mem_yesno = fcst_obj_mf[
                                        fcst_obj_mf['megaframe_idx']==idx].prop
                    mem_exceeds.append(mem_yesno)

                print(mem_exceeds)
                fprob_raw = N.mean(mem_exceeds)
                fprob = constrain(fprob_raw,minmax=(0.01,0.99))

                DKL = -((obs_yesno * N.log2(obs_yesno/fprob))+
                            ((1-obs_yesno)*N.log2((1-obs_yesno)/(1-fprob))))
                # Compute
                # pdb.set_trace()
                # Get matches

                # Lookup in fcst mf

                # compute probs

                # compute DKL

                # Return DKL (can collate over leadtime,)
                # Do as list for all props
                # Then returns from pool can be converted to array?
                # This serves as a histogram for this collection
                # (per leadtime,resolution,mode)

                # Compute hour from diff in obs time and initutc
                # Maybe return the hour of this object as 0/1/2
                DKL_list.append([DKL,leadtime_group,propno])
            pass
            # This is end of looping over forecasts
        # Now combine for all props/fcsts
        DKL_arr = N.array(DKL_list)
        # pdb.set_trace()
        return DKL_arr

    def __gen_UH_loop(obs_obj_mf,dom,mode,MATCH):
        for initutc, casestr, initstr in get_initutcs_strs():
            # Get objects for this time
            for oo in obs_obj_mf.iterrows():
                yield initutc, casestr, initstr, dom, mode, oo[1], MATCH

    def gen_UH_loop(obs_obj_mf,dom,mode,MATCH):
        for oo in obs_obj_mf.iterrows():
            # pdb.set_trace()
            yield dom, mode, oo[1], MATCH

    uh_props = ["midrot_exceed_ID_0","midrot_exceed_ID_1",
        "midrot_exceed_ID_2","midrot_exceed_ID_3",
        "lowrot_exceed_ID_0","lowrot_exceed_ID_1",
        "lowrot_exceed_ID_2","lowrot_exceed_ID_3",]

    modes = ('cellular','linear')
    doms = ("EE3km","EE1km")
    # hours = ("first_hour","second_hour","third_hour")
    # for dom, mode, hr in itertools.product(doms,modes,hours):
    for dom, mode in itertools.product(doms,modes):
        MATCH, mf = get_MATCH(member_names,return_megaframe=True,modes=(mode,))
        # pdb.set_trace()
        obs_obj_mf = mf[
                    (mf['member']=='obs') &
                    # (mf['conv_mode']==mode) &
                    (mf['resolution']==dom)
                    # mf['leadtime_group']==hr,
                    ]
        # Paralellise over dom, init, validutc, leadtime
        # Loop over objects and match ensemble objects
        # After, collate histogram count per info gain/loss?
        itr = gen_UH_loop(obs_obj_mf,dom,mode,MATCH)

        # Can we parallelise over objects? The function can load
        # the forecast matches from the main mf?

        # Need new iterator over these cases but also dom/hour/mode

        if ncpus > 1:
            with multiprocessing.Pool(ncpus) as pool:
                returns = pool.map(compute_UH_DKL,itr)
        else:
            returns = []
            for i in itr:
                returns.append(compute_UH_DKL(i))

        # Collate results

        # Save to pickle (numpy?)

    # PLOT!!

    ##############################

    # Get all objects

    # Get all matches

    # For each object, for each threshold, compute the prob of exceedence
    # This is DKL (divergence), as we capture false alarms too (unlike
    # the REFL/object info gain)

    # Separate by hour


    # Plot as obj info gain
