import pdb
import logging
import os
import glob
import collections
import datetime
import pickle
import itertools
import multiprocessing
import argparse

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

import evac.utils as utils
from evac.datafiles.wrfout import WRFOut
from evac.plot.domains import Domains
from evac.stats.detscores import DetScores

### SWITCHES ###
do_domains = False
do_performance = True

### SETTINGS ###
mp_log = multiprocessing.log_to_stderr()
mp_log.setLevel(logging.INFO)

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
outroot = "/home/nothijngrad/Xmas_Shutdown/pyoutput"

##### OTHER STUFF #####
stars = "*"*10
ncpus = 6
dom_names = ("d01","d02")
domnos = (1,2)
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
# doms = (1,2)
fcst_vrbls = ("Wmax","UH02","UH25","RAINNC","REFL_comp")
obs_vrbls = ("AWS02","AWS25","DZ","ST4","NEXRAD")
fcstmins = N.arange(0,185,5)

#### FOR DEBUGGING ####
# CASES = {
        # datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,22,0,0),],
        # }

# member_names = ['m{:02d}'.format(n) for n in range(1,19)]
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

def fc2ob(fcv,fcfmt):
    """ Look up the corresponding variable/format name for obs, from vrbls.
    """
    # This is the observational variable
    VRBLS = {"REFL_comp":"NEXRAD",}
    obv = VRBLS[fcv]

    # This is the dx for the grid (e.g. 1km)
    *_, dx = fcfmt.split("_")

    # These are the field codes
    OO = {'NEXRAD':'nexrad'}
    obfmt = "_".join((OO[obv],dx))

    # pdb.set_trace()
    return obv,obfmt

def loop_perf(thresh):
    for vrbl in ("REFL_comp",):
        for caseutc, initutcs in CASES.items():
            for initutc in initutcs:
                for mem in member_names:
                    for validmin in fcstmins:
                        validutc = initutc+datetime.timedelta(seconds=60*int(validmin))
                        for fmt in ("d01_3km","d02_1km","d02_3km"):
                            yield vrbl, caseutc, initutc, mem, fmt, \
                                    validutc, thresh

### COLOUR SETTINGS ###
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
COLORS = {"d01":"#1A85FF",
            "d02":"#D41159",}


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
    # For every forecast valid time, load 36 members and 1 obs
    # Parallelise

    def compute_perf(i):
        fcst_vrbl, caseutc, initutc, mem, fcst_fmt, validutc,thresh = i

        fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                        validutc=validutc,caseutc=caseutc,initutc=initutc,
                        mem=mem)

        fcst_data = N.load(fcst_fpath)

        # These are the corresponding obs formats/data
        obs_vrbl, obs_fmt = fc2ob(fcst_vrbl,fcst_fmt)
        obs_fpath = get_extraction_fpaths(vrbl=obs_vrbl,fmt=obs_fmt,
                        validutc=validutc,caseutc=caseutc)
        obs_data = N.load(obs_fpath)

        DS = DetScores(fcst_arr=fcst_data,obs_arr=obs_data,thresh=thresh,
                        overunder='over')
        print("Computed contingency scores for",caseutc,initutc,mem,fcst_fmt,validutc)
        return DS

    for thresh in (15,30,40,50):
        itr = loop_perf(thresh)

        with multiprocessing.Pool(ncpus) as pool:
            results = pool.map(compute_perf,itr)
        # for i in list(itr):
        #     compute_perf(i)
        # 
