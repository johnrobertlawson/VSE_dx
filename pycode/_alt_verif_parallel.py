"""
d01 (3 km):
    * Cut/native (3 km)
    * Cut/interpolated to 5-km (neutral; for StageIV)
d02 (1 km):
    * Native (1 km)
    * Interpolated to 3-km (d01 domain - points must be synced)
    * Interpolated to 5-km (neutral; for StageIV)

"""

import pdb
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


##### SETTINGS #####

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

# Object results
compute_obj = False
plot_obj = False

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
                        datetime.datetime(2017,5,1,19,0,0), 
                        datetime.datetime(2017,5,1,20,0,0), 
                        datetime.datetime(2017,5,1,21,0,0), 
                        datetime.datetime(2017,5,1,22,0,0), 
                        # BROKEN
                        datetime.datetime(2017,5,1,23,0,0),
                        ]
CASES[datetime.datetime(2017,5,2,0,0,0)] = [
                        # Currently running:
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

#### OTHER INFO ####

vrbls = ("W","accum_precip","REFL_comp",)#"UH02","UH05")
fcstmins = N.arange(0,185,5)
grids = ['d01_cut_3km','d01_interp_5km',
        'd02_native_1km','d02_interp_3km','d02_interp_5km']

def do_interp(fmt,old_field,old_lats,old_lons,new_lats,new_lons):
    # First, get lat/lon of every run

    if fmt == "d01_cut_3km":
        # Cut big domain to small domain

if do_interp:
    # Put variables onto numpy grids
    for vrbl in vrbls:
        
