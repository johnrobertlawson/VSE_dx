import pdb
import logging
import os
import glob
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
import scipy
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

from evac.datafiles.wrfout import WRFOut
from evac.datafiles.obsgroup import ObsGroup
import evac.utils as utils
# import unsparsify
from unsparsify import readwdssii

### ARG PARSE ####
parser = argparse.ArgumentParser()

parser.add_argument('-N','--ncpus',dest='ncpus',default=20,type=int)
parser.add_argument('-D','--debug',dest='check_in_serial',default=False,type=bool)

PA = parser.parse_args()
ncpus = PA.ncpus
check_in_serial = PA.check_in_serial

### SETTINGS ###

# Folder key for forecast data (wrfout)
key_wrf = 'ForReal_nco'
# ensroot = '/scratch/john.lawson/WRF/VSE_reso/{}'.format(key_wrf)
ensroot = '/oldscratch/john.lawson/WRF/VSE_reso/{}'.format(key_wrf)

# Folder key for post-processed fields (objects, lat/lon, etc)
key_pp = 'UHFool'
pp_root = '/work/john.lawson/VSE_reso/pp/{}'.format(key_pp)
tempdir_root = '/work/john.lawson/VSE_reso/pp_temp/{}'.format(key_pp)

# Folder key for scores (FSS, etc)
#key_scores = 'AprilFool'
key_scores = "UHFool"
scoreroot = '/work/john.lawson/VSE_reso/scores/{}'.format(key_scores)

# Folder key for plots
# key_plot = 'AprilFool'
key_plot = 'UHFool'
plotroot = '/home/john.lawson/VSE_reso/pyoutput/{}'.format(key_plot)

st4dir = '/work/john.lawson/STAGEIV_data'
mrmsdir = '/work/john.lawson/MRMS_data/rot-dz'
radardir = '/work/john.lawson/NEXRAD_data'


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

##### OTHER STUFF #####

#### FOR DEBUGGING ####
#CASES = { datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,22,0,0),], }
########################

stars = "*"*10
dom_names = ("d01","d02")
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
# member_names = ['m{:02d}'.format(n) for n in range(1,19)]
# member_names = ['m{:02d}'.format(n) for n in range(1,2)]
# doms = (1,2)

# THESE are all possible variables in the script
# Note that UP_HELI_MAX is 2-5 km time-window-max
FCST_VRBLS = ("Wmax","UH02","UH25","RAINNC","REFL_comp","UP_HELI_MAX")
other_list = ["u_shear01","v_shear01","u_shear06","v_shear06","SRH03","CAPE_100mb"]
OBS_VRBLS = ("AWS02","AWS25","DZ","ST4","NEXRAD")

# "NEXRAD"
# These are the requests variables
fcst_vrbls = ("UH02","UH25")
# fcst_vrbls = ("SRH03","u_shear01","v_shear01")
# fcst_vrbls = ("REFL_comp","UH25","UH02","Wmax","RAINNC")
# fcst_vrbls = ("Wmax","RAINNC")
obs_vrbls = ("NEXRAD",)
# obs_vrbls = ("AWS25","AWS02","ST4","DZ","NEXRAD")

# Don't allow computation without both fcst and obs data requested
# The WRF files are needed for lat/lon for interp.
# Maybe not needed once lats.npy and lons.npy are created
assert fcst_vrbls and obs_vrbls

debug_mode = False
# fcstmins = N.arange(0,185,5)
# maxsec = 60*60*3

#### FOR DEBUGGING ####
# CASES = { datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,22,0,0),], }

# fcstmins = N.arange(0,20,5)
# maxsec = 60*60*0.25

#########################
def _save(arr,file):
    if isinstance(arr,N.ma.core.MaskedArray):
        data = arr.data
    elif isinstance(arr,N.ndarray):
        data = arr
    else:
        raise Exception
    # pdb.set_trace()
    N.save(arr=data,file=file)
    return


### FUNCTIONS ###
def get_all_initutcs():
    """ Find all unique initialisation times.
    """
    SET = set()
    for _caseutc,initutcs in CASES.items():
        for i in initutcs:
            SET.add(i)
    return list(SET)

initutcs = get_all_initutcs()

class CaseGrid:
    def __init__(self,Wnc):
        """ Create new grid for 5 km data.

        wrfout_fpath should be the d02 domain.
        """

        if isinstance(Wnc,str):
            Wnc = Dataset(Wnc)
        Wlats = Wnc.variables['XLAT'][0,:,:]
        Wlons = Wnc.variables['XLONG'][0,:,:]

        urcrnrlat = Wlats[-1,-1]
        urcrnrlon = Wlons[-1,-1]
        llcrnrlat = Wlats[0,0]
        llcrnrlon = Wlons[0,0]
        self.bmap = self.create_neutral_bmap(
                    urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon)

        nx, ny = self.dx_to_nxny(5,
                    urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon)
        lons, lats, xx, yy = self.bmap.makegrid(nx,ny,returnxy=True)
        self.lats = lats
        self.lons = lons
        self.xx = xx
        self.yy = yy

        self.act_dx = N.diff(self.xx[0,:]).mean()
        self.act_dy = N.diff(self.yy[:,0]).mean()
        print("Average dx = {:.1f}km and dy = {:.1f}km.".format(self.act_dx/1000,self.act_dy/1000))
        # pdb.set_trace()

    def create_neutral_bmap(self,
                    urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon):
        bmap = Basemap(
                    # width=12000000,height=9000000,
                    urcrnrlat=urcrnrlat,urcrnrlon=urcrnrlon,
                    llcrnrlat=llcrnrlat,llcrnrlon=llcrnrlon,
                    rsphere=(6378137.00,6356752.3142),
                    resolution='l',projection='lcc',
                    lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
        return bmap

    def dx_to_nxny(self,dx,urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon):
        xW, yS = self.bmap(llcrnrlon,llcrnrlat)
        xE, yN = self.bmap(urcrnrlon,urcrnrlat)

        # km difference in each direction
        xdiff_km = N.abs(xE-xW)/1000
        ydiff_km = N.abs(yS-yN)/1000

        # nx needed to give dx requested
        nx = int(N.floor(xdiff_km/dx))
        # assume dx=dy
        ny = int(N.floor(ydiff_km/dx))

        return nx, ny

    def return_latlons(self):
        return self.lats, self.lons

def submit_interp(i):
    _ = interpolate(*i)
    return

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
    if (vrbl in FCST_VRBLS) or (vrbl in other_list): # ("Wmax","UH02","UH25","RAINNC"):
        # TODO: are we not just doing 5-min or 1-hr accum_precip?
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        initHHMM = "{:02d}{:02d}".format(initutc.hour, initutc.minute)
        validHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "{}_{}_{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,initHHMM,
                                        validHHMM,mem)
    elif vrbl in OBS_VRBLS: # ("AWS02","AWS25","DZ","ST4"):
        caseYYYYMMDD = "{:04d}{:02d}{:02d}".format(caseutc.year,caseutc.month,
                                                caseutc.day)
        # utcYYYYMMDD = validutc
        utcHHMM = "{:02d}{:02d}".format(validutc.hour,validutc.minute)
        fname = "{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,utcHHMM)
    return os.path.join(pp_root,caseYYYYMMDD,vrbl,fname)

def get_tempdata_fpath(caseutc,fmt,vrbl,validutc,initutc,mem):
    if caseutc is None:
        casestr = "cNone"
    else:
        casestr = utils.string_from_time('dir',caseutc,strlen='day')

    if fmt is None:
        fmt = "fNone"

    if vrbl is None:
        vrbl = "vNone"

    if validutc is None:
        validstr = "uNone"
    else:
        validstr = utils.string_from_time('dir',validutc,strlen='minute')

    if initutc is None:
        initstr = "iNone"
    else:
        initstr = utils.string_from_time('dir',initutc,strlen='hour')

    if mem is None:
        mem = "mNone"

    fname = "temp_{}_{}_{}_{}_{}_{}.npy".format(casestr,initstr,validstr,vrbl,fmt,mem)
    fpath = os.path.join(tempdir_root,fname)
    utils.trycreate(fpath,debug=False)
    return fpath

def get_data(caseutc,fmt,vrbl=None,validutc=None,initutc=None,mem=None,
                        latlon_only=False,load_latlons=False):
    """ Whatever
    """
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    if mem:
        assert initutc
        assert (vrbl in FCST_VRBLS) or (vrbl in other_list)
        initstr = utils.string_from_time('dir',initutc,strlen='hour')
        memdir = os.path.join(ensroot,initstr,mem)

    latf, lonf = get_llf_fpath(casestr,fmt)
    if latlon_only:
        if load_latlons:
            return N.load(latf), N.load(lonf)
        else:
            return latf, lonf

    else:
        data_fpath = get_tempdata_fpath(caseutc,fmt,vrbl,validutc,initutc,mem)

    if fmt == "d01_raw":
        # Load lat/lon grid
        latf, lonf = get_llf_fpath(casestr,"d01_raw")

        # Load 3-km wrfout domain as is, on 2D
        fname = get_wrfout_fname(initutc,dom=1)
        fpath = os.path.join(memdir,fname)
        W = WRFOut(fpath)

        # if vrbl in ("Wmax","UH02","UH25","RAINNC"):
        data = W.get(utc=validutc,vrbl=vrbl)[0,0,:,:]

    elif fmt == "d02_raw":
        # Load lat/lon grid
        latf, lonf = get_llf_fpath(casestr,"d02_raw")

        # Load 1-km wrfout domain as is, on 2D
        fname = get_wrfout_fname(initutc,dom=2)
        fpath = os.path.join(memdir,fname)
        W = WRFOut(fpath)

        # if vrbl in ("Wmax","UH02","UH25","RAINNC"):
        data = W.get(utc=validutc,vrbl=vrbl)[0,0,:,:]

    elif fmt in ("mrms_aws_raw", "mrms_dz_raw"):
        latf, lonf = get_llf_fpath(casestr,fmt)

        # Load MRMS data, 2D
        FILES, utcs, key,fdir = return_all_mrms(caseutc,vrbl)
        t = closest(utcs,validutc,return_val=True)
        fpath = os.path.join(fdir,FILES[t])
        mrms_nc = Dataset(fpath)

        # d01_nc = get_random_d01(caseutc)
        data = _unsparsify(vrbl=key,mrms_nc=mrms_nc,)#d01_nc=d01_nc)
        with N.errstate(invalid='ignore'):
            data[data<-100] = N.nan
            data[data>100] = N.nan
        # pdb.set_trace()

    elif fmt == "stageiv_raw":
        latf, lonf = get_llf_fpath(casestr,"stageiv_raw")
        # Load stage iv data
        data = ST4.get(utc=validutc)[0,0,:,:]
        # pdb.set_trace()

    elif fmt == 'nexrad_raw':
        latf, lonf = get_llf_fpath(casestr,"nexrad_raw")
        # Load stage iv data
        data = RADARS.get(utc=validutc)

    else:
        print("fmt not valid.")
        assert True is False

    # This is now done in _save function
    # if N.ma.is_masked(data):
        # data = data.data

    # pdb.set_trace()
    _save(arr=data,file=data_fpath)

    if load_latlons:
        return data_fpath, N.load(latf), N.load(lonf)
    else:
        return data_fpath, latf, lonf

def get_random_d01(caseutc):
    initutc = CASES[caseutc][0]
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    initstr = utils.string_from_time('dir',initutc,strlen='hour')
    print("Loading random d01 wrfout for", casestr)

    m01dir = os.path.join(ensroot,initstr,"m01")

    d01_fname = get_wrfout_fname(initutc,1)
    d01_fpath = os.path.join(m01dir,d01_fname)
    d01_nc = Dataset(d01_fpath)

    return d01_nc

def _load_data(nc_fpath,vrbl,npyfpaths):
    W = WRFOut(nc_fpath)
    data = W.get(vrbl=vrbl)[:,0,:,:]

    assert data.shape[0] == len(npyfpaths)

    for tidx,npyfpath in enumerate(npyfpaths):
        utils.trycreate(npyfpath,debug=False)
        _save(arr=data[tidx,:,:],file=npyfpath)
    return

def create_bmap():
    bmap = Basemap(width=12000000,height=9000000,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection='lcc',
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
    return bmap

def closest(arr,val,return_val=False):
    if isinstance(val,datetime.datetime):
        diffs = N.array([abs(a-val) for a in arr])
        idx = N.argmin(diffs)
    else:
        idx = N.argmin(N.abs(arr-val))
    if not return_val:
        return idx
    return arr[idx]

#def cut(data,latsA,lonsA,latsB,lonsB):
#    return utils.return_subdomain(data,latsA,lonsA,cut_to_lats=d02_lats,
#                        cut_to_lons=d02_lons)

def interpolate(dataA,latsA,lonsA,latsB,lonsB,cut_only=False,
                # cut_first=True,
                # remove_negative=False,
                save_to_fpath=None):
    """ Interpolate data on gridA to gridB.

    If cut_first, if gridA is a lot bigger (e.g., StageIV data), cut it to
    a smaller size to improve interpolation efficiency.
    """
    cut_first = True
    if isinstance(dataA,str):
        dataA = N.load(dataA)
    if isinstance(latsA,str):
        latsA = N.load(latsA)
        lonsA = N.load(lonsA)
    if isinstance(latsB,str):
        latsB = N.load(latsB)
        lonsB = N.load(lonsB)

    assert latsA.ndim == 2
    assert latsB.ndim == 2
    # pdb.set_trace()

    if cut_only:
        assert dataA is not None
        dataB, _latsB, _lonsB = utils.return_subdomain(dataA,latsA,lonsA,cut_to_lats=latsB,
                                cut_to_lons=lonsB)
    else:
        if cut_first:
            _dataA = N.copy(dataA)
            _latsA = N.copy(latsA)
            _lonsA = N.copy(lonsA)

            Nlim = N.max(latsB)+0.5
            Elim = N.max(lonsB)+0.5
            Slim = N.min(latsB)-0.5
            Wlim = N.min(lonsB)-0.5

            Nidx = closest(latsA[:,0],Nlim)
            Eidx = closest(lonsA[0,:],Elim)
            Sidx = closest(latsA[:,0],Slim)
            Widx = closest(lonsA[0,:],Wlim)

            old_method = False
            if old_method:
                s1 = slice(Sidx,Nidx+1)
                s2 = slice(Widx,Eidx+1)

                dataA = dataA[s1,s2]
                latsA = latsA[s1,s2]
                lonsA = lonsA[s1,s2]

            else:
                dataA = dataA[Sidx:Nidx+1,Widx:Eidx+1]
                latsA = latsA[Sidx:Nidx+1,Widx:Eidx+1]
                lonsA = lonsA[Sidx:Nidx+1,Widx:Eidx+1]


            # if "mrms" in save_to_fpath:
                # pdb.set_trace()

        bmap = create_bmap()
        xxA, yyA = bmap(lonsA,latsA)
        xxB, yyB = bmap(lonsB,latsB)
        xdB = len(xxB[:,0])
        ydB = len(yyB[0,:])
        # if "nexrad" in save_to_fpath:
        # if "mrms" in save_to_fpath:
        dataA = N.copy(dataA)
        xxA = N.copy(xxA)
        yyA = N.copy(yyA)
        xxB = N.copy(xxB)
        yyB = N.copy(yyB)
        dataB = scipy.interpolate.griddata(points=(xxA.flat,yyA.flat),
                                            values=dataA.flat,
                                            xi=(xxB.flat,yyB.flat),
                                            method='linear').reshape(xdB,ydB)
    # if "mrms" in save_to_fpath:
        # pdb.set_trace()
    if save_to_fpath is not None:
        utils.trycreate(save_to_fpath,debug=False)
        _save(arr=dataB,file=save_to_fpath)
        print("Saved to",save_to_fpath)
        return
    return dataB

def generate_all_runs():
    for initutc,dom,mem in itertools.product(initutcs,dom_names,member_names):
        yield initutc, dom, mem

def generate_all_folders():
    for initutc,mem in itertools.product(initutcs,member_names):
        yield initutc, mem

def generate_fcst_loop():
    for vrbl in fcst_vrbls: # ("Wmax","UH02","UH25","RAINNC"):
        for caseutc, initutcs in CASES.items():
            for initutc in initutcs:
                for mem in member_names:
                    yield caseutc, initutc, mem, vrbl

def generate_itr_from_commands(commands):
    for c in commands:
        yield c

def generate_valid_utcs(initutc):
    if debug_mode:
        utc1 = initutc + datetime.timedelta(seconds=10*60)
    else:
        utc1 = initutc + datetime.timedelta(seconds=3600*3)
    # utc1 = initutcs[-1] + datetime.timedelta(seconds=maxsec)
    utc = initutc
    # print("Generate until"utc1)
    while utc <= utc1:
        yield utc
        # Add five minutes
        utc = utc + datetime.timedelta(seconds=60*5)


def generate_obs_loop():
    for vrbl in obs_vrbls: # ("AWS02","AWS25","DZ","ST4"):
        dt = 60 if (vrbl == "ST4") else 5
        for caseutc, initutcs in CASES.items():
            # Earliest time:
            utc0 = initutcs[0]
            # Latest time is 3 hours after the latest initialisation
            secs = 3600*3 if not debug_mode else 60*10
            utc1 = initutcs[-1] + datetime.timedelta(seconds=secs)
            # utc1 = initutcs[-1] + datetime.timedelta(seconds=maxsec)

            utc = utc0
            while utc <= utc1:
                yield vrbl, utc, caseutc
                # Add five minutes
                utc = utc + datetime.timedelta(seconds=60*dt)


def __get_mrms_extraction_fpath(initstr,validutc,vrbl,fmt):
    HHMM = "{:02d}{:02d}".format(validutc.hour, validutc.minute)
    f = "{}_{}.npy".format(vrbl,HHMM)
    fpath = os.path.join(reproj_obs_root,initstr,fmt,f)
    return fpath

def __get_fcst_extraction_fpaths(initstr,mem,vrbl,fmt,nt=37):
    assert isinstance(dom,str)
    assert isinstance(member,str)
    fpaths = []
    for t in range(nt):
        f = "{}_{}_t{:02d}.npy".format(vrbl,mem,t)
        fpath = os.path.join(reproj_fcst_root,initstr,fmt,f)
        fpaths.append(fpath)
    return fpaths

def get_wrfout_fname(t,dom):
    fname = 'wrfout_d{:02d}_{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(dom,
                t.year,t.month,t.day,t.hour,t.minute,t.second)
    return fname

def get_llf_fpath(casestr,key):
    # LATS
    lats_fname = "lats.npy"
    lats_fpath = os.path.join(pp_root,key,casestr,lats_fname)

    lons_fname = "lons.npy"
    lons_fpath = os.path.join(pp_root,key,casestr,lons_fname)

    return lats_fpath,lons_fpath

def _unsparsify(mrms_nc,vrbl,d01_nc=False):
    # return mrms_low_x_y
    # return unsparsify.do_unsparsify(mrms_nc,vrbl,d01_nc)
    x1,ulat,y1,ulon, varname,ref1 = readwdssii(mrms_nc)
    output = N.flipud(ref1)
    # pdb.set_trace()
    return output


def get_mrms_rotdz_grid(caseutc=None,vrbl=None,nc=None):
    """
    AzShear and DZ grids start at same lat/lon but are of different
    resolutions.

    Listed Latitude/Longitude attributes are top left
    """
    if nc is None:
        nc = open_random_rotdz(caseutc,vrbl)

    # if nc.DataType.startswith("Sparse"):

    ullat = nc.Latitude
    ullon = nc.Longitude
    nlat = nc.dimensions['Lat'].size
    nlon = nc.dimensions['Lon'].size
    dlat = nc.LatGridSpacing
    dlon = nc.LonGridSpacing

    ss_lat = slice(0,nlat*dlat,dlat)
    ss_lon = slice(0,nlon*dlon,dlon)
    _x, _y = N.mgrid[ss_lat,ss_lon]
    lats = -1.0 * (_x - ullat)
    lons = _y + ullon

    # pdb.set_trace()

    return N.flipud(lats),lons

def return_all_mrms(caseutc,vrbl):
    """ Return sorted dictionary of times and files.
    """
    fdir, key, dlat = lookup_mrms_metadata(caseutc,vrbl)
    casestr = utils.string_from_time('dir',caseutc,strlen='day')

    FILES = {}
    # Open all netcdf files in this folder, and create list of times
    nclist = glob.glob(os.path.join(fdir,"*.netcdf"))
    for f in nclist:
        utcstr, _ext = os.path.basename(f).split('.')
        assert _ext == "netcdf"
        fmt = "%Y%m%d-%H%M%S"
        utc = datetime.datetime.strptime(utcstr,fmt)

        FILES[utc] = f
    utcs = sorted(FILES.keys())

    # pdb.set_trace()

    return FILES, utcs, key, fdir


# def get_mrms_rotdz_fpath(caseutc,vrbl):

def lookup_mrms_metadata(caseutc,vrbl):
    """ Return the folder, variable key, grid-spacing of a MRMS dataset.

    fname convention is
    YYYYMMDD-HHMMSS.netcdf

    (Az Shear is per second)

    2016, LL Az Shear:
    /work/john.lawson/MRMS_data/rot-dz/20160331/processed/nofake/merged/MergedLLShear/00.00/*
    LatGridSpacing = 0.01
    MergedLLShear, pixel_x, pixel_y, pixel_count

    2016, ML Az Shear:
    /work/john.lawson/MRMS_data/rot-dz/20160331/processed/nofake/merged/MergedMLShear/00.00/*
    LatGridSpacing = 0.01
    MergedMLShear, pixel_x, pixel_y, pixel_count

    2016, DZ:
    /work/john.lawson/MRMS_data/rot-dz/20160331/processed/nofake/merged/MergedReflectivityQCComposite/00.00/*
    LatGridSpacing: 0.01
    MergedReflectivityQCComposite, ...

    2017, LL Az Shear:
    /work/john.lawson/MRMS_data/rot-dz/20170501/MergedAzShear_0-2kmAGL/00.00/*
    LatGridSpacing = 0.005
    MergedAzShear_0-2kmAGL, ...

    2017, ML Az Shear:
    /work/john.lawson/MRMS_data/rot-dz/20170501/MergedAzShear_2-5kmAGL/00.00/*
    LatGridSpacing: 0.005
    MergedAzShear_2-5kmAGL, ...

    2017 DZ:
    /work/john.lawson/MRMS_data/rot-dz/20170501/MergedReflectivityQCComposite/00.50/*
    LatGridSpacing: 0.01
    MergedReflectivityQCComposite, ...


    """
    casestr = utils.string_from_time('dir',caseutc,strlen='day')

    if caseutc.year == 2016:
        dlat = 0.01
        if vrbl == "DZ":
            key = "MergedReflectivityQCComposite"
        elif vrbl == "AWS02":
            key = 'MergedLLShear'
        elif vrbl == "AWS25":
            key = 'MergedMLShear'
        else:
            raise Exception

        fdir = os.path.join(mrmsdir,casestr,"processed","nofake","merged",
                        key,"00.00")
    elif caseutc.year == 2017:
        if vrbl == "DZ":
            dlat = 0.01
            key = 'MergedReflectivityQCComposite'
            subdir = "00.50"
        elif vrbl == "AWS02":
            dlat = 0.005
            key = 'MergedAzShear_0-2kmAGL'
            subdir = "00.00"
        elif vrbl == "AWS25":
            dlat = 0.005
            key = 'MergedAzShear_2-5kmAGL'
            subdir = "00.00"
        else:
            raise Exception

        fdir = os.path.join(mrmsdir,casestr,key,subdir)
    else:
        raise Exception

    return fdir, key, dlat


def open_random_rotdz(caseutc,vrbl):
    FILES, utcs, _key, fdir = return_all_mrms(caseutc,vrbl)
    nt = len(utcs)
    # t = utcs[0]
    t = utcs[int(nt/2)]
    f = FILES[t]
    # pdb.set_trace()
    # return f
    return Dataset(os.path.join(fdir,f))


def gather_commands_one(i):
    commands = []
    caseutc, initutc, mem, vrbl = i

    initstr = utils.string_from_time('dir',initutc,strlen='hour')
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    memdir = os.path.join(ensroot,initstr,mem)
    # print("Appending interpolation commands for",mem,"forecast grids of",vrbl,"on", initstr)

    for validutc in generate_valid_utcs(initutc):
        #print(stars,validutc,stars)
        ### d01_3km (d01 cut to d02)
        save_to_fpath = get_extraction_fpaths(vrbl,"d01_3km",validutc,
                            caseutc=caseutc,initutc=initutc,mem=mem)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="d01_raw",validutc=validutc,
                                caseutc=caseutc,initutc=initutc,mem=mem)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,True,save_to_fpath))

        ### d02_1km - should be a simple one...
        ### todo: compare interp to raw data! They should be very similar
        save_to_fpath = get_extraction_fpaths(vrbl,"d02_1km",validutc,
                            caseutc=caseutc,initutc=initutc,mem=mem)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="d02_raw",validutc=validutc,
                                caseutc=caseutc,initutc=initutc,mem=mem)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        ### d02_3km (d02 interpolated to 3 km)
        save_to_fpath = get_extraction_fpaths(vrbl,"d02_3km",validutc,
                            caseutc=caseutc,initutc=initutc,mem=mem)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="d02_raw",validutc=validutc,
                                caseutc=caseutc,initutc=initutc,mem=mem)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))
    # pdb.set_trace()
    return commands



# for vrbl, validutc, caseutc in generate_obs_loop():
def gather_commands_two(i):
    commands = []
    vrbl, validutc, caseutc = i
    # print("Appending interpolation commands for observational grids of",vrbl,"at", validutc)

    if vrbl in ("AWS02","AWS25"):
        ### mrms_aws_1km (AWS data interpolated to d02_1km)
        save_to_fpath = get_extraction_fpaths(vrbl,"mrms_aws_1km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        ### mrms_aws_3km (AWS data interpolated to d01_3km)
        save_to_fpath = get_extraction_fpaths(vrbl,"mrms_aws_3km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

    elif vrbl == "DZ":
        ### mrms_dz_1km (DZ data interpolated to d02_1km)
        save_to_fpath = get_extraction_fpaths(vrbl,"mrms_dz_1km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_dz_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        ### mrms_dz_3km (DZ data interpolated to d01_3km)
        save_to_fpath = get_extraction_fpaths(vrbl,"mrms_dz_3km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_dz_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

    elif vrbl == "ST4":
        save_to_fpath = get_extraction_fpaths(vrbl,"stageiv_1km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="stageiv_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        save_to_fpath = get_extraction_fpaths(vrbl,"stageiv_3km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="stageiv_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

    # pdb.set_trace()
    return commands

######### PROCEDURE #########
""" Numpy array extraction for Lawson et al, 2019.

The following are native grids, computed for each case:
    * d01_raw (3 km)
    * d02_raw (1 km)
    * stageiv_raw (4.7 km)
    * mrms_aws_raw (2016: 1 km; 2017: 0.5 km)
    * mrms_dz_raw (1 km)
    * neutral (5 km)
    * NEXRAD (? km)

The following are extra grids computed from above, requiring cut data:
    * d01_3km (d01_raw cut to d02_raw bounds)

The following are cut or interpolated grids, from above, requiring interpolation:
    * d01_5km (d01 data interpolated to neutral_5km)
    * d02_1km (d02 data on native, d02_raw)
    * d02_3km (d02 data interpolated to d01_3km)
    * d02_5km (d02 data interpolated to neutral)
    * mrms_aws_1km (AWS data interpolated to d02_1km)
    * mrms_aws_3km (AWS data interpolated to d01_3km)
    * <><><>mrms_aws_5km (AWS data interpolated to neutral)
    * mrms_dz_1km (cref data interpolated to d02_1km)
    * mrms_dz_3km (cref data interpolated to d01_3km)
    * <><><>mrms_dz_5km (cref data interpolated to neutral)
    * stageiv_5km (Stage IV data interpolated to neutral)
    * nexrad_1km (NEXRAD data to d02_raw)
    * nexrad_3km (NEXRAD data to d02_3km)
    * <><><>nexrad_5km (NEXRAD data to neutral)

This script will compute all lat/lon grids first (see above).

Then, interpolation of fields is performed for:

    1. FORECAST FIELDS (d01, d02):
    * Wmax (column max of W; not verified)
    * UH02 (TODO: write)
    * UH25 (TODO: write)
    * cref (column max of REFL_10CM)
    * RAINNC (check! rain rate - needs to be done hourly)

    2. OBSERVED FIELDS:
    * AWS02 (MRMS LL-AWS)
    * AWS25 (MRMS ML-AWS)
    * DZ (MRMS)
    * Stage IV

When the script is finished, the sliced/interpolated data is then
available to move to another location with all lat/lon requirements
present. TODO: make sure everything's in the same folder.

"""

### Get Stage IV catalogue
if "ST4" in obs_vrbls:
    ST4 = ObsGroup(st4dir,'stageiv')
if "NEXRAD" in obs_vrbls:
    RADARS = ObsGroup(radardir,'radar')

for caseutc, initutcs in CASES.items():
    # We only need to look at the first time, as the grids are the same
    initutc = initutcs[0]
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    initstr = utils.string_from_time('dir',initutc,strlen='hour')
    print("Calculating lat/lon grids for", casestr)

    ### First, let's make lat/lon arrays for each grid for this case
    ### Open if they exist, else create/save

    # We only need to look at the first member, as the grids are the same
    m01dir = os.path.join(ensroot,initstr,"m01")

    # d02
    d02_latf, d02_lonf = get_llf_fpath(casestr,"d02_raw")

    # TODO: we just need to loop to check the lat/lon
    # files have been created, or create them. We can
    # delete the lat/lon array files, and not load them.
    if os.path.exists(d02_latf):
        # d02_lats = N.load(d02_latf)
        # d02_lons = N.load(d02_lonf)
        pass
    else:
        d02_fname = get_wrfout_fname(initutc,2)
        d02_fpath = os.path.join(m01dir,d02_fname)
        d02_nc = Dataset(d02_fpath)
        d02_lats = d02_nc.variables['XLAT'][0,:,:]
        d02_lons = d02_nc.variables['XLONG'][0,:,:]

        utils.trycreate(d02_latf,debug=False)
        _save(arr=d02_lats,file=d02_latf)
        _save(arr=d02_lons,file=d02_lonf)

    # d01
    d01_latf, d01_lonf = get_llf_fpath(casestr,"d01_raw")

    if os.path.exists(d01_latf):
        # d01_lats = N.load(d01_latf)
        # d01_lons = N.load(d01_lonf)
        pass
    else:
        d01_fname = get_wrfout_fname(initutc,1)
        d01_fpath = os.path.join(m01dir,d01_fname)
        d01_nc = Dataset(d01_fpath)
        d01_lats = d01_nc.variables['XLAT'][0,:,:]
        d01_lons = d01_nc.variables['XLONG'][0,:,:]

        utils.trycreate(d01_latf,debug=False)
        _save(arr=d01_lats,file=d01_latf)
        _save(arr=d01_lons,file=d01_lonf)

    # STAGE IV
    if "ST4" in obs_vrbls:
        st4_latf, st4_lonf = get_llf_fpath(casestr,"stageiv_raw")
        if os.path.exists(st4_latf):
            # st4_lats = N.load(st4_latf)
            # st4_lons = N.load(st4_lonf)
            pass
        else:
            st4_lats = ST4.lats
            st4_lons = ST4.lons
            assert st4_lats.ndim == 2

            utils.trycreate(st4_latf,debug=False)
            _save(arr=st4_lats,file=st4_latf)
            _save(arr=st4_lons,file=st4_lonf)

    # MRMS LLAWS/MLAWS (rot)
    mrms_aws_latf, mrms_aws_lonf = get_llf_fpath(casestr,"mrms_aws_raw")
    if os.path.exists(mrms_aws_latf):
        # mrms_aws_lats = N.load(mrms_aws_latf)
        # mrms_aws_lons = N.load(mrms_aws_lonf)
        pass
    else:
        # mrms_fpath = get_mrms_rotdz_fpath(caseutc)
        # mrms_rotdz_nc = Dataset(mrms_fpath)
        mrms_aws_lats,mrms_aws_lons = get_mrms_rotdz_grid(caseutc,"AWS02")
        utils.trycreate(mrms_aws_latf,debug=False)
        _save(arr=mrms_aws_lats,file=mrms_aws_latf)
        _save(arr=mrms_aws_lons,file=mrms_aws_lonf)

    # MRMS DZ (cref)
    mrms_dz_latf, mrms_dz_lonf = get_llf_fpath(casestr,"mrms_dz_raw")
    if os.path.exists(mrms_dz_latf):
        # mrms_dz_lats = N.load(mrms_dz_latf)
        # mrms_dz_lons = N.load(mrms_dz_lonf)
        pass
    else:
        # mrms_fpath = get_mrms_rotdz_fpath(caseutc)
        # mrms_rotdz_nc = Dataset(mrms_fpath)
        mrms_dz_lats,mrms_dz_lons = get_mrms_rotdz_grid(caseutc,"DZ")
        utils.trycreate(mrms_dz_latf,debug=False)
        _save(arr=mrms_dz_lats,file=mrms_dz_latf)
        _save(arr=mrms_dz_lons,file=mrms_dz_lonf)

    # NEXRAD raw
    if "NEXRAD" in obs_vrbls:
        nexrad_latf, nexrad_lonf = get_llf_fpath(casestr,"nexrad_raw")
        if os.path.exists(nexrad_latf):
            pass
        else:
            nexrad_lats = RADARS.lats.astype("float32")
            nexrad_lons = RADARS.lons.astype("float32")
            assert nexrad_lats.ndim == 2

            utils.trycreate(nexrad_latf,debug=False)
            _save(arr=nexrad_lats,file=nexrad_latf)
            _save(arr=nexrad_lons,file=nexrad_lonf)

    # Neutral grid
    neutral_latf, neutral_lonf = get_llf_fpath(casestr,"neutral")
    if os.path.exists(neutral_latf):
        # neutral_lats = N.load(neutral_latf)
        # neutral_lons = N.load(neutral_lonf)
        pass
    else:
        d02_fname = get_wrfout_fname(initutc,2)
        d02_fpath = os.path.join(m01dir,d02_fname)
        d02_nc = Dataset(d02_fpath)

        d02_GRID = CaseGrid(d02_nc)
        # pdb.set_trace()
        neutral_lats, neutral_lons = d02_GRID.return_latlons()
        utils.trycreate(neutral_latf,debug=False)
        _save(arr=neutral_lats,file=neutral_latf)
        _save(arr=neutral_lons,file=neutral_lonf)

    # One last to compute:
    ### d01_3km (d01 cut to d02_raw bounds)

    d01_3km_latf, d01_3km_lonf = get_llf_fpath(casestr,"d01_3km")
    if os.path.exists(d01_3km_latf):
    #    neutral_lats = N.load(neutral_latf)
    #    neutral_lons = N.load(neutral_lonf)
        pass
    else:
        d01_latf, d01_lonf = get_llf_fpath(casestr,"d01_raw")
        d02_latf, d02_lonf = get_llf_fpath(casestr,"d02_raw")
        d01_lats = N.load(d01_latf)
        d01_lons = N.load(d01_lonf)
        d02_lats = N.load(d02_latf)
        d02_lons = N.load(d02_lonf)
        # new_lats,new_lons = interpolate(None,d01_lats,d01_lons,d02_lats,d02_lons,cut_only=True)
        new_lats,new_lons = utils.return_subdomain(None,d01_lats,d01_lons,cut_to_lats=d02_lats,cut_to_lons=d02_lons)
        #new_lats,new_lons = utils.return_subdomain(data=None,lats=d01_lats,
        #                            lons=d01_lons,cut_to_lats=d02_lats,
        #                            cut_to_lons=d02_lons)
        utils.trycreate(d01_3km_latf,debug=False)
        _save(arr=new_lats,file=d01_3km_latf)
        _save(arr=new_lons,file=d01_3km_lonf)
        # pdb.set_trace()

# Now the grids are there!

# Time to interpolate!

# This is a list of lists of arguments to send to interpolate().
# TODO: logic that saves the commands lists to pickle,
# as it takes ages to generate. Then, here we should check
# if that command pickle exists, and load it here. This is
# good for when the scripts need debugging or a new variable
# is computer (with bugs).
# commands = []

#### FIRST: forecast
# for caseutc, initutc, mem, vrbl in generate_fcst_loop():

### GENERATE COMMAND LISTS
print("GENERATING COMMAND LISTS.")

all_cmd = []
for func, genfunc in zip([gather_commands_one,gather_commands_two],
# for func, genfunc in zip([gather_commands_one,],
                [generate_fcst_loop,generate_obs_loop]):
                # [generate_fcst_loop,]):
    if ncpus == 1:
        for i in genfunc():
            all_cmd.append(func(i))
    else:
        with multiprocessing.Pool(ncpus) as pool:
            some_cmd = pool.map(func,genfunc())
            all_cmd.append(some_cmd)
join_all_cmd = all_cmd[0] + all_cmd[1]

print("SUBMIT JOBS.")

# Now to submit them


if (ncpus == 1) or (check_in_serial):
    join_all_cmd = [el for sublist in join_all_cmd for el in sublist]
    for co in join_all_cmd:
        toprint = ["{}\n".format(c) for c in co]
        print(stars,"\nRunning interpolate() with: \n",*toprint,stars)
        interpolate(*co)
        print(stars,stars,"DONE!",stars,stars)
        # pdb.set_trace()
else:
    join_all_cmd = [el for sublist in join_all_cmd for el in sublist]
    # itr = generate_itr_from_commands(join_all_cmd)
    itr = join_all_cmd
    # pdb.set_trace()
    with multiprocessing.Pool(ncpus) as pool:
        pool.map(submit_interp,itr)

print("GENERATING COMMANDS FOR OBS GRIDS.")
# New loop for obs data
commands = []

for vrbl, validutc, caseutc in generate_obs_loop():
    print("Appending interpolation commands for observational grids of",vrbl,"at", validutc)
    do_5km = False

    # do_nexrad = False
    do_nexrad = True if vrbl is "NEXRAD" else False
    if do_nexrad:
        save_to_fpath = get_extraction_fpaths(vrbl,"nexrad_1km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="nexrad_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        save_to_fpath = get_extraction_fpaths(vrbl,"nexrad_3km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="nexrad_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

    # MRMS prods
    # do_mrms = True
    do_mrms = True if ("mrms" in vrbl) else False
    if do_mrms:
        if "AWS" in vrbl:
            save_to_fpath = get_extraction_fpaths(vrbl,"mrms_aws_1km",validutc,caseutc)
            if not os.path.exists(save_to_fpath):
                # data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_1km",validutc=validutc,
                data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_raw",validutc=validutc,
                                        caseutc=caseutc,)
                latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
                commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

            # mrms_aws_3km (AWS 1km data interpolated to d01_3km)
            save_to_fpath = get_extraction_fpaths(vrbl,"mrms_aws_3km",validutc,caseutc)
            if not os.path.exists(save_to_fpath):
                # data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_1km",validutc=validutc,
                data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_raw",validutc=validutc,
                                        caseutc=caseutc,)
                latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
                commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

            if do_5km:
                # mrms_aws_5km (AWS 1km data interpolated to d01_5km)
                save_to_fpath = get_extraction_fpaths(vrbl,"mrms_aws_5km",validutc,caseutc)
                if not os.path.exists(save_to_fpath):
                    # data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_1km",validutc=validutc,
                    data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_aws_raw",validutc=validutc,
                                            caseutc=caseutc,)
                    latsB,lonsB =  get_data(caseutc,"neutral",latlon_only=True)
                    commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        # mrms_dz_1km (cref data interpolated to d02_1km)
        elif vrbl is "DZ":
            save_to_fpath = get_extraction_fpaths(vrbl,"mrms_dz_1km",validutc,caseutc)
            if not os.path.exists(save_to_fpath):
                data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_dz_raw",validutc=validutc,
                                        caseutc=caseutc,)
                latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
                commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

            # mrms_dz_3km (cref 1km data interpolated to d02_3km)
            save_to_fpath = get_extraction_fpaths(vrbl,"mrms_dz_3km",validutc,caseutc)
            if not os.path.exists(save_to_fpath):
                data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_dz_raw",validutc=validutc,
                                        caseutc=caseutc)
                latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
                commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

            if do_5km:
                # mrms_dz_5km (cref 1km data interpolated to d02_5km)
                save_to_fpath = get_extraction_fpaths(vrbl,"mrms_dz_5km",validutc,caseutc)
                if not os.path.exists(save_to_fpath):
                    data,latsA,lonsA = get_data(vrbl=vrbl,fmt="mrms_dz_raw",validutc=validutc,
                                            caseutc=caseutc)
                    latsB,lonsB =  get_data(caseutc,"neutral",latlon_only=True)
                    commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

    elif vrbl == "ST4":
        save_to_fpath = get_extraction_fpaths(vrbl,"stageiv_1km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="stageiv_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d02_raw",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        save_to_fpath = get_extraction_fpaths(vrbl,"stageiv_3km",validutc,caseutc)
        if not os.path.exists(save_to_fpath):
            data,latsA,lonsA = get_data(vrbl=vrbl,fmt="stageiv_raw",validutc=validutc,
                                    caseutc=caseutc,)
            latsB,lonsB =  get_data(caseutc,"d01_3km",latlon_only=True)
            commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

        if do_5km:
            save_to_fpath = get_extraction_fpaths(vrbl,"stageiv_5km",validutc,caseutc)
            if not os.path.exists(save_to_fpath):
                data,latsA,lonsA = get_data(vrbl=vrbl,fmt="stageiv_raw",validutc=validutc,
                                        caseutc=caseutc,)
                latsB,lonsB =  get_data(caseutc,"neutral",latlon_only=True)
                commands.append((data,latsA,lonsA,latsB,lonsB,False,save_to_fpath))

print("SUBMITTING INTERPOLATION COMMANDS.")
if ncpus == 1:
    for co in commands:
        toprint = ["{}\n".format(c) for c in co]
        print(stars,"\nRunning interpolate() with: \n",*toprint,stars)
        interpolate(*co)
        print(stars,stars,"DONE!",stars,stars)
        # pdb.set_trace()
else:
    itr = generate_itr_from_commands(commands)
    with multiprocessing.Pool(ncpus) as pool:
        # pool.map(interpolate,commands)
        # pool.map(interpolate,itr)
        pool.map(submit_interp,itr)
