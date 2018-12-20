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
import scipy
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap

from evac.datafiles.wrfout import WRFOut
from evac.datafiles.obsgroup import ObsGroup
import evac.utils as utils

### SETTINGS ###

# Folder key for forecast data (wrfout)
key_wrf = 'ForReal_nco'
ensroot = '/scratch/john.lawson/WRF/VSE_reso/{}'.format(key_wrf)

# Folder key for post-processed fields (objects, lat/lon, etc)
key_pp = 'Xmas'
pp_root = '/work/john.lawson/VSE_reso/pp/{}'.format(key_pp)

# Folder key for scores (FSS, etc)
key_scores = 'Xmas'
scoreroot = '/work/john.lawson/VSE_reso/scores/{}'.format(key_scores)

# Folder key for plots
key_plot = 'Xmas'
plotroot = '/home/john.lawson/VSE_reso/pyoutput/{}'.format(key_plot)

st4dir = '/work/john.lawson/STAGEIV_data'
mrmsdir = 'work/john.lawson/MRMS_data/rot-dz'


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

# CASES = {
        # datetime.datetime(2016,3,31,0,0,0):[datetime.datetime(2016,3,31,22,0,0),],
        # }


##### OTHER STUFF #####
stars = "*"*10

fcstmins = N.arange(0,185,5)

### FUNCTIONS ###
def get_all_initutcs():
    """ Find all unique initialisation times.
    """
    SET = set()
    for caseutc,initutcs in CASES.items():
        for i in initutcs:
            SET.add(i)
    return list(SET)

initutcs = get_all_initutcs()
dom_names = ("d01","d02")
member_names = ['m{:02d}'.format(n) for n in range(1,37)]
# doms = (1,2)

class CaseGrid:
    def __init__(self,Wnc):
        """ Create new grid for 5 km data.

        wrfout_fpath should be the d02 domain.
        """
        self.bmap = create_bmap()

        if isinstance(Wnc,str):
            Wnc = Dataset(Wnc)
        Wlats = Wnc.variables['XLAT'][:]
        Wlons = Wnc.variables['XLON'][:]

        urcrnrlat = self.lats[-1,-1]
        urcrnrlon = self.lons[-1,-1]
        llcrnrlat = self.lats[0,0]
        llcrnrlon = self.lons[0,0]

        nx, ny = self.dx_to_nxny(dx,urcrnrlat,urcrnrlon,
                        llcrnrlat,llcrnrlon)
        lons, lats, xx, yy = self.bmap.makegrid(nx,ny,returnxy=True)
        self.lats = lats
        self.lons = lons
        self.xx = xx
        self.yy = yy

    def dx_to_nxny(self,dx,urcrnrlat,urcrnrlon,llcrnrlat,llcrnrlon):
        xW, yS = self.bmap(llcrnrlon,llcrnrlat)
        xE, yN = self.bmap(urcrnrlon,urcrnrlat)

        # km difference in each direction
        xdiff_km = abs(xE-xW)/1000
        ydiff_km = abs(yS-yN)/1000

        # nx needed to give dx requested
        nx = int(N.floor(xdiff_km/dx))
        # assume dx=dy
        ny = int(N.floor(ydiff_km/dx))

        return nx, ny

    def return_latlons(self):
        return self.lats, self.lons

    return

def get_extraction_fpaths(vrbl,fmt,validutc,caseutc=None,initutc=None,mem=None):
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
    if vrbl in ("Wmax","UH02","UH25","RAINNC"):
        # TODO: are we not just doing 5-min or 1-hr accum_precip?
        caseYYYYMMDD = caseutc
        initHHMM = initutc
        validHHMM = validutc
        fname = "{}_{}_{}_{}.npy".format(vrbl,fmt,caseYYYYMMDD,initHHMM,validHHMM,mem)
    elif vrbl in ("AWS02","AWS25","DZ","ST4"):
        utcYYYYMMDD = validutc
        utcHHMM = validutc
        fname = "{}_{}_{}_{}.npy".format(vrbl,fmt,utcYYYYMMDD,utcHHMM)
    return os.path.join(pp_root,fname)

def create_bmap():
    bmap = self.create_basemap(width=12000000,height=9000000,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection='lcc',
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
    return bmap

def closest(arr,val):
    return N.argmin(N.abs(arr-val))

#def cut(data,latsA,lonsA,latsB,lonsB):
#    return utils.return_subdomain(data,latsA,lonsA,cut_to_lats=d02_lats,
#                        cut_to_lons=d02_lons)

def interpolate(dataA,latsA,lonsA,latsB,lonsB,cut_only=False,cut_first=True,
                # remove_negative=False,
                save_to_fpath=None):
    """ Interpolate data on gridA to gridB.

    If cut_first, if gridA is a lot bigger (e.g., StageIV data), cut it to 
    a smaller size to improve interpolation efficiency.
    """
    if cut_only:
        return utils.return_subdomain(data,latsA,lonsA,cut_to_lats=latsB,
                                cut_to_lons=lonsB)
    if cut_first:
        Nlim = N.max(latsA)+0.5
        Elim = N.max(lonsA)+0.5
        Slim = N.min(latsA)-0.5
        Wlim = N.min(lonsA)-0.5

        Nidx = closest(latsA[:,0],Nlim)
        Eidx = closest(lonsA[0,:],Elim)
        Sidx = closest(latsA[:,0],Slim)
        Widx = closest(latsA[:,0],Wlim)

        s1 = slice(Sidx,Nidx+1)
        s2 = slice(Widx,Eidx+1)

        dataA = dataA[s1,s2]
        latsA = latsA[s1,s2]
        lonsA = lonsA[s1,s2]

    bmap = create_basemap()
    xxA, yyA = bmap(lonsA,latsA)
    xxB, yyB = bmap(lonsB,lonsB)
    xdB = len(xxB[:,0])
    ydB = len(yyB[0,:])
    dataB = scipy.interpolate.griddata([xxA.flat,yyA.flat],dataA.flat,
                [xxB.flat,yyB.flat],method='linear').reshape(xdB,ydB)
    if save_to_fpath:
        N.save(arr=dataB,file=save_to_fpath)
        return
    return dataB
    
def generate_all_runs():
    for initutc,dom,mem in itertools.product(initutcs,dom_names,member_names):
        yield initutc, dom, mem

def generate_all_folders():
    for initutc,mem in itertools.product(initutcs,member_names):
        yield initutc, mem

def load_data(nc_fpath,vrbl,npyfpaths):
    W = WRFOut(nc_fpath)
    data = W.get(vrbl=vrbl)[:,0,:,:]

    assert data.shape[0] == len(npyfpaths)

    for tidx,npyfpath in enumerate(npyfpaths):
        N.save(arr=data[tidx,:,:],file=npyfpath)
    return

def get_mrms_extraction_fpath(initstr,validutc,vrbl,fmt):
    HHMM = "{:02d}{:02d}".format(validutc.hour, validutc.minute)
    f = "{}_{}.npy".format(vrbl,HHMM)
    fpath = os.path.join(reproj_obs_root,initstr,fmt,f)
    return fpath

def get_fcst_extraction_fpaths(initstr,mem,vrbl,fmt,nt=37):
    assert isinstance(dom,str)
    assert isinstance(member,str)
    fpaths = []
    for t in range(nt):
        f = "{}_{}_t{:02d}.npy".format(vrbl,mem,t)
        fpath = os.path.join(reproj_fcst_root,initstr,fmt,f)
        fpaths.append(fpath)
    return fpaths

def get_wrfout_fname(utc,dom):
    fname = 'wrfout_d{:02d}_{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(dom,
                t.year,t.month,t.day,t.hour,t.minute,t.second)
    return fname

def get_llf_fpath(casestr,key):
    # LATS
    lats_fname = "lats.npy"
    lats_fpath = os.path.join(pp_root,key,casestr,lats_fname)

    lons_fname = "lons.npy"
    lons_fpath = os.path.join(pp_root,key,casestr,lons_fpath)

    return lats_fpath,lons_fpath

def unsparsify(nc,vrbl):
    """
    Pat and Tony's method.

    Args:
        nc: Dataset object, MRMS data.
    """
    grid_type = nc.DataType
    mrms_low_var = vrbl

    if (grid_type[0:2] == 'Sp'): #If SparseLatLonGrid 
        pixel = len(low_fin.dimensions["pixel"])
        if (pixel > 0):
            pixel_x_full = low_fin.variables["pixel_x"][:]
            pixel_y_full = low_fin.variables["pixel_y"][:]
            pixel_value_full = low_fin.variables[mrms_low_var][:]
            pixel_count_full = low_fin.variables["pixel_count"][:]

            pixel_x_full = pixel_x_full[pixel_value_full > 0.0001]  
            pixel_y_full = pixel_y_full[pixel_value_full > 0.0001]
            pixel_count_full = pixel_count_full[pixel_value_full > 0.0001]
            pixel_value_full = pixel_value_full[pixel_value_full > 0.0001]

            pixel_x_transpose = xlat_indices.shape[0] - pixel_x_full
            pixel_x = pixel_x_full[(pixel_x_transpose > min_lat) & 
                                    (pixel_x_transpose < max_lat) & 
                                    (pixel_y_full > min_lon) & 
                                    (pixel_y_full < max_lon) & 
                                    (pixel_value_full > aws_thresh_1)]
            pixel_y = pixel_y_full[(pixel_x_transpose > min_lat) & 
                                    (pixel_x_transpose < max_lat) & 
                                    (pixel_y_full > min_lon) & 
                                    (pixel_y_full < max_lon) & 
                                    (pixel_value_full > aws_thresh_1)]
            pixel_value = pixel_value_full[(pixel_x_transpose > min_lat) & 
                                    (pixel_x_transpose < max_lat) & 
                                    (pixel_y_full > min_lon) & 
                                    (pixel_y_full < max_lon) & 
                                    (pixel_value_full > aws_thresh_1)]
            pixel_count = pixel_count_full[(pixel_x_transpose > min_lat) & 
                                    (pixel_x_transpose < max_lat) & 
                                    (pixel_y_full > min_lon) & 
                                    (pixel_y_full < max_lon) & 
                                    (pixel_value_full > aws_thresh_1)]
            #      print 'aws low pixel count orig: ', len(pixel_value)

            pixel_value_temp = pixel_value[pixel_count > 1]
            pixel_x_temp = pixel_x[pixel_count > 1]
            pixel_y_temp = pixel_y[pixel_count > 1]
            pixel_count_temp = pixel_count[pixel_count > 1]

            for i in range(0, len(pixel_count_temp)):
                for j in range(1, pixel_count_temp[i]):
                    temp_y_index = pixel_y_temp[i] + j
                    temp_x_index = pixel_x_temp[i]            
                    if (temp_y_index < max_lon):
                        pixel_x = np.append(pixel_x, temp_x_index)
                        pixel_y = np.append(pixel_y, temp_y_index)
                        pixel_value = np.append(pixel_value, pixel_value_temp[i])

            #... tortured way of flipping lat values, but it works
            pixel_x = abs(pixel_x - (xlat_full.shape[0] - min_lat)) 
            pixel_y = pixel_y - min_lon

            mrms_low_pixel_x_val = x[pixel_x, pixel_y]  
            mrms_low_pixel_y_val = y[pixel_x, pixel_y]
            mrms_low_pixel_value = pixel_value
            #KD Tree searchable index of mrms observations
            mrms_low_x_y = np.dstack([mrms_low_pixel_y_val, mrms_low_pixel_x_val])[0] 

    elif (grid_type[0:2] == 'La'): #if LatLonGrid
        pixel_value_full = low_fin.variables[mrms_low_var][:]
        pixel_value_full = pixel_value_full.ravel()
        pixel_x_full = xlat_indices.ravel() 
        pixel_y_full = xlon_indices.ravel()

        pixel_x_full = pixel_x_full[pixel_value_full > 0.0001]  
        pixel_y_full = pixel_y_full[pixel_value_full > 0.0001]
        pixel_value_full = pixel_value_full[pixel_value_full > 0.0001]

        pixel_x_transpose = xlat_full.shape[0] - pixel_x_full
        pixel_x = pixel_x_full[(pixel_x_transpose > min_lat) & 
                                (pixel_x_transpose < max_lat) & 
                                (pixel_y_full > min_lon) & 
                                (pixel_y_full < max_lon) & 
                                (pixel_value_full > aws_thresh_1)]
        pixel_y = pixel_y_full[(pixel_x_transpose > min_lat) & 
                                (pixel_x_transpose < max_lat) & 
                                (pixel_y_full > min_lon) & 
                                (pixel_y_full < max_lon) & 
                                (pixel_value_full > aws_thresh_1)]
        mrms_low_pixel_value = pixel_value_full[(pixel_x_transpose > min_lat) & 
                                (pixel_x_transpose < max_lat) & 
                                (pixel_y_full > min_lon) & 
                                (pixel_y_full < max_lon) & 
                                (pixel_value_full > aws_thresh_1)]
        #... tortured way of flipping lat values, but it works
        pixel_x = abs(pixel_x - (xlat_full.shape[0] - min_lat)) 
        pixel_y = pixel_y - min_lon
        mrms_low_pixel_x_val = x[pixel_x, pixel_y]
        mrms_low_pixel_y_val = y[pixel_x, pixel_y]
        #KD Tree searchable index of mrms observations
        mrms_low_x_y = np.dstack([mrms_low_pixel_y_val, mrms_low_pixel_x_val])[0] 
    else: 
       print 'aws low unknown grid type!!!!'

    return mrms_low_x_y


def get_mrms_rotdz_grid(caseutc,vrbl):
    """
    AzShear and DZ grids start at same lat/lon but are of different
    resolutions.

    Listed Latitude/Longitude attributes are top left
    """
    assert vrbl in ("dz","aws")

    nc = open_random_rotdz(caseutc,vrbl)

    # if nc.DataType.startswith("Sparse"):

    ullat = nc.Latitude
    ullon = nc.Longitude
    nlat = nc.dimensions['Lat'].size
    nlon = nc.dimensions['Lon'].size
    dlat = nc.LatGridSpacing

    ss = slice(0,nlat*dlat,dlat)
    _x, _y = N.mgrid[ss,ss]
    lats = _x + ullat
    lons = _y + ullon

    return lats,lons

def load_mrms(utc,vrbl):
    """
    Return a 2-D array of data for a given time.

    Lat/lon?
    """
    
    data = unsparsify(nc)
    return data
    

def return_all_mrms(caseutc):
    """ Return sorted dictionary of times and files.
    """
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    if vrbl == "DZ":
        v = "MergedReflectivityQC"
    elif vrbl == "AzShear":
        v = 0

    if caseutc.year == 2016:
        pass
    elif caseutc.year == 2017:
        pass
    elif caseutc.year == 2018:
        pass
    else:
        # fdir = os.path.join(mrmsdir,casestr,"processed","nofake","merged",v,"00.00")
        raise Exception

    FILES = {}
    # Open all netcdf files in this folder, and create list of times
    nclist = glob.glob(os.path.join(fdir,"*.netcdf"))
    for f in nclist:
        utcstr, _ext = os.path.basename(nclist).split('.')
        assert _ext == "netcdf"
        fmt = "%Y%m%d-%H%D%S"
        utc = datetime.datetime.strptime(utcstr,fmt)

        FILES[utc] = fname
        utcs = sorted(FILES.keys())

    return sorted(FILES)
    


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
    assert caseutc.year in (2016,2017)
    assert vrbl in ("DZ","LLAWS","MLAWS")

    casestr = utils.string_from_time('dir',caseutc,strlen='day')

    if caseutc.year == 2016:
        dlat = 0.01
        if vrbl == "DZ":
            key = "MergedReflectivityQCComposite"
        elif vrbl == "LLAWS":
            key = 'MergedLLShear'
        elif vrbl == "MLAWS":
            key = 'MergedMLShear'

        fdir = os.path.join(mrmsroot,casestr,"processed","nofake","merged",
                        key,"00.00")
    elif caseutc.year == 2017:
        if vrbl == "DZ":
            dlat = 0.01
            key = 'MergedReflectivityQCComposite'
            subdir = "00.50"
        elif vrbl == "LLAWS":
            dlat = 0.005
            key = 'MergedAzShear_0-2kmAGL'
            subdir = "00.00"
        elif vrbl == "MLAWS":
            dlat = 0.005
            key = 'MergedAzShear_2-5kmAGL'
            subdir = "00.00"

        fdir = os.path.join(mrmsroot,casestr,key,subdir)
    
    return fdir, key, dlat


def open_random_rotdz(caseutc,vrbl):
    FILES = return_all_mrms(caseutc)
    f = FILES[utcs[0]]
    # return f
    return Dataset(f)



######### PROCEDURE #########
""" Numpy array extraction for Lawson et al, 2019.

The following are native grids, computed for each case:
    * d01_raw (3 km)
    * d02_raw (1 km)
    * stageiv_raw (4.7 km)
    * mrms_aws_raw (2016: 1 km; 2017: 0.5 km)
    * mrms_dz_raw (1 km)
    * neutral (5 km)

The following are extra grids computed from above, requiring cut data:
    * d01_3km (d01_raw cut to d02_raw bounds)
    
The following are cut or interpolated grids, from above, requiring interpolation:
    * d01_5km (d01 data interpolated to neutral_5km)
    * d02_1km (d02 data on native, d02_raw)
    * d02_3km (d02 data interpolated to d01_3km)
    * d02_5km (d02 data interpolated to neutral)
    * mrms_aws_1km (AWS data interpolated to d02_1km)
    * mrms_aws_3km (AWS data interpolated to d01_3km)
    * mrms_aws_5km (AWS data interpolated to neutral)
    * mrms_dz_1km (cref data interpolated to d02_1km)
    * mrms_dz_3km (cref data interpolated to d01_3km)
    * mrms_dz_5km (cref data interpolated to neutral)
    * stageiv_5km (Stage IV data interpolated to neutral)

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
# Variables for plotting
vrbls = ("maxW","accum_precip","REFL_comp",)#"UH02","UH05")
all_folders = list(generate_all_folders())

### Get Stage IV catalogue
ST4 = ObsGroup(st4dir,'stageiv')

# for i in all_folders:
for caseutc, initutc in CASES.items():
    casestr = utils.string_from_time('dir',caseutc,strlen='day')
    initstr = utils.string_from_time('dir',initutc,strlen='hour')

    ### First, let's make lat/lon arrays for each grid for this case
    ### Open if they exist, else create/save

    m01dir = os.path.join(ensroot,initstr,"m01")

    # d02
    d02_latf, d02_lonf = get_llf_fpath(casestr,"d02_raw")

    # TODO: we just need to loop to check the lat/lon
    # files have been created, or create them. We can
    # delete the lat/lon array files, and not load them.
    if os.path.exists(d02_latf):
        d02_lats = N.load(do2_latf)
        d02_lons = N.load(d02_lonf)
    else:
        d02_nc = Dataset(d01_fname)
        d02_lats = d02_nc.variables['XLAT'][:]
        d02_lons = d02_nc.variables['XLON'][:]

        d02_fname = get_wrfout_name(initutc,2)
        d02_fpath = os.path.join(m01dir,do2_fname)
        d02_nc = Dataset(d02_fname)
        d02_lats = d02_nc.variables['XLAT'][:]
        d02_lons = d02_nc.variables['XLON'][:]

        # Make sure these are 2D.
        assert d02_lats.ndim == 2

        N.save(arr=d02_lats,file=d02_latf)
        N.save(arr=d02_lons,file=d02_lonf)

    # d01
    d01_latf, d01_lonf = get_llf_fpath(casestr,"d01_raw")

    if os.path.exists(d01_latf):
        d01_lats = N.load(do1_latf)
        d01_lons = N.load(d01_lonf)
    else:
        d01_nc = Dataset(d01_fname)
        d01_lats = d01_nc.variables['XLAT'][:]
        d01_lons = d01_nc.variables['XLON'][:]

        d01_fname = get_wrfout_name(initutc,1)
        d01_fpath = os.path.join(m01dir,do1_fname)
        d01_nc = Dataset(d01_fname)
        d01_lats = d01_nc.variables['XLAT'][:]
        d01_lons = d01_nc.variables['XLON'][:]

        N.save(arr=d01_lats,file=d01_latf)
        N.save(arr=d01_lons,file=d01_lonf)

    # STAGE IV
    st4_latf, st4_lonf = get_llf_fpath(casestr,"stageiv_raw")
    if os.path.exists(st4_latf):
        st4_lats = N.load(st4_latf)
        st4_lons = N.load(st4_lonf)
    else:
        st4_lats, st4_lons = ST4.return_latlon()
        assert st4_lats.ndim == 2

        N.save(arr=st4_lats,file=st4_latf)
        N.save(arr=st4_lons,file=st4_lonf)

    # MRMS LLAWS/MLAWS (rot)
    mrms_aws_latf, mrms_aws_lonf = get_llf_fpath(casestr,"mrms_aws_raw")
    if os.path.exists(mrms_aws_latf):
        mrms_aws_lats = N.load(mrms_aws_latf)
        mrms_aws_lons = N.load(mrms_aws_lonf)
    else:
        # mrms_fpath = get_mrms_rotdz_fpath(caseutc)
        # mrms_rotdz_nc = Dataset(mrms_fpath)
        mrms_aws_lats,mrms_aws_lons = get_mrms_rotdz_grid(caseutc,"aws")
        N.save(arr=mrms_aws_lats,file=mrms_aws_latf)
        N.save(arr=mrms_aws_lons,file=mrms_aws_lonf)

    # MRMS DZ (cref)
    mrms_dz_latf, mrms_dz_lonf = get_llf_fpath(casestr,"mrms_dz_raw")
    if os.path.exists(mrms_dz_latf):
        mrms_dz_lats = N.load(mrms_dz_latf)
        mrms_dz_lons = N.load(mrms_dz_lonf)
    else:
        # mrms_fpath = get_mrms_rotdz_fpath(caseutc)
        # mrms_rotdz_nc = Dataset(mrms_fpath)
        mrms_dz_lats,mrms_dz_lons = get_mrms_rotdz_grid(caseutc,"dz")
        N.save(arr=mrms_dz_lats,file=mrms_dz_latf)
        N.save(arr=mrms_dz_lons,file=mrms_dz_lonf)

    # Neutral grid
    neutral_latf, neutral_lonf = get_llf_fpath(casestr,"neutral")
    if os.path.exists(neutral_latf):
        neutral_lats = N.load(neutral_latf)
        neutral_lons = N.load(neutral_lonf)
    else:
        d02_GRID = CaseGrid(d02_nc)
        neutral_lats, neutral_lons = d02_GRID.return_latlons()
        N.save(arr=neutral_lats,file=neutral_latf)
        N.save(arr=neutral_lons,file=neutral_lonf)

    # One last to compute:
    ### d01_3km (d01 cut to d02_raw bounds) 
    
    d01_3km_latf, d01_3km_lonf = get_llf_fpath(casestr,"d01_3km")
    if os.path.exists():
    #    neutral_lats = N.load(neutral_latf)
    #    neutral_lons = N.load(neutral_lonf)
        pass
    else:
        new_lats,new_lons = cut(None,d01_lats,d01_lons,d02_lats,d02_lons)
        #new_lats,new_lons = utils.return_subdomain(data=None,lats=d01_lats,
        #                            lons=d01_lons,cut_to_lats=d02_lats,
        #                            cut_to_lons=d02_lons)
        N.save(arr=new_lats,file=d01_3km_latf)
        N.save(arr=new_lons,file=d01_3km_lonf)

# Now the grids are there!

# Time to interpolate!

# This is a list of lists of arguments to send to interpolate().
commands = []
# for case, mem, vrbl, utc?
for case in cases:
    # Do verification separately, and loop over all verification times
    # HERE

    # Forecast products
    for mem in member_names:
        memdir = os.path.join(ensroot,initstr,mem)

        ### d01_3km (d01 cut to d02)
        for utc in valid_utcs:
            save_to_fpath = get_extraction_fpaths(vrbl,"d01_3km",utc,
            if not os.path.exists(save_to_fpath):
                data, latsA, lonsB = get_data(case, vrbl, utc, "d01_raw")
                #lats, lons =  get_latlons(case, vrbl, utc, "d01_3km")
                latsB, lonsB =  get_latlons(case, vrbl, utc, "d02_raw")

                # Need fpath for interpolated grids to be saved.
                commands.append(data,latsA,lonsA,latsB,lonsB,True,save_to_fpath)

        ### d01_5km (d01 interpolated to neutral_5km)

        
        # Create .npy file name
        # If not there, append the function to the command list
        commands.append(interpolate, )

















        # d02_1km

        # Get filepath for d02
        fname = utils.string_from_time('wrfout',initutc,dom="d02")
        d02_nc_fpath = os.path.join(memdir,fname)

        # d02 native 1km grid
        for vrbl in vrbls:
            npyfs = get_extraction_fpaths(initstr=initstr,mem=mem,
                                            vrbl=vrbl,fmt='d02_1km')
            commands.append([d02_nc_fpath, vrbl, npyfs])

        # Save d02 native 1km lat/lon grid
        if mem == 'm01':
            for vrbl in ("XLAT","XLONG"):
                npyf = get_extraction_fpaths(initstr=initstr,mem=mem,
                                            vrbl=vrbl,fmt="d02_1km",nt=1)
                commands.append((d02_nc_fpath, vrbl, npyf))

# Now to submit them
with multiprocessing.pool(ncpus) as Pool:
    pool.map(load_data,commands)

# d01_3km and d01_5km
commands = []
for i in all_folders:
    # This time, load lat/lon within functions for interp
    do2latf = get_extraction_fpaths(initstr=initstr,mem=mem,
                                vrbl="XLAT",fmt="d02_1km",nt=1)
    d02lonf = get_extraction_fpaths(initstr=initstr,mem=mem,
                                vrbl="XLON",fmt="d02_1km",nt=1)

    # Get filepath for d01
    fname = utils.string_from_time('wrfout',initutc,dom="d01")
    d01_nc_fpath = os.path.join(memdir,fname)

    # Cut d01 to d02 and save native 3km array
    # Save new lat/lon too
    for vrbl in vrbls:
        npyfs = get_extraction_fpaths(initstr=initstr,mem=mem,
                                     vrbl=vrbl,fmt="d01_3km")
        commands.append(d01_nc_fpath, vrbl, npyfs, d02latf, d02lonf)

    # Interpolate to 5 km for stageiv
    

# Now to submit them
with multiprocessing.pool(ncpus) as Pool:
    pool.map(do_fcst_interp,commands)
 

    # Now we can interp d02 to 3 km:
    for vrbl in vrbls:
        npyfs = get_extraction_fpaths(initstr=initstr,mem=mem,
                                     vrbl=vrbl,fmt="d02_3km")
        yield do_fcst_interp, d02_nc_fpath, vrbl, npyfs, d01_cutlats, d01_cutlons

    # Interp to 5km and save
    for vrbl in vrbls:
        npyfs = get_extraction_fpaths(initstr=initstr,mem=mem,
                                     vrbl=vrbl,fmt="d01_5km")
        yield do_fcst_interp, d01_nc_fpath, vrbl, npyfs, st4lats, st4lons


    # Load mrms for this timerange
    for utc in fcst_utcs:
        fname = utils.string_from_time('mrms',initutc,dom="d01")
        FOLDER = "NEED FOLDERS HERE"
        nc_fpath = os.path.join(memdir,FOLDER,fname)

        # Save 1km npy (domain of d02)
        for vrbl in ("DZ","AWS02","AWS25"):
            # npyfs = get_extraction_fpaths(initstr,2,mem,vrbl,fmt,nt=1)
            npyf = get_mrms_extraction_fpath(initstr,utc,vrbl,fmt)
            yield do_mrms_to_1km, nc_fpath, npy_fpath, d02lats, d02lons, vrbl


        # Save 3km npy (domain of d01)

return

