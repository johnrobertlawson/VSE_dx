import pdb
import os
import collections
import datetime
import itertools

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import evac.utils as utils
from evac.datafiles.gribfile import GribFile
from evac.datafiles.narrfile import NARRFile

outroot = "/Users/john.lawson/data/figures"
narrroot = '/Users/john.lawson/data/AprilFool/NARR'
# The grib table is messed up
# Zstr = "Total precipitation anomaly of at least 10 mm"
Zstr = "HGT"
Zlv = "level 500"

CASES_narr = collections.OrderedDict()
CASES_narr[datetime.datetime(2016,3,31,0,0,0)] = datetime.datetime(2016,3,31,12,0,0)
CASES_narr[datetime.datetime(2017,5,1,0,0,0)] = datetime.datetime(2017,5,1,12,0,0)
CASES_narr[datetime.datetime(2017,5,2,0,0,0)] = datetime.datetime(2017,5,2,12,0,0)
CASES_narr[datetime.datetime(2017,5,4,0,0,0)] = datetime.datetime(2017,5,4,12,0,0)

# Load tornado report CSV to plot lat/lon


# Load severe hail reports, or draw a swathe of SVR reports with Python?
# Could plot hail reports as proxy for mesocyclonic activity.

fig,_axes = plt.subplots(ncols=4,nrows=4,figsize=(12,10))
fig_fname = "case_outline.png"
fig_fpath = os.path.join(outroot,fig_fname)
axes = _axes.flat

# For each case...
for caseutc, plotutc in CASES_narr.items():
    casestr = utils.string_from_time('dir',plotutc,strlen='hour')
    narr_fname = f'merged_AWIP32.{casestr}.3D'
    narr_fpath = os.path.join(narrroot,narr_fname)

    # Load NARR data
    # G = GribFile(fpath)
    G = NARRFile(narr_fpath)
    data_925 = G.get("HGT",level="level 925")[0,0,:,:]
    data_500 = G.get("HGT",level="level 500")[0,0,:,:]

    # lats = G.lats
    # lons = G.lons

    # Plot 500 hPa height, 925 hPa height from NARR
    ax = next(axes)
    m = create_bmap(50.0,-55.0,25.0,-130.0,ax=ax)
    x,y = m(G.lons,G.lats)
    m.drawcoastlines()
    m.drawmapboundary(fill_color='gray')
    m.fillcontinents(color="lightgray",lake_color="gray")
    m.drawstates()
    m.drawcountries()

    kws = {'linewidths':0.7}

    # mediumorchid
    # purple
    m.contour(x,y,data_500,colors='k',levels=N.arange(4000,6000,60),**kws)
    m.contour(x,y,data_925,colors='mediumorchid',levels=N.arange(0,2000,40),**kws)

    # Plot shear and CAPE from EE3km
    fcst_data, fcst_lats, fcst_lons = load_fcst_dll(fcst_vrbl,fcst_fmt,
                            validutc,caseutc,initutc,mem)
    fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                                    validutc=validutc,caseutc=caseutc,initutc=initutc,
                                    mem=mem)

    # maybe boundaries/theta-e?

    # Thumbnail of reflectivity with reports (tornado and/or hail)

    fig.savefig(fig_fpath)
    pdb.set_trace()

fig.savefig(fig_fpath)
plt.close(fig)
