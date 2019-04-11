import glob
import pdb

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm


def sparseFull(a):

    """This expands a sparse grid, incorporating all data values"""
    """SLOWWWWWWWWW"""

    # Create empty shape for filling
    b = np.full([len(a.dimensions['Lat']),len(a.dimensions['Lon'])], np.nan)

    # Create empty shape for filling
    xlen = len(a.dimensions['Lon'])
    ylen = len(a.dimensions['Lat'])

    # Gather into four arrays
    varname = a.TypeName
    var = a.variables[varname][:]
    pixel_x = a.variables['pixel_x'][:]
    pixel_y = a.variables['pixel_y'][:]
    pixel_count = a.variables['pixel_count'][:]

    # Loop through the four arrays simultaneously, populating 2D array 'b'
    for w,x,y,z in zip(var, pixel_x, pixel_y, pixel_count):
        distToEdge = xlen - y
        # check if pixel count exceeds distance to right edge of domain
        if z > distToEdge:
            b[x,y:xlen] = w
            pixelsLeft = z - distToEdge
            # check if pixels remaining are less than grid width
            if pixelsLeft <= xlen:
                b[x+1,0:(pixelsLeft-1)] = w
            else:
                rowsLeft, pixelCount_RemainingRow = divmod(pixelsLeft, xlen)
                b[x:(x+rowsLeft+1),0:7001] = w
                # check if pixels are remaining
                if pixelCount_RemainingRow > 0:
                    b[(x+rowsLeft+1),0:pixelCount_RemainingRow] = w
        else:
            b[x,y:(y + z)] = w

    return b


def readwdssii(fin):

    """This loads in the file"""

    if isinstance(fin,str):
        a = nc.Dataset(fin,)#format="NETCDF3_64BIT_OFFSET")
    else:
        a = fin

    xlen = len(a.dimensions['Lon'])
    ylen = len(a.dimensions['Lat'])
    lat = a.Latitude
    lon = a.Longitude
    try:
        varname = a.TypeName


        var = a.variables[varname][:]

        if a.DataType[0:6] == 'Sparse':

            var = sparseFull(a)

    except:
        print('except')
        return xlen,ylen,varname,0

    # pdb.set_trace()
    a.close()

    return xlen, lat, ylen, lon, varname, var

