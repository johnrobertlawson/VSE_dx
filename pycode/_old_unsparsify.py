import pdb

import numpy as N
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

np = N

def create_bmap():
    bmap = Basemap(width=12000000,height=9000000,
                rsphere=(6378137.00,6356752.3142),
                resolution='l',projection='lcc',
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.0)
    return bmap


def do_unsparsify(mrms_nc,vrbl,d01_nc):
    """
    Pat and Tony's method.

    Args:
        nc: Dataset object, MRMS data.
    """
    # lats, lons = get_mrms_rotdz_grid(nc=mrms_nc)


    ### Pat stuff
    newse_lat = d01_nc.variables['XLAT'][0,:,:]
    newse_lon = d01_nc.variables['XLONG'][0,:,:]
    cen_lat = d01_nc.CEN_LAT
    cen_lon = d01_nc.CEN_LON
    stand_lon = d01_nc.STAND_LON
    # TRUE_LAT1 in the original....
    true_lat1 = d01_nc.TRUELAT1
    true_lat2 = d01_nc.TRUELAT2
    resolution = 'i'
    area_thresh = 1000.0


    ### Pat's data extraction ###
    grid_type = mrms_nc.DataType


    ### 0-2 km shear

    if "Shear" in vrbl:
        ### Pat's building grid ###
        nw_lat = mrms_nc.Latitude
        nw_lon = mrms_nc.Longitude
        lat_dy = mrms_nc.LatGridSpacing
        lon_dx = mrms_nc.LonGridSpacing
        lat_length = len(mrms_nc.dimensions["Lat"])
        lon_length = len(mrms_nc.dimensions["Lon"])

        lat_range = N.arange((nw_lat-((lat_length-1)*lat_dy)),(nw_lat+0.00001),lat_dy)
        lon_range = N.arange(nw_lon,(nw_lon+((lon_length-0.99999)*lon_dx)),lon_dx)
        xlon_full, xlat_full = N.meshgrid(lon_range, lat_range)

        lon_range_indices = N.arange(0,lon_length)
        lat_range_indices = N.arange(0,lat_length)

        xlon_indices, xlat_indices = N.meshgrid(lon_range_indices, lat_range_indices)

        min_lat = (N.abs(xlat_full[:,0]-N.min(newse_lat))).argmin()
        max_lat = (N.abs(xlat_full[:,0]-N.max(newse_lat))).argmin()
        min_lon = (N.abs(xlon_full[0,:]-N.min(newse_lon))).argmin()
        max_lon = (N.abs(xlon_full[0,:]-N.max(newse_lon))).argmin()

        xlat = xlat_full[min_lat:max_lat,min_lon:max_lon]
        xlon = xlon_full[min_lat:max_lat,min_lon:max_lon]

        sw_xlat = xlat[0,0]
        sw_xlon = xlon[0,0]
        ne_xlat = xlat[-1,-1]
        ne_xlon = xlon[-1,-1]

        bmap = Basemap(llcrnrlon=sw_xlon, llcrnrlat=sw_xlat, urcrnrlon=ne_xlon, urcrnrlat=ne_xlat, projection='lcc', lat_1=true_lat1, lat_2=true_lat2, lat_0=cen_lat, lon_0=cen_lon, resolution = resolution, area_thresh = area_thresh)
        # bmap = create_bmap()

        x_offset, y_offset = bmap(cen_lon, cen_lat)
        x, y = bmap(xlon[:], xlat[:])

        x_full, y_full = bmap(xlon_full[:], xlat_full[:])

        x = x - x_offset
        y = y - y_offset

        x_full = x_full - x_offset
        y_full = y_full - y_offset

        min_x, min_y = bmap(lon_range[min_lon], lat_range[min_lat])
        max_x, max_y = bmap(lon_range[max_lon], lat_range[max_lat])

        min_x = min_x - x_offset
        max_x = max_x - x_offset
        min_y = min_y - y_offset
        max_y = max_y - y_offset
        # bmap = create_bmap()
        # x, y = bmap(lons, lats)

        ### HACKY
        # xlat_indices = N.indices(lats.shape)
        # xlon_indices = N.indices(lons.shape)
        # min_lat = N.min(lats)
        # min_lon = N.min(lons)
        # max_lat = N.max(lats)
        # max_lon = N.max(lons)
        aws_thresh_1 = 0.0
        # xlat_full = lats
        # xlon_full = lons

        mrms_low_var = vrbl

        if (grid_type[0:2] == 'Sp'): #If SparseLatLonGrid
            pixel = len(mrms_nc.dimensions["pixel"])
            if (pixel > 0):
                pixel_x_full = mrms_nc.variables["pixel_x"][:]
                pixel_y_full = mrms_nc.variables["pixel_y"][:]
                pixel_value_full = mrms_nc.variables[mrms_low_var][:]
                pixel_count_full = mrms_nc.variables["pixel_count"][:]

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
                            pixel_x = N.append(pixel_x, temp_x_index)
                            pixel_y = N.append(pixel_y, temp_y_index)
                            pixel_value = N.append(pixel_value, pixel_value_temp[i])

                #... tortured way of flipping lat values, but it works
                pixel_x = abs(pixel_x - (xlat_full.shape[0] - min_lat))
                pixel_y = pixel_y - min_lon

                mrms_low_pixel_x_val = x[pixel_x, pixel_y]
                mrms_low_pixel_y_val = y[pixel_x, pixel_y]
                mrms_low_pixel_value = pixel_value
                #KD Tree searchable index of mrms observations
                mrms_low_x_y = N.dstack([mrms_low_pixel_y_val, mrms_low_pixel_x_val])[0]
                data = mrms_low_x_y

        elif (grid_type[0:2] == 'La'): #if LatLonGrid
            pixel_value_full = mrms_nc.variables[mrms_low_var][:]
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
            mrms_low_x_y = N.dstack([mrms_low_pixel_y_val, mrms_low_pixel_x_val])[0]
            data = mrms_low_x_y
        else:
            raise Exception


    # DZ
    elif "Comp" in vrbl:
        mrms_dz_var = vrbl
        dz_thresh_1 = -35.0

        dz_nw_lat = mrms_nc.Latitude
        dz_nw_lon = mrms_nc.Longitude
        dz_lat_dy = mrms_nc.LatGridSpacing
        dz_lon_dx = mrms_nc.LonGridSpacing
        dz_lat_length = len(mrms_nc.dimensions["Lat"])
        dz_lon_length = len(mrms_nc.dimensions["Lon"])

        dz_lat_range = np.arange((dz_nw_lat-((dz_lat_length-1)*dz_lat_dy)),(dz_nw_lat+0.00001),dz_lat_dy)
        dz_lon_range = np.arange(dz_nw_lon,(dz_nw_lon+((dz_lon_length-0.99999)*dz_lon_dx)),dz_lon_dx)
        dz_xlon_full, dz_xlat_full = np.meshgrid(dz_lon_range, dz_lat_range)

        dz_lon_range_indices = np.arange(0,dz_lon_length)
        dz_lat_range_indices = np.arange(0,dz_lat_length)

        dz_xlon_indices, dz_xlat_indices = np.meshgrid(dz_lon_range_indices, dz_lat_range_indices)

        dz_min_lat = (np.abs(dz_xlat_full[:,0]-np.min(newse_lat))).argmin()
        dz_max_lat = (np.abs(dz_xlat_full[:,0]-np.max(newse_lat))).argmin()
        dz_min_lon = (np.abs(dz_xlon_full[0,:]-np.min(newse_lon))).argmin()
        dz_max_lon = (np.abs(dz_xlon_full[0,:]-np.max(newse_lon))).argmin()

        dz_xlat = dz_xlat_full[dz_min_lat:dz_max_lat,dz_min_lon:dz_max_lon]
        dz_xlon = dz_xlon_full[dz_min_lat:dz_max_lat,dz_min_lon:dz_max_lon]

        dz_sw_xlat = dz_xlat[0,0]
        dz_sw_xlon = dz_xlon[0,0]
        dz_ne_xlat = dz_xlat[-1,-1]
        dz_ne_xlon = dz_xlon[-1,-1]

        dz_map = Basemap(llcrnrlon=dz_sw_xlon, llcrnrlat=dz_sw_xlat, urcrnrlon=dz_ne_xlon, urcrnrlat=dz_ne_xlat, projection='lcc', lat_1=true_lat1, lat_2=true_lat2, lat_0=cen_lat, lon_0=cen_lon, resolution = resolution, area_thresh = area_thresh)

        dz_x_offset, dz_y_offset = dz_map(cen_lon, cen_lat)
        dz_x, dz_y = dz_map(dz_xlon[:], dz_xlat[:])

        dz_x_full, dz_y_full = dz_map(dz_xlon_full[:], dz_xlat_full[:])

        dz_x_full = dz_x_full - dz_x_offset
        dz_y_full = dz_y_full - dz_y_offset

        dz_x = dz_x - dz_x_offset
        dz_y = dz_y - dz_y_offset

        dz_min_x, dz_min_y = dz_map(dz_lon_range[dz_min_lon], dz_lat_range[dz_min_lat])
        dz_max_x, dz_max_y = dz_map(dz_lon_range[dz_max_lon], dz_lat_range[dz_max_lat])

        dz_min_x = dz_min_x - dz_x_offset
        dz_max_x = dz_max_x - dz_x_offset
        dz_min_y = dz_min_y - dz_y_offset
        dz_max_y = dz_max_y - dz_y_offset

        grid_type = mrms_nc.DataType

        if (grid_type[0:2] == 'Sp'): #If SparseLatLonGrid
            pixel = len(mrms_nc.dimensions["pixel"])
            if (pixel > 0):
                pixel_x_full = mrms_nc.variables["pixel_x"][:]
                pixel_y_full = mrms_nc.variables["pixel_y"][:]
                pixel_value_full = mrms_nc.variables[mrms_dz_var][:]
                pixel_count_full = mrms_nc.variables["pixel_count"][:]

                pixel_x_full = pixel_x_full[pixel_value_full > 0.01]
                pixel_y_full = pixel_y_full[pixel_value_full > 0.01]
                pixel_count_full = pixel_count_full[pixel_value_full > 0.01]
                pixel_value_full = pixel_value_full[pixel_value_full > 0.01]

                pixel_x_transpose = dz_xlat_indices.shape[0] - pixel_x_full
                pixel_x = pixel_x_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
                pixel_y = pixel_y_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
                pixel_value = pixel_value_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
                pixel_count = pixel_count_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
                pixel_value_temp = pixel_value[pixel_count > 1]
                pixel_x_temp = pixel_x[pixel_count > 1]
                pixel_y_temp = pixel_y[pixel_count > 1]
                pixel_count_temp = pixel_count[pixel_count > 1]
                for i in range(0, len(pixel_count_temp)):
                    for j in range(1, pixel_count_temp[i]):
                        temp_y_index = pixel_y_temp[i] + j
                        temp_x_index = pixel_x_temp[i]
                        if (temp_y_index < dz_max_lon):
                            pixel_x = np.append(pixel_x, temp_x_index)
                            pixel_y = np.append(pixel_y, temp_y_index)
                            pixel_value = np.append(pixel_value, pixel_value_temp[i])
            pixel_x = abs(pixel_x - (dz_xlat_full.shape[0] - dz_min_lat)) #... tortured way of flipping lat values, but it works
            pixel_y = pixel_y - dz_min_lon

            mrms_dz_pixel_x_val = dz_x[pixel_x, pixel_y]
            mrms_dz_pixel_y_val = dz_y[pixel_x, pixel_y]
            mrms_dz_pixel_value = pixel_value
            mrms_dz_x_y = np.dstack([mrms_dz_pixel_y_val, mrms_dz_pixel_x_val])[0] #KD Tree searchable index of mrms observations
            data = mrms_dz_x_y
        elif (grid_type[0:2] == 'La'): #if LatLonGrid
            pixel_value_full = mrms_nc.variables[mrms_dz_var][:]
            pixel_value_full = pixel_value_full.ravel()
            pixel_x_full = dz_xlat_indices.ravel()
            pixel_y_full = dz_xlon_indices.ravel()
            pixel_x_full = pixel_x_full[pixel_value_full > 0.01]
            pixel_y_full = pixel_y_full[pixel_value_full > 0.01]
            pixel_value_full = pixel_value_full[pixel_value_full > 0.01]

            pixel_x_transpose = dz_xlat_full.shape[0] - pixel_x_full
            pixel_x = pixel_x_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
            pixel_y = pixel_y_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]

            mrms_dz_pixel_value = pixel_value_full[(pixel_x_transpose > dz_min_lat) & (pixel_x_transpose < dz_max_lat) & (pixel_y_full > dz_min_lon) & (pixel_y_full < dz_max_lon) & (pixel_value_full > dz_thresh_1)]
            pixel_x = abs(pixel_x - (dz_xlat_full.shape[0] - dz_min_lat)) #... tortured way of flipping lat values, but it works
            pixel_y = pixel_y - dz_min_lon

            mrms_dz_pixel_x_val = dz_x[pixel_x, pixel_y]
            mrms_dz_pixel_y_val = dz_y[pixel_x, pixel_y]
            mrms_dz_x_y = np.dstack([mrms_dz_pixel_y_val, mrms_dz_pixel_x_val])[0] #KD Tree searchable index of mrms observations
            data = mrms_dz_x_y
        else:
            raise Exception

    pdb.set_trace()
    return data

