import os
import pdb

import matplotlib as M
import matplotlib.pyplot as plt

import evac.utils as utils

fig,axes = plt.subplots(nrows=4,ncols=3)

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

KEYTIMES = {
        datetime.datetime(2016,3,31,0,0,0) : datetime.datetime(2016,3,31,18,0,0),
        datetime.datetime(2017,5,1,0,0,0) : datetime.datetime(2017,5,1,18,0,0),
        datetime.datetime(2017,5,2,0,0,0) : datetime.datetime(2017,5,2,21,0,0),
        datetime.datetime(2017,5,4,0,0,0) : datetime.datetime(2017,5,4,21,0,0),



for n, ax in axes.flat:
    # nc is column number
    # nr is row number

    # https://rda.ucar.edu/datasets/ds608.0/
    # Load reanalysis data depending on row

    # Plot data depending on column
    if nc == 0:
        # CAPE and SHEAR
    elif nc == 1: 
        # 500/925 hPa Z and sfc fronts
        axes[0,n].contourf()
    elif nc == 2:
        # Reflectivity + tornado reports
