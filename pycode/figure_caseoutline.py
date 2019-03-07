import pdb
import os

import numpy as N

import evac.utils as utils
from evac.datafiles.gribfile import GribFile


# Load NARR data
G = GribFile(f)

# Load tornado report CSV to plot lat/lon


# Load severe hail reports, or draw a swathe of SVR reports with Python?
# Could plot hail reports as proxy for mesocyclonic activity.


# For each case...
for case in cases:

    # Plot shear and CAPE from NARR

    # Plot 300 hPa winds, 500 hPa height, MSLP from NARR
    # maybe sfc fronts estimated from theta-e or WPC analyses?

    # Thumbnail of reflectivity with reports (tornado and/or hail)


fig.savefig(fpath)
plt.close(fig)
