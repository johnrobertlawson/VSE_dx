""" Plot and return percentile data from extracted VSE_dx data
"""

import os
import pdb
import collections
import datetime

import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt

### paths etc
# dataroot = '/Users/john.lawson/data/AprilFool'
# extractroot = '/Users/john.lawson/data/AprilFool'
extractroot = '/work/john.lawson/VSE_reso/pp/AprilFool'
# outdir = '/Users/john.lawson/VSE_dx/pyoutput'
outdir = '/home/john.lawson/VSE_dx/pyoutput'

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

debug_cases = None
if debug_cases == 'one':
    CASES = collections.OrderedDict()
    CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                            datetime.datetime(2016,3,31,22,0,0),]
elif debug_cases == 'four':
    CASES = collections.OrderedDict()
    CASES[datetime.datetime(2016,3,31,0,0,0)] = [
                            datetime.datetime(2016,3,31,21,0,0),
                            ]
    CASES[datetime.datetime(2017,5,1,0,0,0)] = [
                            datetime.datetime(2017,5,1,21,0,0),
                            ]
    CASES[datetime.datetime(2017,5,2,0,0,0)] = [
                            datetime.datetime(2017,5,3,1,0,0),
                            ]
    CASES[datetime.datetime(2017,5,4,0,0,0)] = [
                            datetime.datetime(2017,5,5,0,0,0),
                            ]
else:
    pass


def get_extraction_fpaths(vrbl,fmt,validutc,caseutc,initutc=None,mem=None):
    """ Return the file path for the .npy of an interpolated field.

    Something like:

    FORECASTS:
    uh02_d02_3km_20160331_0100_0335_m01.npy

    OBS:
    aws02_mrms_rot_3km_20160331_0335.npy
    """
    fcst_vrbls = ("REFL_comp",)
    obs_vrbls = ("NEXRAD",)
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
        fname = "{}_{}_{}.npy".format(fmt,caseYYYYMMDD,utcHHMM)
    else:
        raise Exception
    return os.path.join(extractroot,caseYYYYMMDD,vrbl,fname)

def loop_obj_fcst(fcst_vrbl,fcstmins,fcst_fmt,members,shuffle=False):
    if shuffle is True:
        f = shuffled_copy
    else:
        f = list

    for mem in f(members):
        for fcstmin in f(fcstmins):
            for caseutc, initutcs in CASES.items():
                for initutc in f(initutcs):
                    validutc = initutc+datetime.timedelta(
                                        seconds=60*int(fcstmin))
                    yield fcst_vrbl, fcst_fmt, validutc, caseutc, initutc, mem

def loop_obj_obs(obs_vrbl):
    obtimes = set()
    casetimes = {}
    for caseutc, initutcs in CASES.items():
        casetimes[caseutc] = set()
        for initutc in initutcs:
            for t in fcstmins:
                validutc = initutc+datetime.timedelta(seconds=60*int(t))
                obtimes.add(validutc)
                casetimes[caseutc].add(validutc)
    for t in obtimes:
        case = None
        for caseutc in CASES.keys():
            if t in casetimes[caseutc]:
                case = caseutc
                break
        assert case is not None
        yield obs_vrbl, t, case

# Load data
# REFL_comp_d01_3km_20160331_1900_1900_m30.npy
# DZ_mrms_dz_3km_20160331_2300.npy
#NEXRAD_nexrad


BIG_DATAS = dict()
domains = ("d01_3km","d02_1km","NEXRAD_nexrad_3km","NEXRAD_nexrad_1km")
fcst_domains = ("d01_3km","d02_1km")
obs_domains = ("NEXRAD_nexrad_3km","NEXRAD_nexrad_1km")
nmems = 18
member_names = ['m{:02d}'.format(n) for n in range(1,nmems+1)]

for domain in domains:
    BIG_DATAS[domain] = None


fcst_vrbl = "REFL_comp"
obs_vrbl = "NEXRAD"
fcstmins = N.arange(5,185,5)

# FCSTS FIRST
for domain in fcst_domains:
    itr = loop_obj_fcst(fcst_vrbl,fcstmins,domain,member_names)
    for i in itr:
        fcst_vrbl, fcst_fmt, validutc, caseutc, initutc, mem = i

        fcst_fpath = get_extraction_fpaths(vrbl=fcst_vrbl,fmt=fcst_fmt,
                validutc=validutc,caseutc=caseutc,initutc=initutc,
                mem=mem)
        data = N.load(fcst_fpath).flatten()

        data[data<0.0] = 0.0

        if BIG_DATAS[domain] is None:
            BIG_DATAS[domain] = data
        else:
            BIG_DATAS[domain] = N.append(BIG_DATAS[domain],data)

for domain in obs_domains:
    itr = loop_obj_obs(obs_vrbl)
    for i in itr:
        obs_vrbl, t, case = i

        obs_fpath = get_extraction_fpaths(vrbl=obs_vrbl,fmt=domain,
                            validutc=t,caseutc=case)
        data = N.load(obs_fpath).flatten()

        data[data<0.0] = 0.0

        if BIG_DATAS[domain] is None:
            BIG_DATAS[domain] = data
        else:
            BIG_DATAS[domain] = N.append(BIG_DATAS[domain],data)

# pdb.set_trace()
colors = ['brown','blue','red','green']
COLORS = {}
for n,domain in enumerate(domains):
    COLORS[domain] = colors[n]

fig,ax = plt.subplots(1)
fname = f"pc_scatter.png"
fpath = os.path.join(outdir,fname)
for domain in domains:
    label = domain
    c = COLORS[domain]
    for pc in N.arange(90,100,0.1):
        ax.scatter(pc,N.nanpercentile(BIG_DATAS[domain],pc),
            color=c,s=10,marker='*',label=label)
        label = None
    # ax.set_xticklabels(N.arange(85,100,0.1),N.arange(85,100,0.1))

ax.legend()
fig.tight_layout()
fig.savefig(fpath)
plt.close(fig)
# TRY: load all data at once and get data
# IF MEMORY LIMIT: try rounding to 0.001 dBZ; or bin by same, looping by time


### Count each percentile (try different ranges; more granularity for high pcs)


### Plot scatter plot for the four domains (2x obs, 2x fcst EPS)


### Determine best percentile for object identification
# From graph or via numpy ?


# do test
