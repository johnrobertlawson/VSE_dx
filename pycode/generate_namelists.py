""" This script generates namelist.input files for
the VSE grid-resolution experiment. 

Change "settings and constants" to customise (e.g., number
of ensemble members). Script will return N namelist.input.n files,
where N is the number of ensembles, and n is from
1 to N inclusive. This file is placed in the folder where
the script runs from.

Usage from command line:

$ python3 generate_namelists.py [--onlydo x] [--overwrite] 
                                [--donotenforce] 
                                [--casestr YYYYMMDD] [--initstr YYYYMMDDHH]

Option --onlydo x, where 0 < x < N, only does one member (optional)
Option --overwrite will overwrite any existing files in the script directory
Option --donotenforce will ignore any missing parameters in the namelist.input
Option --casestr YYYYMMDD will choose setting particular to the case
Option --initstr YYYYMMDDHH will choose setting particular to the init time
"""
######## IMPORTS ########
import os
import argparse
import random
import glob
import numpy as N
import itertools
import pdb
import time
import sys
import datetime

######## CONSTANTS AND SETTINGS ########
#### USER-CONFIGURABLE HERE! ! ! ! #####
history_outname = False # Change this to absolute path of output file and location
debug = False # Print more output
runlen = 3 # Run length in hours

######## COMMAND LINE LOGIC ########
parser = argparse.ArgumentParser()
parser.add_argument('--onlydo', type=int, default=False)
parser.add_argument('--key', type=str, default=False)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--donotenforce',action='store_false')
parser.add_argument('--casestr',type=int)
parser.add_argument('--initstr',type=int)
parser.add_argument('--nmems',type=int)
NS = parser.parse_args()

nens = int(NS.nmems) # Number of ensemble members
assert 0 < nens < 100

# option to only generate one namelist
onlydo = NS.onlydo 

# Kills script if the template namelist is missing key parameters
enforce_all = NS.donotenforce 

# Year from case date and init time
casestr = str(NS.casestr)
casedt = datetime.datetime.strptime(casestr,"%Y%m%d")
caseyear, casemonth, caseday, *_ = list(casedt.timetuple())
casehour = 0

initstr = str(NS.initstr)
initdt = datetime.datetime.strptime(initstr,"%Y%m%d%H")
inityear, initmonth, initday, inithour, *_ = list(initdt.timetuple())

# End date for run
enddt = initdt + datetime.timedelta(seconds=int(60*60*runlen))
endyear, endmonth, endday, endhour, *_ = list(enddt.timetuple())

if NS.key:
    key = NS.key
else:
    raise Exception

path_to_namelist_template = '/home/john.lawson/VSE_reso/pycode/namelists/{}/{}/namelist.input.template'.format(
                                    key,NS.casestr)

######## LOGIC FOR SETTINGS ########
ensnums = range(1,nens+1) # names of each ensemble
nidxs = range(nens) # indices of each ensemble
doms = 2 # number of domains
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'namelists',key,str(NS.casestr))
#outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),NS.casestr,'namelists') # Where to output namelist files
namelists = ['namelist.input.{0}'.format(n) for n in ensnums]
paths_to_nl = {}
for nidx,n in zip(nidxs,ensnums):
    paths_to_nl[n] = {'old': path_to_namelist_template, 'new':os.path.join(outdir,namelists[nidx])}

######## FIRST CHECK TO SEE IF NAMELISTS MIGHT BE OVERWRITTEN ########
# nlfs = glob.glob(os.path.join(cwd,'namelist.*'))
# for k,v in paths_to_nl[n].items():
for nidx,n in zip(nidxs,ensnums):
    if onlydo and (n != onlydo):
        print("Skipping this one.")
        continue
    if os.path.isfile(paths_to_nl[n]['new']):
        if not NS.overwrite:
            raise Exception("Add --overwrite to command line to ignore existing namelists.")
        else:
            rmcmd = 'rm {0}'.format(paths_to_nl[n]['new'])
            os.system(rmcmd)


######## FUNCTIONS ########
def edit_namelist(f,sett,newval,enforce_all=False,doms=1,precision=3):
    """ A single integer or float will be repeated to all domains.
    A list/tuple of values, length dom, will be assigned the each domain.
    A list/tuple of one value (len 1) will be assigned only do the first domain
        (i.e. the rest are blank).

    if newval == None, it deletes the entire line.
    """
    if doms < 1:
        raise ValueError

    # Create list of values for writing to line
    if newval is None:
        newvals = None
    elif not isinstance(newval,(tuple,list)):
        newvals = [newval,]*doms
    else:
        if len(newval) == 1:
            # Only one value despite more than one domain
            newvals = [newval[0],]
        else:
            newvals = list(newval)
    
    # Set format of number
    if newvals is None:
        pass
    elif isinstance(newvals[0],(float,N.float)):
        fm = "{:.3f}"
    elif isinstance(newvals[0],(int,N.int)):
        fm = "{}"
    elif isinstance(newvals[0],str):
        fm = "{}"
    else:
       raise Exception("Format of number is",type(newvals[0]))

    # Get length of string so we can make the namelist pretty
    if newvals is not None:
        pad = 6 - len(eval(''.join(("'",fm,"'",'.format(newvals[0])'))))
        if pad<1:
            pad = 1

    fs = open(f,'r')
    # with open(f,'r') as fopen:
    # flines = open(f,'r').readlines()
    flines = fs.readlines()
    # pdb.set_trace()
    done= False
    for idx, line in enumerate(flines):
        if sett in line:
            if newvals is None:
                flines[idx] = " \n"
            else:
                # Make sure the match occurs at start of line
                # OR begins with a space
                # (No partial matches!)
                if ' '+sett in line:
                    pass
                elif (line.find(sett)) == 0:
                    pass
                else:
                    continue

                # Make sure the match ends in a space or equals
                if sett+' ' in line:
                    pass
                elif sett+'=' in line:
                    pass
                else:
                    continue

                # print("Setting {0} is in line.".format(sett))
                fs.close()
                # spaces = 38 - len(sett)
                spaces = 36
                # print(sett,spaces)
                # if not isinstance(newval,(tuple,list)):
                newval = ''.join([''.join((fm,',',' '*pad)).format(v) for v in newvals])
                # newval = ''.join(fm,',',' '*5).format(newval)*doms
                # elif doms == 2:
                    # newval = '{0},     {1},'.format(newval[0],newval[1])
                # elif doms == 3:
                    # newval = '{0},     {1},     {2},'.format(newval[0],newval[1],newval[2])

                flines[idx] = " {0: <{sp}}= {1}\n".format(sett,newval,sp=spaces)
            nameout = open(f,'w',1)
            nameout.writelines(flines)
            nameout.close()
            done = True
            break
    fs.close()
    if not done:
        if enforce_all:
            raise ValueError("Setting",sett,"not found in namelist.")
        else:
            print("Warning: setting",sett,"not found in namelist.")
    return


######## CREATE DICTIONARY OF NAMELIST SETTINGS ########
NL = {n:{} for n in ensnums}

### SETTINGS FOR ALL MEMBERS
for n in ensnums:

    NL[n]['reset_interval1'] = None
    # Set output location/name of wrfout
    if history_outname:
        NL[n]['history_outname'] = "'{0}'".format(history_outname)
        # ALLDICT['cu_physics'] = 0

# Urban physics and surface physics are just default for both years
#sf_urban_physics = 0 
# ALLDICT['sf_surface_physics'] = 3

####### VSE SETTINGS #######
#### SETTINGS THAT DEPEND ON THE CASE ####

for n in ensnums:
    # check date/time
    NL[n]['run_hours'] = (3,)
    NL[n]['start_month'] = initmonth
    NL[n]['start_day'] = initday
    NL[n]['start_hour'] = inithour
    NL[n]['end_month'] = endmonth
    NL[n]['end_day'] = endday
    NL[n]['end_hour'] = endhour
    # NL[n]['

    # inner grid size
    if casestr == '20170501':
        d01 = 221
        d02 = 322
    elif casestr in ('20170502','20170504'):
        d01 = 251
        d02 = 322
    elif casestr == '20160331':
        d01 = 250
        d02 = 322
    else:
        raise Exception

    NL[n]['e_we'] = (d01,d02)
    NL[n]['e_sn'] = (d01,d02)
    # NL[n]['
    
    # inner grid location
    if True:
        ii = 61
        jj = 61
    NL[n]['i_parent_start'] = (0,ii)
    NL[n]['j_parent_start'] = (0,jj)

    # 2016 v 2017
    if caseyear == 2016:
        NL[n]['mp_physics'] = 8 # Thompson
    else:
        NL[n]['mp_physics'] = 18 # NSSL 2-mom

for n in ensnums:
    NL[n]['input_from_file'] = ('.true.','.false.')
    NL[n]['fine_input_stream'] = 0
    NL[n]['history_interval'] = 5
    NL[n]['frames_per_outfile'] = 9999
    NL[n]['cycling'] = ('.false.',)
    NL[n]['all_ic_times'] = ('.false.',)
    # NL[n]['time_step'] = 9
    NL[n]['max_dom'] = (2,)
    NL[n]['e_vert'] = 51
    NL[n]['dx'] = (3000.0,1000.0)
    NL[n]['dy'] = (3000.0,1000.0)
    NL[n]['parent_id'] = (0,1)
    NL[n]['i_parent_start'] = (0,61)
    NL[n]['j_parent_start'] = (0,61)
    NL[n]['parent_grid_ratio'] = (1,3)
    NL[n]['parent_time_step_ratio'] = (1,3)
    NL[n]['numtiles'] = (1,)
    NL[n]['nproc_x'] = (-1,)
    NL[n]['nproc_y'] = (-1,)
    NL[n]['reorder_mesh'] = ('.false.',)
    NL[n]['radt'] = 1
    # NL[n]['

######## SETTINGS THAT DEPEND ON THE EXPERIMENT ########
### NOTE: a tuple or list value for a namelist parameter
#         indicates the setting is different for each domain

# These are variable
bl_pbl_physics = None 
# YSU (1)
# MYJ (2) <----
# MYNN 2.5 (5) <----
# MYNN 3.0 (6)
# ACM2 (7)
# Shin-Hong (11)

ra_lw_physics = None
# RRTM (1)
# RRTMG (4)

ra_sw_physics = None
# Dudhia (1)
# RRTMG (4)

##### 2016 NEWS-E #####
# mp_physics = 8


# ra_sw_physics
# 1, 4, 1, 4, 1, 4...

# sf_sfclay_physics
# 1, 1, 2, 2, 5, 5...

# bl_pbl_physics
# 1, 1, 2, 2, 5, 5...

# cu_physics = 0

######## Mixed Physics SETTINGS ########
for n in ensnums:
    if n%2: # odd
        NL[n]['ra_sw_physics'] = 1
        NL[n]['ra_lw_physics'] = 1
    else: # even
        NL[n]['ra_sw_physics'] = 4
        NL[n]['ra_lw_physics'] = 4

    # The 600 here is a placeholder for any multiple of 6 that is the
    # number of ensemble members. Breaks if nmems/nens is below 6.
    bl_mems = sorted(x+(6*n) for x in [1,2] for n in range(int(600/6)))

    if n in bl_mems: # member 1,2,7,8...
        NL[n]['bl_pbl_physics'] = 1
        NL[n]['sf_sfclay_physics'] = 1
    elif n in [a+2 for a in bl_mems]: # member 3,4,9,10...
        NL[n]['bl_pbl_physics'] = 2
        NL[n]['sf_sfclay_physics'] = 2
    elif n in [a+4 for a in bl_mems]: # member 5,6,11,12...
        NL[n]['bl_pbl_physics'] = 5
        NL[n]['sf_sfclay_physics'] = 5
    else:
        raise Exception

######## GENERATE NAMELISTS ########
for nidx,n in zip(nidxs,ensnums):
    print("Ensemble run ",n)
    if onlydo and (n != onlydo):
        print("Skipping this one.")
        continue

    # COPY NAMELIST FROM TEMPLATE
    copy_cmd = 'cp {0} {1}'.format(paths_to_nl[n]['old'],paths_to_nl[n]['new'])
    os.system(copy_cmd)

    # EDIT NAMELISTS
    for k,v in NL[n].items():
        # The 'enforce_all' keyword enforces all namelist changes
        # try:
        edit_namelist(paths_to_nl[n]['new'],k,v,enforce_all=enforce_all,doms=doms)
        # except ValueError:
            # print("Check namelist template - it is missing the setting",k)
                # raise Exception
        # else:
            # if debug:
        if enforce_all:
            print("Changed setting",k,"to",v)
    print("Completed editing {0}".format(paths_to_nl[n]['new']))
