""" Run ensembles for Vortex-SE project."""
import pdb
import os
from pathlib import PosixPath
import datetime
import time

import numpy as N

from evac.lazy.lazyensemble import LazyEnsemble
import evac.utils.utils as utils

### Type "vsedone" to see list of cases already ran. ###

CASES = {
        datetime.datetime(2016,3,31,0,0,0) : [
            datetime.datetime(2016,3,31,19,0,0),
            datetime.datetime(2016,3,31,20,0,0),
            datetime.datetime(2016,3,31,21,0,0),
            datetime.datetime(2016,3,31,22,0,0),
            datetime.datetime(2016,3,31,23,0,0),
            ],

        datetime.datetime(2017,5,1,0,0,0) : [
            datetime.datetime(2017,5,1,19,0,0),
            datetime.datetime(2017,5,1,20,0,0),
            datetime.datetime(2017,5,1,21,0,0),
            datetime.datetime(2017,5,1,22,0,0),
            datetime.datetime(2017,5,1,23,0,0),
            ],

        datetime.datetime(2017,5,2,0,0,0) : [
            datetime.datetime(2017,5,3,0,0,0),
            datetime.datetime(2017,5,3,1,0,0),
            datetime.datetime(2017,5,3,2,0,0),
            datetime.datetime(2017,5,3,3,0,0),
            datetime.datetime(2017,5,3,4,0,0),
            ],

        datetime.datetime(2017,5,4,0,0,0) : [
            datetime.datetime(2017,5,4,22,0,0),
            datetime.datetime(2017,5,4,23,0,0),
            datetime.datetime(2017,5,5,0,0,0),
            datetime.datetime(2017,5,5,1,0,0),
            datetime.datetime(2017,5,5,2,0,0),
            ],
        }

key = 'ForReal_nco'
check_before = False

for caseutc, initutcs in CASES.items():
    for initutc in initutcs:

        # Skip if already created

        timestr = utils.string_from_time('dir',initutc)
        rootdir = PosixPath('/scratch/john.lawson/WRF/VSE_reso/{}/{}/'.format(key,timestr))

        if os.path.exists(str(rootdir)):
            print("Skipping this case; already ran.")
            continue

        print("Running WRF for",initutc)

        wrfdir = PosixPath('/scratch/john.lawson/WRF/VSE_reso/my_build/WRFV3/run')
        # wrfdir = PosixPath('/scratch/software/Odin/WRF/Intel/WRFV4/WRF/run/')
        casestr = utils.string_from_time('dir',caseutc,strlen='day')
        initstr = utils.string_from_time('dir',initutc,strlen='hour')
        wofdir = PosixPath('/work/wof/realtime/{}'.format(casestr))
        rundir = rootdir

        runsec = utils.hr_to_sec(3)

        nmems = 36
        # nmems = 4

        newse_mems =  ['mem{}'.format(n) for n in range(1,nmems+1)]
        membernames = ['m{:02d}'.format(n) for n in range(1,nmems+1)]
        memnos = N.arange(1,nmems+1)

        ### GENERATE NAMELISTS ###
        # This is ugly as sin
        otherdir = PosixPath('/home/john.lawson/VSE_reso/otherfiles')
        namelistdir = PosixPath('/home/john.lawson/VSE_reso/pycode/namelists/{}/{}'.format(key,casestr))
        og_nl_path = wofdir / 'mem1' / 'namelist.input'

        utils.trycreate(namelistdir,isdir=True)

        # new_nl_path = namelistdir / 'namelist.input.og_template'
        new_nl_path = namelistdir / 'namelist.input.template'
        if not new_nl_path.exists():
            utils.bridge('copy',og_nl_path,new_nl_path)

        import subprocess
        try:
            cmd = ("python generate_namelists.py --donotenforce --nmems {}"
                                    " --key {} --initstr {}"
                                    " --overwrite --casestr {}".format(nmems,key,initstr,casestr))
            subprocess.check_call(cmd.split())
        except:
            utils.wowprint("There's a **problem**!")
            raise
        else:
            print("Namelists generated.")
        # pdb.set_trace()
        # utils.trycreate(namelistdir,is_dir=True)

        # Dictionary for each member
        # {mem:{(from,to),command)}}
        icdict = {}
        lbcdict = {}
        icbc_cmd = 'copy'  
        for newse_mem,mem,memno in zip(newse_mems,membernames,memnos):
            wrfout_needed = utils.string_from_time('wrfout',initutc,dom=1) + '_{}'.format(memno)
            path_to_ic = wofdir / 'WRFOUT' / wrfout_needed
            path_to_newinput = rundir / mem / 'wrfinput_d01'
            icdict[mem] = {}
            icdict[mem][(path_to_ic,path_to_newinput)] = icbc_cmd
            LBCdir = wofdir / newse_mem
            lbcdict[mem] = {}
            for wrfbdy in LBCdir.glob('wrfbdy*'):
                lbcdict[mem][(wrfbdy,rundir/mem)] = icbc_cmd

        init_kwargs = dict(
            path_to_exedir = wrfdir ,
            path_to_datadir = rundir,
            path_to_namelistdir = namelistdir,
            # path_to_icbcdir = rootdir/'test_icbcdir',
            path_to_icdir = False,
            path_to_lbcdir = False,
            path_to_outdir = rootdir/'outdir',
            path_to_batch = otherdir/'run_wrf.job',
            initutc = initutc,
            ndoms = 2,
            membernames = membernames,
            # endutc = datetime.datetime(2016,4,1,0,0,0),
            runsec = runsec,
            nl_per_member = 'dot_number',
            #nl_suffix = 'dot_number'
            ics = icdict,
            lbcs = lbcdict,
            dryrun = False,
            )

        # Note Odin is 'double threaded processes' so 48 
        #   CPUs will appears in squeue as 96.

        run_kwargs = dict(
                            # cpus=48,
                            cpus=24,
                            # nodes=2,
                            nodes=1,
                            # first=1,
                            merge_lbcs=True,
                            )

        ### PREREQUISITES ###
        # config = 'newse'
        config = 'default'
        if config == 'newse':
            pres = ['wrf.exe' ,  'qr_acr_qg.bin' ,
                    'freezeH2O.bin' , 'qr_acr_qs.bin' ,
                    'LANDUSE.TBL', 'GENPARM.TBL', 'SOILPARM.TBL', 'ETAMPNEW_DATA',
                    'gribmap.txt', 'RRTM_DATA', 'RRTMG_LW_DATA', 'RRTMG_SW_DATA',
                    'tr49t67', 'tr49t85', 'tr67t85', 'VEGPARM.TBL',
                    # NEW
                    'ozone_plev.formatted','ozone_lat.formatted','ozone.formatted']
        elif config == 'default':
            pres = ['wrf.exe' , 
                    'LANDUSE.TBL', 'GENPARM.TBL', 'SOILPARM.TBL', 'ETAMPNEW_DATA',
                    'gribmap.txt', 'RRTM_DATA', 'RRTMG_LW_DATA', 'RRTMG_SW_DATA',
                    'tr49t67', 'tr49t85', 'tr67t85', 'VEGPARM.TBL',
                    # NEW
                    'ozone_plev.formatted','ozone_lat.formatted','ozone.formatted']
        PRES = {wrfdir/p:'copy' for p in pres}

        # To test, let's use the same namelist for every member
        # But a different set of initial conditions

        ### PROCEDURE ###
        L = LazyEnsemble(**init_kwargs)
        # L.run_all_members(PRES,check_domains=True,**run_kwargs)
        # pdb.set_trace()
        if check_before:
            print('===== MAKE SURE TEMPLATE IS FINISHED =====')
            pdb.set_trace()
            print('===== MAKE SURE TEMPLATE IS FINISHED =====')
        L.run_all_members(PRES,**run_kwargs)

        sleepsec = 3*60*60
        print("Now sleeping for {} min".format(sleepsec/60))
        time.sleep(sleepsec)
