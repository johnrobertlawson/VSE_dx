import os
import pdb

from objectID import ObjectID

# rootdir = '/work/john.lawson/VSE_reso/objectID_3km/'
rootdir = '/work/john.lawson/VSE_reso/objectID_1km/'

kwargs = dict(
        wrfdir = '/scratch/john.lawson/WRF/VSE_reso/ForReal_nco/2017050501',
        sumdir = os.path.join(rootdir,'summaryfiles'),
        objdir = os.path.join(rootdir,'objectfiles'),
        # mrmsdir = '/work/john.lawson/MRMS_data',
        mrmsdir = '/work1/skinnerp/MRMS_verif/mrms_cressman/20170504',
        fcst_nt = int(180/5) + 1,
        # JRL TODO: shouldn't need the slash
        plotdir = os.path.join(rootdir,'plots/'),
        ne = 36,
        # ndoms = 1,
        doms = (2,),
        # doms = (1,),
        datestr = '20170504',
        ncpus = 30,
        do_all = False,
        # do_ens = False,
        # do_env = False,
        # do_swt = False,
        # do_sum = False,
        )

oID = ObjectID(**kwargs)
oID.wrfout_to_summary()
oID.plot_summaries()
oID.create_rot_qc()
# oID.match_object_fields()
# oID.plot_stats()
# oID.extract_attributes()
