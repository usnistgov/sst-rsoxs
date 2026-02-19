import numpy as np
import copy
import datetime

import bluesky.plan_stubs as bps

from nbs_bl.plans.scans import nbs_count, nbs_list_scan, nbs_energy_scan
from nbs_bl.beamline import GLOBAL_BEAMLINE as bl
from nbs_bl.hw import (
    en,
    mir1,
    fs1_cam,
    fs6_cam,
    mirror2,
    grating,
    mir3,
    fs7_cam,
    slitsc,
    slits1,
    izero_y,
    slits2,
    slits3,
    manipulator,
    sam_Th,
    #waxs_det,
    #Det_W,
    fs13_cam,
    dm7_y,
)


from rsoxs.configuration_setup.configurations_instrument import load_configuration
from rsoxs.Functions.alignment import (
    #load_configuration, 
    load_samp, 
    rotate_now
    )
from rsoxs.HW.energy import set_polarization
from ..alignment.m3 import *
from ..alignment.energy_calibration import *




