from cycler import cycler
from itertools import chain
from bluesky.utils import Msg, short_uid as _short_uid
from bluesky.protocols import Readable, Flyable
import bluesky.utils as utils
from bluesky.plan_stubs import trigger_and_read, move_per_step, stage, unstage
import bluesky.plans as bp
import bluesky.plan_stubs as bps
from bluesky.plan_stubs import (
    checkpoint,
    abs_set,
    sleep,
    trigger,
    read,
    wait,
    create,
    save,
    unstage,
)
from bluesky.preprocessors import finalize_decorator
from bluesky.preprocessors import rewindable_wrapper, finalize_wrapper
from bluesky.utils import short_uid, separate_devices, all_safe_rewind
from collections import defaultdict
from bluesky import preprocessors as bpp
from bluesky import FailedStatus
import numpy as np
import redis_json_dict
from functools import partial
from ophyd import Device, Signal
from ophyd.status import StatusTimeoutError
import warnings
from copy import deepcopy
from nbs_bl.beamline import GLOBAL_BEAMLINE as bl
from nbs_bl.hw import (
    en,
    mir3,
    izero_mesh,
    shutter_open_time,
    shutter_control,
    shutter_enable,
    #shutter_open_set
    sam_X,
    sam_Y,
    sam_Z,
    sam_Th,
    beamstop_waxs,
    #waxs_det,
    #Beamstop_SAXS,
    #saxs_det,
    DiodeRange,
    Sample_TEY, 
    #Beamstop_SAXS_int,
    beamstop_waxs,
    Sample_TEY_int, 
    ring_current,
)
from ..HW.energy import (
    mono_en,
    epu_gap,
    grating_to_250,
    grating_to_rsoxs,
    grating_to_1200,
    set_polarization,
    #Mono_Scan_Speed_ev,
    #Mono_Scan_Start,
    #Mono_Scan_Start_ev,
    #Mono_Scan_Stop,
    #Mono_Scan_Stop_ev,
    get_gap_offset,
)
from ..HW.signals import (
    default_sigs,
)
from ..HW.lakeshore import tem_tempstage
from ..Functions.alignment import rotate_now
#from ..Functions.common_procedures import set_exposure


from nbs_bl.printing import run_report
from nbs_bl.plans.scans import nbs_list_scan, nbs_gscan
from nbs_bl.utils import merge_func

from ..startup import RE
from bluesky.utils import ensure_generator, short_uid as _short_uid, single_gen
from bluesky.preprocessors import plan_mutator


from .per_steps import (
    take_exposure_corrected_reading,
    one_nd_sticky_exp_step
)


run_report(__file__)

@merge_func(nbs_gscan, use_func_name=False, omit_params=["motor"])
def variable_energy_scan(*args, **kwargs):
    yield from bps.mv(shutter_control, 1)
    yield from finalize_wrapper(
        plan = nbs_gscan(en.energy, *args, **kwargs),
        final_plan= post_scan_hardware_reset()
    )

## TODO: Is this being used in rsoxsCustom.py?
@merge_func(variable_energy_scan, use_func_name=False, omit_params=['per_step'])
def rsoxs_step_scan(*args, extra_dets=[], n_exposures=1, **kwargs):
    """
    Step scanned RSoXS function with WAXS Detector

    Parameters
    ----------
    n_exposures : int, optional
        If greater than 1, take multiple exposures per step
    """
    waxs_det = bl["waxs_det"]

    old_n_exp = waxs_det.number_exposures
    waxs_det.number_exposures = n_exposures
    _extra_dets = [waxs_det]
    _extra_dets.extend(extra_dets)
    rsoxs_per_step=partial(one_nd_sticky_exp_step,
                    take_reading=partial(take_exposure_corrected_reading,
                                        shutter = shutter_control,
                                        check_exposure=False))
    yield from variable_energy_scan(*args, extra_dets=_extra_dets, per_step=rsoxs_per_step, **kwargs)
    waxs_det.number_exposures = old_n_exp




def post_scan_hardware_reset():
    ## Make sure the shutter is closed, and the scanlock if off after a scan, even if it errors out
    yield from bps.mv(en.scanlock, 0)
    yield from bps.mv(shutter_control, 0)

## Everything below is Eliot's old code.  Above is new scans.

