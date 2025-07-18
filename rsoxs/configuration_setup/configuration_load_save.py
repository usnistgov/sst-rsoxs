## New spreadsheet loader and saver

import os
import numpy as np
import pandas as pd
import ast
import copy
import json
import datetime
import re, warnings, httpx
import uuid


from nbs_bl.devices.sampleholders import SampleHolderBase
from nbs_bl.hw import manipulator

from ..plans.default_energy_parameters import energyListParameters

from .configuration_load_save_sanitize import (
    load_configuration_spreadsheet_local, 
    save_configuration_spreadsheet_local,
    get_sample_dictionary_nbs_format_from_rsoxs_config,
)
from ..redis_config import rsoxs_config



def sync_rsoxs_config_to_nbs_manipulator():
    """
    Converts metadata from rsoxs_config["bar"] to format used by nbs-bl.
    Then updates maniuplator sample list.
    Intended to be run anywhere rsoxs_config["bar"] is updated.
    TODO: this function needs to be run manually anytime rsoxs_config["bar"] is updated manually.
    """

    samples_dictionary_nbs_format = get_sample_dictionary_nbs_format_from_rsoxs_config(configuration=copy.deepcopy(rsoxs_config["bar"]))
    manipulator.load_sample_dict(samples_dictionary_nbs_format)



def load_sheet(file_path):
    """
    Loads spreadsheet and updates sample configuration in RSoXS control computer.
    """    

    ## Update rsoxs_config, used in rsoxs codebase
    configuration = load_configuration_spreadsheet_local(file_path=file_path)
    rsoxs_config["bar"] = copy.deepcopy(configuration)
    print("Replaced persistent configuration with configuration loaded from file path: " + str(file_path))

    sync_rsoxs_config_to_nbs_manipulator()
    
    
    return

def save_sheet(file_path, file_label):
    ## Test comment + more comment
    save_configuration_spreadsheet_local(configuration=rsoxs_config["bar"], file_path=file_path, file_label=file_label)
    return








