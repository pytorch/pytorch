import os
import glob

from .env import IS_WINDOWS, IS_CONDA, CONDA_DIR, check_env_flag, gather_paths
from .rocm import USE_ROCM, ROCM_HOME


USE_MIOPEN = False
MIOPEN_LIB_DIR = None
MIOPEN_INCLUDE_DIR = None
MIOPEN_LIBRARY = None
if USE_ROCM and not check_env_flag('NO_MIOPEN'):
    USE_MIOPEN = True
    MIOPEN_LIB_DIR = ROCM_HOME + "/miopen/lib"
    MIOPEN_INCLUDE_DIR = ROCM_HOME + "/miopen/include/miopen"
    MIOPEN_LIBRARY = "MIOpen"
    MIOPEN_FOUND = "True"
