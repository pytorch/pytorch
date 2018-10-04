import platform
import glob
import os
import sys

from itertools import chain
from .env import check_env_flag


USE_MKLDNN = False
if not check_env_flag('NO_MKLDNN'):
    USE_MKLDNN = True
