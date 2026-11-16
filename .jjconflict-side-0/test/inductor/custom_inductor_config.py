# Owner(s): ["module: inductor"]

# This module is used in test_codecache.py to verify the correctness
# of FXGraphHashDetails when a custom inductor backend registers its own
# config object

import sys

from torch.utils._config_module import install_config_module


enable_optimisation: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
