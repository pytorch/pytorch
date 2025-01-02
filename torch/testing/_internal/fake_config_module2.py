import sys

from torch.utils._config_module import install_config_module


e_aliasing_bool = False

install_config_module(sys.modules[__name__])
