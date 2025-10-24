import sys
from typing import Callable, Optional

from torch.utils._config_module import install_config_module


e_list = [1]
e_set = {1}
e_func: Optional[Callable] = None

install_config_module(sys.modules[__name__])
