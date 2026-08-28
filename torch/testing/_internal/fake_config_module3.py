import sys
from typing import Callable  # noqa: UP035

from torch.utils._config_module import install_config_module


e_list = [1]
e_set = {1}
e_func: Callable | None = None

install_config_module(sys.modules[__name__])
