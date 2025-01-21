import sys

from torch.utils._config_module import Config, install_config_module


e_aliasing_bool = False

e_env_default_multi: int = Config(env_name_default=["ENV_ONE", "ENV_TWO"], default=0)
e_env_force_multi: int = Config(env_name_force=["ENV_TWO", "ENV_ONE"], default=0)

install_config_module(sys.modules[__name__])
