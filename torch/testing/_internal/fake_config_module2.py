import sys

from torch.utils._config_module import Config, install_config_module


e_aliasing_bool = False

e_env_default_multi: bool = Config(
    env_name_default=["ENV_TRUE", "ENV_FALSE"], default=False
)
e_env_force_multi: bool = Config(env_name_force=["ENV_FAKE", "ENV_TRUE"], default=False)

install_config_module(sys.modules[__name__])
