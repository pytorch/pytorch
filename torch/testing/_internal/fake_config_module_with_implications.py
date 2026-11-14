import sys
from typing import Literal

from torch.utils._config_module import Config, install_config_module


mode: Literal["default", "strict"] = Config(
    default="default",
    implies={"strict": {"enabled": True, "nested.enabled": True}},
)
enabled = Config(default=False, implies={True: {"downstream": 1}})
downstream = 0
conflict = Config(default=False, implies={True: {"enabled": False}})
unrelated = 1


class nested:
    enabled = False


install_config_module(sys.modules[__name__])
