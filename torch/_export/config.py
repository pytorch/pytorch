"""
Configuration module for torch.export.export.

This module contains various configuration flags and settings that control torch.export's
behavior, including:
- Runtime behavior flags
- Debugging and development options
"""

import sys
from typing import Any, TYPE_CHECKING

from torch.utils._config_module import install_config_module


# this flag controls whether we use new functional tracer. It
# should be True in the long term.
use_new_tracer_experimental = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

    def _make_closure_patcher(**changes: Any) -> Any: ...


install_config_module(sys.modules[__name__])
