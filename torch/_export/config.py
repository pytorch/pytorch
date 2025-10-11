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

# this flag is used to control whether we want to instrument
# fake tensor creation to track potential leaks. It is off
# by default, but user can turn it on to debug leaks.
detect_non_strict_fake_tensor_leaks = False

# error on potentially pre-dispatch/non-strict tracing limitation
# this type of error usually happens when we encounter an op
# that we don't know how to proxy, resulting in untracked fake tensors
error_on_lifted_constant_tensors = True

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

    def _make_closure_patcher(**changes: Any) -> Any: ...


install_config_module(sys.modules[__name__])
