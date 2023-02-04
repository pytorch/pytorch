# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/observer.py`, while adding an import statement
here.
"""
import sys
import warnings

from torch.ao.quantization import observer as __orig_mod
from torch.utils._migration_utils import (
    _get_ao_migration_warning_str,
    _AO_MIGRATION_DEPRECATED_NAME_PREFIX
)

_deprecated_names = [
    "_PartialWrapper",
    "_with_args",
    "_with_callable_args",
    "ABC",
    "ObserverBase",
    "_ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "HistogramObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "NoopObserver",
    "_is_activation_post_process",
    "_is_per_channel_script_obs_instance",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "default_observer",
    "default_placeholder_observer",
    "default_debug_observer",
    "default_weight_observer",
    "default_histogram_observer",
    "default_per_channel_weight_observer",
    "default_dynamic_quant_observer",
    "default_float_qparams_observer",
]

for orig_name in _deprecated_names:
    target_obj_name = f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{orig_name}"
    target_obj = getattr(__orig_mod, orig_name)
    setattr(sys.modules[__name__], target_obj_name, target_obj)
del target_obj_name
del target_obj

def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(_get_ao_migration_warning_str(__name__, name))
        return globals()[f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{name}"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
