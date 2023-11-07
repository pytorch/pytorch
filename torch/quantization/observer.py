# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/observer.py`, while adding an import statement
here.
"""
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    _is_per_channel_script_obs_instance,
    _ObserverBase,
    _PartialWrapper,
    _with_args,
    _with_callable_args,
    ABC,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_histogram_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_weight_observer,
    get_observer_state_dict,
    HistogramObserver,
    load_observer_state_dict,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    RecordingObserver,
)
