# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/qconfig.py`, while adding an import statement
here.
"""
from torch.ao.quantization.qconfig import (
    _add_module_to_qconfig_obs_ctr,
    _assert_valid_qconfig,
    default_activation_only_qconfig,
    default_debug_qconfig,
    default_dynamic_qconfig,
    default_per_channel_qconfig,
    default_qat_qconfig,
    default_qat_qconfig_v2,
    default_qconfig,
    default_weight_only_qconfig,
    float16_dynamic_qconfig,
    float16_static_qconfig,
    float_qparams_weight_only_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig,
    per_channel_dynamic_qconfig,
    QConfig,
    qconfig_equals,
    QConfigAny,
    QConfigDynamic,
)
