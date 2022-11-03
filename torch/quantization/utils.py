# flake8: noqa: F401
r"""
Utils shared by different modes of quantization (eager/graph)

This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/utils.py`, while adding an import statement
here.
"""

from torch.ao.quantization.utils import (
    _activation_dtype,
    _activation_is_int8_quantized,
    _activation_is_statically_quantized,
    _calculate_qmin_qmax,
    _check_min_max_valid,
    _get_combined_dict,
    _get_qconfig_dtypes,
    _get_qparam_dict,
    _get_quant_type,
    _get_swapped_custom_module_class,
    _getattr_from_fqn,
    _is_per_channel,
    _is_per_tensor,
    _weight_dtype,
    _weight_is_quantized,
    _weight_is_statically_quantized,
)
