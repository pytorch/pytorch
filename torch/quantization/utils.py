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
    activation_dtype,
    activation_is_int8_quantized,
    activation_is_statically_quantized,
    calculate_qmin_qmax,
    check_min_max_valid,
    get_combined_dict,
    get_qconfig_dtypes,
    get_qparam_dict,
    get_quant_type,
    get_swapped_custom_module_class,
    getattr_from_fqn,
    is_per_channel,
    is_per_tensor,
    weight_dtype,
    weight_is_quantized,
    weight_is_statically_quantized,
)
