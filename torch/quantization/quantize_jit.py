# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize_jit.py`, while adding an import statement
here.
"""

from torch.ao.quantization.quantize_jit import (
    _check_forward_method,
    _check_is_script_module,
    _convert_jit,
    _prepare_jit,
    _prepare_ondevice_dynamic_jit,
    _quantize_jit,
    convert_dynamic_jit,
    convert_jit,
    fuse_conv_bn_jit,
    prepare_dynamic_jit,
    prepare_jit,
    quantize_dynamic_jit,
    quantize_jit,
    script_qconfig,
    script_qconfig_dict,
)
