# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize.py`, while adding an import statement
here.
"""

from torch.ao.quantization.quantize import (
    _add_observer_,
    _convert,
    _get_observer_dict,
    _get_unique_devices_,
    _is_activation_post_process,
    _observer_forward_hook,
    _propagate_qconfig_helper,
    _register_activation_post_process_hook,
    _remove_activation_post_process,
    _remove_qconfig,
    add_quant_dequant,
    convert,
    prepare,
    prepare_qat,
    propagate_qconfig_,
    quantize,
    quantize_dynamic,
    quantize_qat,
    swap_module,
)
