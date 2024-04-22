# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantization_mappings.py`, while adding an import statement
here.
"""
from torch.ao.quantization.quantization_mappings import (
    _get_special_act_post_process,
    _has_special_act_post_process,
    _INCLUDE_QCONFIG_PROPAGATE_LIST,
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
    DEFAULT_MODULE_TO_ACT_POST_PROCESS,
    DEFAULT_QAT_MODULE_MAPPINGS,
    DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
    get_default_compare_output_module_list,
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
    get_default_static_quant_module_mappings,
    get_dynamic_quant_module_class,
    get_quantized_operator,
    get_static_quant_module_class,
    no_observer_set,
)
