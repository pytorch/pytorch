# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/ns/_numeric_suite.py`, while adding an import statement
here.
"""

from torch.ao.ns._numeric_suite import (
    _convert_tuple_to_list,
    _dequantize_tensor_list,
    _find_match,
    _get_logger_dict_helper,
    _is_identical_module_type,
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
    get_logger_dict,
    get_matching_activations,
    Logger,
    NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
    OutputLogger,
    prepare_model_outputs,
    prepare_model_with_stubs,
    Shadow,
    ShadowLogger,
)
