# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.utils import (
    graph_pretty_str,
    _get_per_tensor_qparams,
    quantize_node,
    _get_custom_module_class_keys,
    _get_linear_prepack_op_for_dtype,
    _get_qconv_prepack_op,
    _get_qconv_op,
    _get_new_attr_name_with_prefix,
    _graph_module_from_producer_nodes,
    _assert_and_get_unique_device,
    _create_getattr_from_value,
    _create_qparam_nodes,
    _all_node_args_have_no_tensors,
    _node_return_type_is_int,
    _get_non_observable_arg_indexes_and_types,
    _is_get_tensor_info_node,
    _maybe_get_next_module
)
