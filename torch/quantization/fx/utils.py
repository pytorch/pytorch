# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.utils import (
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    create_getattr_from_value,
    get_custom_module_class_keys,
    get_linear_prepack_op_for_dtype,
    get_new_attr_name_with_prefix,
    get_non_observable_arg_indexes_and_types,
    get_qconv_prepack_op,
    graph_module_from_producer_nodes,
    maybe_get_next_module,
)
