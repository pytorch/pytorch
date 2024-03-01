# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx._equalize import (
    _convert_equalization_ref,
    _InputEqualizationObserver,
    _WeightEqualizationObserver,
    calculate_equalization_scale,
    clear_weight_quant_obs_node,
    convert_eq_obs,
    CUSTOM_MODULE_SUPP_LIST,
    custom_module_supports_equalization,
    default_equalization_qconfig,
    EqualizationQConfig,
    fused_module_supports_equalization,
    get_equalization_qconfig_dict,
    get_layer_sqnr_dict,
    get_op_node_and_weight_eq_obs,
    input_equalization_observer,
    is_equalization_observer,
    maybe_get_next_equalization_scale,
    maybe_get_next_input_eq_obs,
    maybe_get_weight_eq_obs_node,
    nn_module_supports_equalization,
    node_supports_equalization,
    remove_node,
    reshape_scale,
    scale_input_observer,
    scale_weight_functional,
    scale_weight_node,
    update_obs_for_equalization,
    weight_equalization_observer,
)
