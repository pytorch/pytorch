# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx._equalize import (
    reshape_scale,
    _InputEqualizationObserver,
    _WeightEqualizationObserver,
    calculate_equalization_scale,
    EqualizationQConfig,
    input_equalization_observer,
    weight_equalization_observer,
    default_equalization_qconfig,
    fused_module_supports_equalization,
    nn_module_supports_equalization,
    custom_module_supports_equalization,
    node_supports_equalization,
    is_equalization_observer,
    get_op_node_and_weight_eq_obs,
    maybe_get_weight_eq_obs_node,
    maybe_get_next_input_eq_obs,
    maybe_get_next_equalization_scale,
    scale_input_observer,
    scale_weight_node,
    scale_weight_functional,
    clear_weight_quant_obs_node,
    remove_node,
    update_obs_for_equalization,
    convert_eq_obs,
    _convert_equalization_ref,
    get_layer_sqnr_dict,
    get_equalization_qconfig_dict,
    CUSTOM_MODULE_SUPP_LIST,
)
