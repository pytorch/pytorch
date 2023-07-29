# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.pattern_utils import (
    _register_fusion_pattern,
    _register_quant_pattern,
    get_default_fusion_patterns,
    get_default_output_activation_post_process_map,
    get_default_quant_patterns,
    QuantizeHandler,
)

# QuantizeHandler.__module__ = _NAMESPACE
_register_fusion_pattern.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_fusion_patterns.__module__ = "torch.ao.quantization.fx.pattern_utils"
_register_quant_pattern.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_quant_patterns.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_output_activation_post_process_map.__module__ = (
    "torch.ao.quantization.fx.pattern_utils"
)

# __all__ = [
#     "QuantizeHandler",
#     "_register_fusion_pattern",
#     "get_default_fusion_patterns",
#     "_register_quant_pattern",
#     "get_default_quant_patterns",
#     "get_default_output_activation_post_process_map",
# ]
