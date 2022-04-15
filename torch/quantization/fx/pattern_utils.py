# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.pattern_utils import (
    QuantizeHandler,
    MatchResult,
    register_fusion_pattern,
    get_default_fusion_patterns,
    register_quant_pattern,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map
)

_NAMESPACE = 'torch.quantization.fx.pattern_utils'
# QuantizeHandler.__module__ = _NAMESPACE
MatchResult.__module__ = _NAMESPACE
register_fusion_pattern.__module__ = _NAMESPACE
get_default_fusion_patterns.__module__ = _NAMESPACE
register_quant_pattern.__module__ = _NAMESPACE
get_default_quant_patterns.__module__ = _NAMESPACE
get_default_output_activation_post_process_map.__module__ = _NAMESPACE
