from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.quantization_types import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
# from .quantization_patterns import BinaryOpQuantizeHandler
from ..observer import ObserverBase
import copy

# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
QuantizeHandler = Any

# pattern for conv bn fusion
DEFAULT_FUSION_PATTERNS = OrderedDict()
def register_fusion_pattern(pattern):
    def insert(fn):
        DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert

def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    return copy.copy(DEFAULT_FUSION_PATTERNS)

DEFAULT_QUANTIZATION_PATTERNS = OrderedDict()

# Mapping from pattern to activation_post_process(observer/fake_quant) constructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_fixed_qparams_range_0to1_fake_quant
DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP = {}
DEFAULT_OUTPUT_OBSERVER_MAP = {}

# Register pattern for both static quantization and qat
def register_quant_pattern(pattern, fixed_qparams_observer=None):
    def insert(fn):
        DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn
    return insert

# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    return copy.copy(DEFAULT_QUANTIZATION_PATTERNS)

# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map(is_training) -> Dict[Pattern, ObserverBase]:
    if is_training:
        return copy.copy(DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(DEFAULT_OUTPUT_OBSERVER_MAP)

# Example use of register pattern function:
# @register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
# class ConvOrLinearBNReLUFusion():
#     def __init__(...):
#         ...
#

def sorted_patterns_dict(patterns_dict: Dict[Pattern, QuantizeHandler]) -> Dict[Pattern, QuantizeHandler]:
    """
    Return a sorted version of the patterns dictionary such that longer patterns are matched first,
    e.g. match (F.relu, F.linear) before F.relu.
    This works for current use cases, but we may need to have a more clever way to sort
    things to address more complex patterns
    """

    def get_len(pattern):
        """ this will calculate the length of the pattern by counting all the entries
        in the pattern.
        this will make sure (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) comes before
        (nn.BatchNorm, nn.Conv2d) so that we can match the former first
        """
        len = 0
        if isinstance(pattern, tuple):
            for item in pattern:
                len += get_len(item)
        else:
            len += 1
        return len

    return OrderedDict(sorted(patterns_dict.items(), key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1))
