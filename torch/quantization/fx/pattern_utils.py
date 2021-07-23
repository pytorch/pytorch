import torch
from collections import OrderedDict
from typing import Dict, Any, Tuple, List, Optional
from torch.fx.graph import (
    Node,
)
from .quantization_types import Pattern
from ..qconfig import QConfigAny
# from .quantization_patterns import BinaryOpQuantizeHandler


# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
QuantizeHandler = Any

MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                    QConfigAny]

# pattern for conv bn fusion
DEFAULT_FUSION_PATTERNS = OrderedDict()
def register_fusion_pattern(pattern):
    def insert(fn):
        DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert

def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    return DEFAULT_FUSION_PATTERNS

DEFAULT_QUANTIZATION_PATTERNS = OrderedDict()
# a map from pattern to activation_post_process(observer/fake_quant) consstructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_affine_fixed_qparam_fake_quant
DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP = dict()

# Register pattern for both static quantization and qat
def register_quant_pattern(pattern, output_activation_post_process=None):
    def insert(fn):
        DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if output_activation_post_process is not None:
            DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP[pattern] = output_activation_post_process
        return fn
    return insert

# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    return DEFAULT_QUANTIZATION_PATTERNS

# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map() -> Dict[Pattern, torch.quantization.observer.ObserverBase]:
    return DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP


# Example use of register pattern function:
# @register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
# class ConvBNReLUFusion():
#     def __init__(...):
#         ...
#
