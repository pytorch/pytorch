from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import Quantizer
from .fuse import Fuser
from .custom_module_class import (
    register_observed_custom_module_mapping,
    register_quantized_custom_module_mapping,
)
