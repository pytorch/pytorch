from typing import Any
from ..utils import Pattern  # noqa: F401

# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
QuantizerCls = Any
