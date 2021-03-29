from typing import Union, Callable, Tuple, Any

Pattern = Union[Callable, Tuple[Callable, Callable], Tuple[Callable, Callable, Callable]]

# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
QuantizerCls = Any
