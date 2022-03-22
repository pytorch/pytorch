import torch
from torch.fx.graph import (
    Node,
    Graph,
)

from ..utils import (
    get_qconfig_dtypes,
    activation_dtype,
)

from .utils import (
    quantize_node,
)

from .quantization_patterns import (
    QuantizeHandler,
)

from ..qconfig import QConfigAny

from typing import Any, Callable, Dict, Tuple

# TODO: remove
class CommonQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    pass
