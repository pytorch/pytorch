from torch._ops import ops
from . import _qscheme as _QScheme

relu = ops.quantized.relu
sum_relu = ops.quantized.add_relu
