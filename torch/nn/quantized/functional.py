from torch._ops import ops
from torch._C import _add_docstr

relu = ops.quantized.relu
sum_relu = ops.quantized.sum_relu
