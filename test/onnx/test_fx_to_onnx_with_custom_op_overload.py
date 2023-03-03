import torch
from torch.library import Library
from torch.fx.experimental.proxy_tensor import make_fx
import onnxscript

from torch.onnx._internal import diagnostics, fx
import torch.onnx._internal.fx.custom_operator

@fx.custom_operator.custom_op_overload("custom::op(Tensor x) -> Tensor")
def custom_op(x):
    return x * 2 + 1

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.custom", version=1)
@onnxscript.script(opset=CUSTOM_OPSET)
def custom_op_exporter(x):
    return CUSTOM_OPSET.my_op(x)

fx.custom_operator._register_custom_op_overload(torch.ops.custom.op.default, "custom_op", custom_op_exporter)

def f(x):
    return custom_op(x) * 3

onnx_model = fx.export_after_normalizing_args_and_kwargs(
    f, torch.randn(3), use_binary_format=False
)
print(onnx_model)
