
import torch

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.specialize_float = False

import torch
torch._dynamo.config.capture_scalar_outputs = True
@torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)
def f(x, y):
    return x + y
# f(torch.randn(3), torch.tensor(3.0))
# f(torch.randn(3), torch.tensor(4.0))
f(torch.randn(3), 3.0)
f(torch.randn(3), 4.0)
