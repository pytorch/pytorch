import torch

# All this does is load the dynamic library which runs the global op registration code.
import custom_op

output = torch.ops.custom.op(torch.ones(5), 2.0, 3)
assert type(output) == list
assert len(output) == 3
assert all([tensor.allclose(torch.ones(5) * 2) for tensor in output])
