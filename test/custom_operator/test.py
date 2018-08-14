import torch

torch.ops.load_library('custom_op.so')

output = torch.ops.custom.op(torch.ones(5), 2.0, 3)
assert type(output) == list
assert len(output) == 3
assert all(tensor.allclose(torch.ones(5) * 2) for tensor in output)
