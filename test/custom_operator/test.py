import os
import torch

library_path = os.path.abspath('build/libcustom_op.so')
torch.ops.load_library(library_path)
assert torch.ops.loaded_libraries == {library_path}

output = torch.ops.custom.op(torch.ones(5), 2.0, 3)
assert type(output) == list
assert len(output) == 3
assert all(tensor.allclose(torch.ones(5) * 2) for tensor in output)
print('success')
