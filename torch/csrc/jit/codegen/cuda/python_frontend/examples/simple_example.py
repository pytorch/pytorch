import torch
from torch import ones
from torch._C._nvfuser import Fusion, FusionDefinition, DataType

# Construct and Define Fusion
fusion = Fusion()

with FusionDefinition(fusion) as fd :
    t0 = fd.define_tensor(sizes=[5], strides=[1])
    t1 = fd.ops.abs(t0)
    fd.add_output(t1)

fusion.print_ir()

# Execute Fusion
#input1 = ones(5, device='cuda')
input1 = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fusion.execute([input1])

print(outputs[0])
