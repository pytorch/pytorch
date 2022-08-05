import torch

from torch._C._nvfuser import Fusion, FusionDefinition, DataType

# Construct and Define Fusion
fusion = Fusion()

with FusionDefinition(fusion) as fd :
    t0 = fd.define_tensor(3, DataType.Half)
    t1 = fd.define_tensor(1, DataType.Half)
    s0 = fd.define_scalar()

    c0 = fd.define_constant(3.0)

    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.mul(t3, s0)
    t5 = fd.ops.relu(t4)
    t6 = fd.ops.sum(t5, [-1], False, DataType.Float)

    t7 = fd.ops.cast(t6, DataType.Half)
    fd.add_output(t7)

fusion.print_ir()

# Execute Fusion
input1 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)
input2 = torch.ones(8, device='cuda', dtype=torch.float16)

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fusion.execute([input1, input2, 2.0])

print(outputs[0])
