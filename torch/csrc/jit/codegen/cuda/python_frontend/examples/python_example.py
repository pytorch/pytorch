import torch

from torch._C._nvfuser import Fusion, FusionDefinition

# Construct and Define Fusion
fusion = Fusion()

with FusionDefinition(fusion) as fd :
    t0 = fd.define_tensor(3)
    t1 = fd.define_tensor(1)
    s0 = fd.define_scalar()

    fd.add_input(t0)
    fd.add_input(t1)
    fd.add_input(s0)

    c0 = fd.define_constant(3.0)

    t1_b = fd.Ops.broadcast(t1, [True, True, False])
    t2 = fd.Ops.add(t0, t1)
    t3 = fd.Ops.mul(t2, c0)
    t4 = fd.Ops.mul(t3, s0)
    t5 = fd.Ops.relu(t4)
    t6 = fd.Ops.sum(t5, [-1], False)

    fd.add_output(t6)

fusion.print_ir()

# Execute Fusion
input1 = torch.ones(2, 4, 8, device='cuda')
input2 = torch.ones(8, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fusion.execute([input1, input2, 2.0])

print(outputs[0])
