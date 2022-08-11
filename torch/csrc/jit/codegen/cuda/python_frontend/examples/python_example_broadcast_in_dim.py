import torch

from torch._C._nvfuser import FusionManager, FusionDefinition

# Construct and Define Fusion
fm1 = FusionManager.get()

with FusionDefinition(fm1) as fd :
    t0 = fd.define_tensor(1)
    t1 = fd.define_tensor(3)

    t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
    t2 = fd.ops.add(t0_b, t1)

    fd.add_output(t2)

fm1.print_ir()

# Execute Fusion
input1 = torch.ones(3, device='cuda')
input2 = torch.ones(2, 3, 4, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fm1.execute([input1, input2])

print(outputs[0])

fm2 = FusionManager.get()

input1 = torch.ones(1, 1, 4, device='cuda')
input2 = torch.ones(2, 3, 4, device='cuda')

with FusionDefinition(fm2) as fd :
    t0 = fd.define_tensor(sizes=input1.size(), strides=input1.stride())
    t1 = fd.define_tensor(sizes=input2.size(), strides=input2.stride())

    t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [0, 1, 2])
    t2 = fd.ops.add(t0_b, t1)

    fd.add_output(t2)

fm2.print_ir()

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fm2.execute([input1, input2])

print(outputs[0])
