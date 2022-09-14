import torch

from torch._C._nvfuser import Fusion, FusionDefinition
import torch._prims as prims
import torch._refs as refs

# Construct and Define Fusion
fusion1 = Fusion()

with FusionDefinition(fusion1) as fd :
    t0 = fd.define_tensor(1)
    t1 = fd.define_tensor(3)

    t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
    t2 = fd.ops.add(t0_b, t1)

    fd.add_output(t2)

fusion1.print_ir()

# Execute Fusion
input1 = torch.randn(3, device='cuda')
input2 = torch.randn(2, 3, 4, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    o = fusion1.execute([input1, input2])[0]

assert(o.shape == torch.Size([2, 3, 4]))

# Reference in prim torch
ref_o = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [1]), input2)
assert(ref_o.allclose(o))
assert(ref_o.shape == o.shape)

fusion2 = Fusion()

input1 = torch.randn(1, 1, 4, device='cuda')
input2 = torch.randn(2, 3, 4, device='cuda')

with FusionDefinition(fusion2) as fd :
    t0 = fd.define_tensor(sizes=input1.size(), strides=input1.stride())
    t1 = fd.define_tensor(sizes=input2.size(), strides=input2.stride())

    t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [0, 1, 2])
    t2 = fd.ops.add(t0_b, t1)

    fd.add_output(t2)

fusion2.print_ir()

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    o = fusion2.execute([input1, input2])[0]

assert(o.shape == torch.Size([2, 3, 4]))

# Reference in prim torch
ref_o = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [0, 1, 2]), input2)
assert(ref_o.allclose(o))
assert(ref_o.shape == o.shape)

# Construct and Define Fusion
fusion3 = Fusion()

with FusionDefinition(fusion3) as fd :
    # t0 = fd.define_tensor(2)
    t0 = fd.define_tensor([3, 1], [1, 1])
    t1 = fd.define_tensor(1)

    t1_b = fd.ops.broadcast_in_dim(t1, [3, 3], [0])  # 1 -> 0
    t2 = fd.ops.add(t0, t1_b)

    fd.add_output(t2)

fusion3.print_ir()

# Execute Fusion
input1 = torch.randn(3, 1, device='cuda')
input2 = torch.randn(3, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    o = fusion3.execute([input1, input2])[0]

assert(o.shape == torch.Size([3, 3]))

# Reference in prim torch
ref_o = refs.add(input1, prims.broadcast_in_dim(input2, [3, 3], [0]))
assert(ref_o.allclose(o))
assert(ref_o.shape == o.shape)
