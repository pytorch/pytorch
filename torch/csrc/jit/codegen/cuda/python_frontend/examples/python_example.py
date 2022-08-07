import torch
from torch._C._nvfuser import FusionManager, FusionDefinition, DataType

# Construct and Define Fusion
fmanager = FusionManager()

with FusionDefinition(fmanager) as fd :
    t0 = fd.define_tensor(3)
    t1 = fd.define_tensor(3)
    s0 = fd.define_scalar()

    t2 = fd.ops.add(t0, t1)
    
    c0 = fd.define_constant(3.0)

    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
    t5 = fd.ops.isfinite(t4)

    fd.add_output(t4)
    fd.add_output(t5)

fmanager.print_ir()

# Execute Fusion
input1 = torch.ones(2, 4, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fmanager.execute([input1, input2, 2.0])

print(outputs[0])
print(outputs[1])

with FusionDefinition(fmanager) as fd :
    t0 = fd.define_tensor(3)
    t1 = fd.define_tensor(3)
    s0 = fd.define_scalar()
    
    t2 = fd.ops.add(t0, t1)

    c0 = fd.define_constant(3.0)
    
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
    t5 = fd.ops.isfinite(t4)

    fd.add_output(t4)
    fd.add_output(t5)

input3 = torch.ones(2, 4, 8, device='cuda')
input4 = torch.ones(2, 4, 8, device='cuda')

for _ in range(5) :
    outputs = fmanager.execute([input3, input4, 2.0])

print(outputs[0])
print(outputs[1])
