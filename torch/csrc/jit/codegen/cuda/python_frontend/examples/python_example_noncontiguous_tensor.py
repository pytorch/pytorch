import torch

from torch._C._nvfuser import Fusion, FusionDefinition

# Construct and Define Fusion
fusion = Fusion()

with FusionDefinition(fusion) as fd :
    t0 = fd.define_tensor(3, [False, False, False])
    t1 = fd.define_tensor(3, [True, True, True])

    fd.add_input(t0)
    fd.add_input(t1)
    print("Input1 Contiguity:", t0)
    print("Input2 Contiguity:", t1)

    t2 = fd.Ops.add(t0, t1)

    print("Output Contiguity:", t2, "\n")
    fd.add_output(t2)

fusion.print_ir()
fusion.print_kernel()

# Execute Fusion
input1 = torch.Tensor([4, 3, 2, 1]).cuda().unsqueeze(0).unsqueeze(-1)
input1 = (input1 + torch.zeros(2, 4, 3, device='cuda')).transpose(1, 2)
input2 = torch.Tensor([1, 2, 3, 4]).cuda().unsqueeze(0).unsqueeze(0)
input2 = input2 + torch.zeros(2, 3, 4, device='cuda')

# Kernel compilation should be cached for the 2nd iteration
# with input tensors of the same shape
for _ in range(5) :
    outputs = fusion.execute([input1, input2])

print("\nInput1 Size and Stride:", input1.size(), input1.stride())
print(input1)
print("Input2 Size and Stride:", input2.size(), input2.stride())
print(input2)
print("Output Size and Stride:", outputs[0].size(), outputs[0].stride())
print("Output Tensor is expected to have all 5's when contiguity is correctly specified.\n", outputs[0])

assert outputs[0].equal(torch.full((2, 3, 4), 5.0, device='cuda')), "Contiguity is not properly specified!"
