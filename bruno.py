import torch
import torch.nn as nn


dtype = torch.float
dtype = torch.float64

m = nn.Conv3d(2, 2, kernel_size=3, groups=2).to("cuda", dtype)
# get output value
i = torch.randn(2, 2, 6, 6, 6, device="cuda", dtype=dtype).div_(2).requires_grad_()
output = m(i)
print("Testing 3dCONV")

print(dtype)
# get gradient
grad_output = torch.randn(2, 2, 4, 4, 4, device="cuda", dtype=dtype) / 2
output.backward(grad_output)

# generate two pointwise convolutions with kernel
# weights and biases as slices of the dephtwise op and make sure
# that the concatenated outputs are the same
m1 = nn.Conv3d(1, 1, kernel_size=3).to("cuda", dtype)
m1.weight.data = m.weight.data[:1].clone()
m1.bias.data = m.bias.data[:1].clone()
i1 = i.detach()[:, :1].clone().requires_grad_()
output1 = m1(i1)
output1.backward(grad_output[:, :1].contiguous())

m2 = nn.Conv3d(1, 1, kernel_size=3).to("cuda", dtype)
m2.weight.data.copy_(m.weight.data[1:])
m2.bias.data.copy_(m.bias.data[1:])
i2 = i.detach()[:, 1:].clone().requires_grad_()
output2 = m2(i2)
output2.backward(grad_output[:, 1:].contiguous())

# print(output, torch.cat([output1, output2], 1))