import torch

print(torch.__version__)

group_val = 24
ifm = torch.ones([1, group_val, 6, 6], dtype=torch.float32)
weights = torch.ones([group_val, 1, 3, 3], dtype=torch.float32)
op = torch.nn.Conv2d(
        in_channels=group_val,
        out_channels=group_val,
        kernel_size=[3,3],
        stride=[2,2],
        padding=[1,1],
        dilation=[1,1],
        groups=group_val,
        bias=False,
        padding_mode='zeros'
    )

op.weight.data = weights
res = op(ifm)
print(res)
grad_in = torch.ones(res.shape, dtype=torch.float32)
res.backward(grad_in)
print(op.weight.grad[0])