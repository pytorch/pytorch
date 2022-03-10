import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

torch.manual_seed(42)

device = 'lazy'
dtype = torch.float32

x = torch.randn(2, 3, 4, device=device, dtype=dtype)
y = torch.randn(2, 3, 4, device=device, dtype=dtype)
z = torch.randn(2, 1, 1, device=device, dtype=dtype)
t = torch.randn(2, 3, 4, device=device, dtype=dtype)

print((x / y + z))
print(x.type_as(t))
print(x.relu())
print(x.sign())
print((x <= y))
print(x.reciprocal())
print(x.sigmoid())
print(x.sinh())
print(torch.where(x <= y, z, t))
print(torch.addcmul(x, y, z, value=0.1))
print(torch.remainder(x, y))

print(metrics.metrics_report())
