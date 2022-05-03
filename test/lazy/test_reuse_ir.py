# Owner(s): ["oncall: jit"]

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ts_backend
import torch._lazy.metrics as metrics

torch._lazy.ts_backend.init()
torch._lazy.config.set_reuse_ir(True)

def testAddSub():
    device = 'cuda'
    x = torch.randn(2, 3, 4, device=device)
    y = torch.randn(2, 3, 4, device=device)
    z = torch.zeros(2, 3, 4, device=device)

    device = 'lazy'
    x_lazy = x.detach().clone().to(device=device)
    y_lazy = y.detach().clone().to(device=device)
    z_lazy = z.detach().clone().to(device=device)

    for i in range(10):
        if i < 5:
            z += (x + y)
        else:
            z += (x - y)

    for i in range(10):
        if i < 5:
            z_lazy += (x_lazy + y_lazy)
        else:
            z_lazy += (x_lazy - y_lazy)
        torch._lazy.mark_step()

    torch.testing.assert_close(z.cpu(), z_lazy.cpu())
    # print({name: metrics.counter_value(name) for name in metrics.counter_names()})

if __name__ == '__main__':
    testAddSub()
