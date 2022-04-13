import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._ltc_enable_reuse_ir()

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
        ltm.mark_step()

    torch.testing.assert_close(z.cpu(), z_lazy.cpu())
    print(metrics.metrics_report())

if __name__ == '__main__':
    testAddSub()
