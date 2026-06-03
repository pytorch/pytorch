"""
test/test_mps_loss_ops.py — correctness tests for Metal loss kernels on MPS.

Run with:
    python test/test_mps_loss_ops.py -v

Each test computes the loss on CPU with float32 and on MPS with the target
dtype, then compares within appropriate tolerances.
"""
import unittest
import torch
import torch.nn.functional as F


def _cpu_ref(fn, *args, **kwargs):
    """Run fn on float32 CPU."""
    cpu_args = [a.detach().float().cpu() if isinstance(a, torch.Tensor) else a for a in args]
    return fn(*cpu_args, **kwargs)


def _mps(t, dtype=torch.float32):
    return t.to(device="mps", dtype=dtype)


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]


class TestMSELoss(unittest.TestCase):
    atol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}

    def _check(self, shape, dtype, reduction):
        x_cpu = torch.randn(shape)
        y_cpu = torch.randn(shape)
        ref = F.mse_loss(x_cpu.float(), y_cpu.float(), reduction=reduction)
        got = F.mse_loss(_mps(x_cpu, dtype), _mps(y_cpu, dtype), reduction=reduction)
        torch.testing.assert_close(
            got.float().cpu(), ref,
            atol=self.atol[dtype], rtol=self.rtol[dtype],
            msg=f"MSE {shape} {dtype} {reduction}",
        )

    def test_float32_mean(self):   self._check((1024,),     torch.float32, "mean")
    def test_float32_sum(self):    self._check((1024,),     torch.float32, "sum")
    def test_float32_none(self):   self._check((64, 64),    torch.float32, "none")
    def test_float16_mean(self):   self._check((1024,),     torch.float16, "mean")
    def test_float16_none(self):   self._check((64, 64),    torch.float16, "none")
    def test_bfloat16_mean(self):  self._check((1024,),     torch.bfloat16, "mean")
    def test_large_mean(self):     self._check((256 * 256,), torch.float32, "mean")

    def test_backward_mean(self):
        x = torch.randn(128, requires_grad=False)
        y = torch.randn(128)
        x_mps = _mps(x).requires_grad_(True)
        y_mps = _mps(y)
        x_cpu = x.clone().requires_grad_(True)
        loss_mps = F.mse_loss(x_mps, y_mps, reduction="mean")
        loss_mps.backward()
        loss_cpu = F.mse_loss(x_cpu, y, reduction="mean")
        loss_cpu.backward()
        torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_backward_none(self):
        x = torch.randn(128, requires_grad=False)
        y = torch.randn(128)
        x_mps = _mps(x).requires_grad_(True)
        y_mps = _mps(y)
        x_cpu = x.clone().requires_grad_(True)
        grad = torch.ones(128)
        loss_mps = F.mse_loss(x_mps, y_mps, reduction="none")
        loss_mps.backward(_mps(grad))
        loss_cpu = F.mse_loss(x_cpu, y, reduction="none")
        loss_cpu.backward(grad)
        torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


class TestSmoothL1Loss(unittest.TestCase):
    def _check(self, shape, dtype, reduction, beta=1.0):
        x = torch.randn(shape)
        y = torch.randn(shape)
        ref = F.smooth_l1_loss(x.float(), y.float(), reduction=reduction, beta=beta)
        got = F.smooth_l1_loss(_mps(x, dtype), _mps(y, dtype), reduction=reduction, beta=beta)
        atol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(got.float().cpu(), ref, atol=atol, rtol=atol,
                                   msg=f"SmoothL1 {shape} {dtype} {reduction}")

    def test_float32_mean(self):   self._check((1024,),   torch.float32, "mean")
    def test_float32_sum(self):    self._check((1024,),   torch.float32, "sum")
    def test_float32_none(self):   self._check((64, 64),  torch.float32, "none")
    def test_float16_mean(self):   self._check((1024,),   torch.float16, "mean")
    def test_bfloat16_mean(self):  self._check((1024,),   torch.bfloat16, "mean")
    def test_beta_small(self):     self._check((512,),    torch.float32, "mean", beta=0.1)
    def test_beta_large(self):     self._check((512,),    torch.float32, "mean", beta=5.0)

    def test_backward(self):
        x = torch.randn(256, requires_grad=False)
        y = torch.randn(256)
        x_mps = _mps(x).requires_grad_(True)
        x_cpu = x.clone().requires_grad_(True)
        F.smooth_l1_loss(x_mps, _mps(y), reduction="mean").backward()
        F.smooth_l1_loss(x_cpu, y, reduction="mean").backward()
        torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


class TestHuberLoss(unittest.TestCase):
    def _check(self, shape, dtype, reduction, delta=1.0):
        x = torch.randn(shape)
        y = torch.randn(shape)
        ref = F.huber_loss(x.float(), y.float(), reduction=reduction, delta=delta)
        got = F.huber_loss(_mps(x, dtype), _mps(y, dtype), reduction=reduction, delta=delta)
        atol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(got.float().cpu(), ref, atol=atol, rtol=atol,
                                   msg=f"Huber {shape} {dtype} {reduction}")

    def test_float32_mean(self):   self._check((1024,),  torch.float32, "mean")
    def test_float32_none(self):   self._check((64, 64), torch.float32, "none")
    def test_float16_mean(self):   self._check((1024,),  torch.float16, "mean")
    def test_bfloat16_sum(self):   self._check((1024,),  torch.bfloat16, "sum")
    def test_delta_small(self):    self._check((512,),   torch.float32, "mean", delta=0.5)

    def test_backward(self):
        x = torch.randn(256, requires_grad=False)
        y = torch.randn(256)
        x_mps = _mps(x).requires_grad_(True)
        x_cpu = x.clone().requires_grad_(True)
        F.huber_loss(x_mps, _mps(y), reduction="mean").backward()
        F.huber_loss(x_cpu, y, reduction="mean").backward()
        torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


class TestBCELoss(unittest.TestCase):
    def _check(self, shape, dtype, reduction, weight=False):
        x = torch.sigmoid(torch.randn(shape))
        y = torch.randint(0, 2, shape).float()
        w = torch.rand(shape) if weight else None
        ref = F.binary_cross_entropy(x.float(), y.float(),
                                      weight=w, reduction=reduction)
        got = F.binary_cross_entropy(
            _mps(x, dtype), _mps(y, dtype),
            weight=_mps(w, dtype) if w is not None else None,
            reduction=reduction,
        )
        atol = 1e-4 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(got.float().cpu(), ref, atol=atol, rtol=atol,
                                   msg=f"BCE {shape} {dtype} {reduction} w={weight}")

    def test_float32_mean(self):       self._check((1024,),  torch.float32, "mean")
    def test_float32_sum(self):        self._check((1024,),  torch.float32, "sum")
    def test_float32_none(self):       self._check((64, 64), torch.float32, "none")
    def test_float16_mean(self):       self._check((1024,),  torch.float16, "mean")
    def test_bfloat16_mean(self):      self._check((1024,),  torch.bfloat16, "mean")
    def test_with_weight_mean(self):   self._check((512,),   torch.float32, "mean", weight=True)
    def test_with_weight_none(self):   self._check((64, 64), torch.float32, "none", weight=True)

    def test_backward_mean(self):
        x = torch.sigmoid(torch.randn(128))
        y = torch.randint(0, 2, (128,)).float()
        x_mps = _mps(x).requires_grad_(True)
        x_cpu = x.clone().requires_grad_(True)
        F.binary_cross_entropy(x_mps, _mps(y), reduction="mean").backward()
        F.binary_cross_entropy(x_cpu, y, reduction="mean").backward()
        torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)


class TestNLLLoss(unittest.TestCase):
    def _check(self, N, C, dtype, reduction, ignore_index=-1):
        x = torch.log_softmax(torch.randn(N, C), dim=1)
        t = torch.randint(0, C, (N,))
        if ignore_index >= 0:
            t[0] = ignore_index  # trigger ignore path
        ref = F.nll_loss(x.float(), t, reduction=reduction, ignore_index=ignore_index)
        got = F.nll_loss(_mps(x, dtype), t.to("mps"),
                          reduction=reduction, ignore_index=ignore_index)
        atol = 1e-4 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(got.float().cpu(), ref, atol=atol, rtol=atol,
                                   msg=f"NLL ({N},{C}) {dtype} {reduction}")

    def test_float32_mean(self):        self._check(256, 10,   torch.float32, "mean")
    def test_float32_sum(self):         self._check(256, 10,   torch.float32, "sum")
    def test_float32_none(self):        self._check(256, 10,   torch.float32, "none")
    def test_float16_mean(self):        self._check(256, 10,   torch.float16, "mean")
    def test_bfloat16_mean(self):       self._check(256, 10,   torch.bfloat16, "mean")
    def test_large_C(self):             self._check(32,  1000, torch.float32, "mean")
    def test_ignore_index(self):        self._check(128, 10,   torch.float32, "mean", ignore_index=5)


class TestCrossEntropyLoss(unittest.TestCase):
    def _check(self, N, C, dtype, reduction):
        x = torch.randn(N, C)
        t = torch.randint(0, C, (N,))
        ref = F.cross_entropy(x.float(), t, reduction=reduction)
        got = F.cross_entropy(_mps(x, dtype), t.to("mps"), reduction=reduction)
        atol = 1e-3 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(got.float().cpu(), ref, atol=atol, rtol=atol,
                                   msg=f"CE ({N},{C}) {dtype} {reduction}")

    def test_float32_mean(self):   self._check(256, 10,   torch.float32, "mean")
    def test_float32_sum(self):    self._check(256, 10,   torch.float32, "sum")
    def test_float32_none(self):   self._check(256, 10,   torch.float32, "none")
    def test_float16_mean(self):   self._check(256, 10,   torch.float16, "mean")
    def test_bfloat16_mean(self):  self._check(256, 10,   torch.bfloat16, "mean")
    def test_large_C(self):        self._check(32,  1000, torch.float32, "mean")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("MPS not available — skipping")
        raise SystemExit(0)
    unittest.main(verbosity=2)
