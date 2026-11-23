# Owner(s): ["module: optimizer"]

import copy

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


class TestSGDClipping(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("backend", ["single", "foreach", "fused"])
    @parametrize("max_norm", [0.0, 1.0, 1e6])
    @parametrize(
        "options",
        [
            {},
            {"momentum": 0.9, "dampening": 0.1, "weight_decay": 0.01},
            {"momentum": 0.9, "nesterov": True},
            {"maximize": True, "weight_decay": 0.01},
        ],
    )
    def test_matches_separate_clipping(self, device, dtype, backend, max_norm, options):
        kwargs = {"fused": backend == "fused", "foreach": backend == "foreach"}
        params = [
            torch.nn.Parameter(torch.randn(n, device=device, dtype=dtype))
            for n in [37, 64, 0, 9]
        ]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        groups = [{"params": params[:2]}, {"params": params[2:], "lr": 0.02}]
        ref_groups = [{"params": reference[:2]}, {"params": reference[2:], "lr": 0.02}]
        opt = torch.optim.SGD(
            groups, lr=0.01, max_grad_norm=max_norm, **kwargs, **options
        )
        ref_opt = torch.optim.SGD(ref_groups, lr=0.01, **kwargs, **options)
        for _ in range(3):
            raw_grads = []
            for p, q in zip(params[:-1], reference[:-1]):
                p.grad = torch.randn_like(p)
                q.grad = p.grad.clone()
                raw_grads.append(p.grad.clone())
            torch.nn.utils.clip_grad_norm_(reference, max_norm)
            ref_opt.step()
            opt.step()
            for p, q in zip(params, reference):
                self.assertEqual(p, q, rtol=0, atol=0)
                if "momentum_buffer" in opt.state[p]:
                    self.assertEqual(
                        opt.state[p]["momentum_buffer"],
                        ref_opt.state[q]["momentum_buffer"],
                        rtol=0,
                        atol=0,
                    )
            for p, grad in zip(params, raw_grads):
                self.assertEqual(p.grad, grad, rtol=0, atol=0)
            self.assertIsNone(params[-1].grad)

    @parametrize("max_norm", [-1.0, float("nan"), float("inf")])
    def test_invalid_threshold(self, device, max_norm):
        p = torch.nn.Parameter(torch.ones(1, device=device))
        with self.assertRaisesRegex(ValueError, "Invalid max_grad_norm"):
            torch.optim.SGD([p], max_grad_norm=max_norm)

    def test_closure_and_state_dict(self, device):
        p = torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device))
        opt = torch.optim.SGD([p], lr=0.1, max_grad_norm=1.0)

        def closure():
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            return loss

        self.assertEqual(opt.step(closure), 25.0)
        self.assertEqual(p.grad, torch.tensor([6.0, 8.0], device=device))
        self.assertEqual(p, torch.tensor([2.94, 3.92], device=device))
        other = torch.optim.SGD([p], max_grad_norm=None)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(other.param_groups[0]["max_grad_norm"], 1.0)
        old_state = copy.deepcopy(opt.state_dict())
        del old_state["param_groups"][0]["max_grad_norm"]
        other.load_state_dict(old_state)
        self.assertIsNone(other.param_groups[0]["max_grad_norm"])

    def test_rejects_inconsistent_groups(self, device):
        params = [torch.nn.Parameter(torch.ones(1, device=device)) for _ in range(2)]
        opt = torch.optim.SGD(
            [{"params": [params[0]]}, {"params": [params[1]], "max_grad_norm": 2.0}],
            max_grad_norm=1.0,
        )
        for p in params:
            p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(ValueError, "same for all parameter groups"):
            opt.step()
        self.assertEqual(params[0], torch.ones_like(params[0]))

    @parametrize("value", [0.0, float("nan"), float("inf")])
    def test_nonfinite_and_zero_gradients(self, device, value):
        p = torch.nn.Parameter(torch.ones(5, device=device))
        q = torch.nn.Parameter(p.detach().clone())
        p.grad = torch.full_like(p, value)
        q.grad = p.grad.clone()
        raw_grad = p.grad.clone()
        opt = torch.optim.SGD(
            [p], lr=0.1, fused=device.startswith("cuda"), max_grad_norm=1.0
        )
        ref = torch.optim.SGD([q], lr=0.1, fused=device.startswith("cuda"))
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        ref.step()
        opt.step()
        self.assertEqual(p, q)
        self.assertEqual(p.grad, raw_grad)

    def test_mixed_dtypes(self, device):
        params = [
            torch.nn.Parameter(torch.ones(17, device=device, dtype=dtype))
            for dtype in (torch.float32, torch.float64, torch.bfloat16)
        ]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        opt = torch.optim.SGD(params, lr=0.1, fused=True, max_grad_norm=1.0)
        ref = torch.optim.SGD(reference, lr=0.1, fused=True)
        for p, q in zip(params, reference):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        raw_grads = [p.grad.clone() for p in params]
        torch.nn.utils.clip_grad_norm_(reference, 1.0)
        ref.step()
        opt.step()
        for p, q, grad in zip(params, reference, raw_grads):
            self.assertEqual(p, q, rtol=0, atol=0)
            self.assertEqual(p.grad, grad, rtol=0, atol=0)

    @parametrize("kind", ["sparse", "complex"])
    def test_rejects_unsupported_gradients(self, device, kind):
        dtype = torch.complex64 if kind == "complex" else torch.float32
        p = torch.nn.Parameter(torch.ones(3, device=device, dtype=dtype))
        p.grad = torch.ones_like(p)
        if kind == "sparse":
            p.grad = p.grad.to_sparse()
        opt = torch.optim.SGD([p], max_grad_norm=1.0)
        with self.assertRaisesRegex(RuntimeError, "dense native real tensors"):
            opt.step()
        self.assertEqual(p, torch.ones_like(p))

    def test_no_gradients(self, device):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        opt = torch.optim.SGD([p], fused=True, max_grad_norm=1.0)
        opt.step()
        self.assertEqual(p, torch.ones_like(p))
        self.assertIsNone(p.grad)

    def test_rejects_differentiable(self, device):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        with self.assertRaisesRegex(ValueError, "does not support differentiable"):
            torch.optim.SGD([p], differentiable=True, max_grad_norm=1.0)

    @onlyCUDA
    @parametrize("tensor_lr", [False, True])
    def test_amp_unscale_and_overflow(self, device, tensor_lr):
        p = torch.nn.Parameter(torch.ones(9, device=device))
        lr = torch.tensor(0.1, device=device) if tensor_lr else 0.1
        opt = torch.optim.SGD([p], lr=lr, fused=True, max_grad_norm=1.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
        scaler.scale(p.sum()).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        self.assertEqual(p, torch.full_like(p, 1.0 - 0.1 / (3.0 + 1e-6)))
        before = p.detach().clone()
        opt.zero_grad()
        scaler.scale(p.sum()).backward()
        p.grad.fill_(float("inf"))
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        self.assertEqual(p, before)

    @onlyCUDA
    def test_amp_requires_unscale(self, device):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        opt = torch.optim.SGD([p], fused=True, max_grad_norm=1.0)
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(p.sum()).backward()
        with self.assertRaisesRegex(RuntimeError, "Call GradScaler.unscale_"):
            scaler.step(opt)
        self.assertEqual(p, torch.ones_like(p))


instantiate_device_type_tests(TestSGDClipping, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
