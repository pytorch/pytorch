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


class TestAdamClipping(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("optimizer", ["Adam"])
    @parametrize("amsgrad", [False, True])
    @parametrize("backend", ["single", "foreach", "fused"])
    @parametrize("max_norm", [0.0, 1.0, 1e6])
    @parametrize(
        "options",
        [
            {},
            {"betas": (0.8, 0.95), "weight_decay": 0.01},
            {"maximize": True, "weight_decay": 0.01},
        ],
    )
    def test_matches_separate_clipping(
        self, device, dtype, optimizer, amsgrad, backend, max_norm, options
    ):
        kwargs = {
            "fused": backend == "fused",
            "foreach": backend == "foreach",
            "amsgrad": amsgrad,
        }
        params = [
            torch.nn.Parameter(torch.randn(n, device=device, dtype=dtype))
            for n in [37, 64, 0, 9]
        ]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        groups = [{"params": params[:2]}, {"params": params[2:], "lr": 0.02}]
        ref_groups = [{"params": reference[:2]}, {"params": reference[2:], "lr": 0.02}]
        opt_cls = getattr(torch.optim, optimizer)
        opt = opt_cls(groups, lr=0.01, max_grad_norm=max_norm, **kwargs, **options)
        ref_opt = opt_cls(ref_groups, lr=0.01, **kwargs, **options)
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
                if p in opt.state:
                    for key in opt.state[p]:
                        self.assertEqual(
                            opt.state[p][key], ref_opt.state[q][key], rtol=0, atol=0
                        )
            for p, grad in zip(params, raw_grads):
                self.assertEqual(p.grad, grad, rtol=0, atol=0)
            self.assertIsNone(params[-1].grad)

    @parametrize("optimizer", ["Adam"])
    @parametrize("max_norm", [-1.0, float("nan"), float("inf")])
    def test_invalid_threshold(self, device, optimizer, max_norm):
        p = torch.nn.Parameter(torch.ones(1, device=device))
        with self.assertRaisesRegex(ValueError, "Invalid max_grad_norm"):
            getattr(torch.optim, optimizer)([p], max_grad_norm=max_norm)

    @parametrize("optimizer", ["Adam"])
    def test_closure_and_state_dict(self, device, optimizer):
        p = torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device))
        opt = getattr(torch.optim, optimizer)([p], lr=0.1, max_grad_norm=1.0)

        def closure():
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            return loss

        self.assertEqual(opt.step(closure), 25.0)
        self.assertEqual(p.grad, torch.tensor([6.0, 8.0], device=device))
        q = torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device))
        q.grad = p.grad.clone()
        ref = getattr(torch.optim, optimizer)([q], lr=0.1)
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        ref.step()
        self.assertEqual(p, q)
        other = getattr(torch.optim, optimizer)([p], max_grad_norm=None)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(other.param_groups[0]["max_grad_norm"], 1.0)
        old_state = copy.deepcopy(opt.state_dict())
        del old_state["param_groups"][0]["max_grad_norm"]
        other.load_state_dict(old_state)
        self.assertIsNone(other.param_groups[0]["max_grad_norm"])

    @parametrize("optimizer", ["Adam"])
    def test_rejects_inconsistent_groups(self, device, optimizer):
        params = [torch.nn.Parameter(torch.ones(1, device=device)) for _ in range(2)]
        opt = getattr(torch.optim, optimizer)(
            [{"params": [params[0]]}, {"params": [params[1]], "max_grad_norm": 2.0}],
            max_grad_norm=1.0,
        )
        for p in params:
            p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(ValueError, "same for all parameter groups"):
            opt.step()
        self.assertEqual(params[0], torch.ones_like(params[0]))

    @parametrize("optimizer", ["Adam"])
    @parametrize("value", [0.0, float("nan"), float("inf")])
    def test_nonfinite_and_zero_gradients(self, device, optimizer, value):
        p = torch.nn.Parameter(torch.ones(5, device=device))
        q = torch.nn.Parameter(p.detach().clone())
        p.grad = torch.full_like(p, value)
        q.grad = p.grad.clone()
        raw_grad = p.grad.clone()
        opt = getattr(torch.optim, optimizer)(
            [p], lr=0.1, fused=device.startswith("cuda"), max_grad_norm=1.0
        )
        ref = getattr(torch.optim, optimizer)(
            [q], lr=0.1, fused=device.startswith("cuda")
        )
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        ref.step()
        opt.step()
        self.assertEqual(p, q)
        self.assertEqual(p.grad, raw_grad)

    @parametrize("optimizer", ["Adam"])
    def test_mixed_dtypes(self, device, optimizer):
        params = [
            torch.nn.Parameter(torch.ones(17, device=device, dtype=dtype))
            for dtype in (torch.float32, torch.float64, torch.bfloat16)
        ]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        opt = getattr(torch.optim, optimizer)(
            params, lr=0.1, fused=True, max_grad_norm=1.0
        )
        ref = getattr(torch.optim, optimizer)(reference, lr=0.1, fused=True)
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

    @parametrize("optimizer", ["Adam"])
    @parametrize("kind", ["sparse", "complex"])
    def test_rejects_unsupported_gradients(self, device, optimizer, kind):
        dtype = torch.complex64 if kind == "complex" else torch.float32
        p = torch.nn.Parameter(torch.ones(3, device=device, dtype=dtype))
        p.grad = torch.ones_like(p)
        if kind == "sparse":
            p.grad = p.grad.to_sparse()
        opt = getattr(torch.optim, optimizer)([p], max_grad_norm=1.0)
        with self.assertRaisesRegex(RuntimeError, "dense native real tensors"):
            opt.step()
        self.assertEqual(p, torch.ones_like(p))

    @parametrize("optimizer", ["Adam"])
    def test_no_gradients(self, device, optimizer):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        opt = getattr(torch.optim, optimizer)([p], fused=True, max_grad_norm=1.0)
        opt.step()
        self.assertEqual(p, torch.ones_like(p))
        self.assertIsNone(p.grad)

    @parametrize("optimizer", ["Adam"])
    def test_rejects_differentiable(self, device, optimizer):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        with self.assertRaisesRegex(ValueError, "does not support differentiable"):
            getattr(torch.optim, optimizer)([p], differentiable=True, max_grad_norm=1.0)

    @parametrize("optimizer", ["Adam"])
    @onlyCUDA
    @parametrize("tensor_lr", [False, True])
    def test_amp_unscale_and_overflow(self, device, optimizer, tensor_lr):
        p = torch.nn.Parameter(torch.ones(9, device=device))
        lr = torch.tensor(0.1, device=device) if tensor_lr else 0.1
        opt = getattr(torch.optim, optimizer)([p], lr=lr, fused=True, max_grad_norm=1.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
        scaler.scale(p.sum()).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        q = torch.nn.Parameter(torch.ones_like(p))
        q.grad = torch.ones_like(q)
        ref = getattr(torch.optim, optimizer)([q], lr=lr, fused=True)
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        ref.step()
        self.assertEqual(p, q, rtol=0, atol=0)
        self.assertEqual(p.grad, torch.ones_like(p))
        state_before = copy.deepcopy(opt.state[p])
        before = p.detach().clone()
        opt.zero_grad()
        scaler.scale(p.sum()).backward()
        p.grad.fill_(float("inf"))
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        self.assertEqual(p, before)
        self.assertEqual(opt.state[p], state_before)

    @parametrize("optimizer", ["Adam"])
    @onlyCUDA
    def test_amp_requires_unscale(self, device, optimizer):
        p = torch.nn.Parameter(torch.ones(3, device=device))
        opt = getattr(torch.optim, optimizer)([p], fused=True, max_grad_norm=1.0)
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(p.sum()).backward()
        with self.assertRaisesRegex(RuntimeError, "Call GradScaler.unscale_"):
            scaler.step(opt)
        self.assertEqual(p, torch.ones_like(p))

    @parametrize("optimizer", ["Adam"])
    @onlyCUDA
    @parametrize("amsgrad", [False, True])
    def test_cuda_graph(self, device, optimizer, amsgrad):
        p = torch.nn.Parameter(torch.randn(65, device=device))
        q = torch.nn.Parameter(p.detach().clone())
        raw_grad = torch.randn_like(p)
        p.grad, q.grad = raw_grad.clone(), raw_grad.clone()
        cls = getattr(torch.optim, optimizer)
        opt = cls([p], fused=True, capturable=True, amsgrad=amsgrad, max_grad_norm=1.0)
        ref = cls([q], fused=True, capturable=True, amsgrad=amsgrad)

        def reference_step():
            torch.nn.utils.clip_grad_norm_([q], 1.0)
            ref.step()

        opt.step()
        reference_step()
        p.grad.copy_(raw_grad)
        q.grad.copy_(raw_grad)
        reference_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reference_graph):
            reference_step()
        clipped_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(clipped_graph):
            opt.step()
        for _ in range(3):
            q.grad.copy_(raw_grad)
            reference_graph.replay()
            clipped_graph.replay()
            self.assertEqual(p, q, rtol=0, atol=0)
            self.assertEqual(opt.state[p], ref.state[q], rtol=0, atol=0)
            self.assertEqual(p.grad, raw_grad, rtol=0, atol=0)

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("backend", ["single", "foreach", "fused"])
    def test_decoupled_weight_decay(self, device, dtype, backend):
        p = torch.nn.Parameter(torch.randn(37, device=device, dtype=dtype))
        q = torch.nn.Parameter(p.detach().clone())
        kwargs = dict(
            fused=backend == "fused", foreach=backend == "foreach", weight_decay=0.01
        )
        opt = torch.optim.Adam(
            [p], max_grad_norm=1.0, decoupled_weight_decay=True, **kwargs
        )
        ref = torch.optim.AdamW([q], **kwargs)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
            raw_grad = p.grad.clone()
            torch.nn.utils.clip_grad_norm_([q], 1.0)
            ref.step()
            opt.step()
            self.assertEqual(p, q, rtol=0, atol=0)
            self.assertEqual(opt.state[p], ref.state[q], rtol=0, atol=0)
            self.assertEqual(p.grad, raw_grad, rtol=0, atol=0)


instantiate_device_type_tests(TestAdamClipping, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
