# Owner(s): ["module: optimizer"]

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


class TestFusedSGDClipping(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("tensor_lr", [False, True])
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
    @torch.no_grad()
    def test_matches_separate_clipping(
        self, device, dtype, tensor_lr, max_norm, options
    ):
        params = [torch.randn(n, device=device, dtype=dtype) for n in [37, 64, 65, 0]]
        reference = [torch.nn.Parameter(p.clone()) for p in params]
        lr = torch.tensor(0.01, device=device) if tensor_lr else 0.01
        optimizer = torch.optim.SGD(reference, lr=lr, fused=True, **options)
        kwargs = dict(
            weight_decay=0.0,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            lr=lr,
        )
        kwargs.update(options)
        buffers = [torch.empty_like(p) for p in params] if kwargs["momentum"] else []
        op = (
            torch.ops.aten._fused_sgd_.grad_clip_tensor_lr
            if tensor_lr
            else torch.ops.aten._fused_sgd_.grad_clip
        )
        for step in range(3):
            grads = [torch.randn_like(p) for p in params]
            raw_grads = [g.clone() for g in grads]
            for p, g in zip(reference, grads):
                p.grad = g.clone()
            norm = torch.nn.utils.get_total_norm(grads)
            coef = (max_norm / (norm + 1e-6)).clamp(max=1).to(torch.float64)
            torch.nn.utils.clip_grad_norm_(reference, max_norm)
            optimizer.step()
            op(
                params,
                grads,
                buffers,
                is_first_step=step == 0 and bool(buffers),
                grad_clip_coef=coef,
                **kwargs,
            )
            for p, q, grad, raw_grad in zip(params, reference, grads, raw_grads):
                self.assertEqual(p, q, rtol=0, atol=0)
                self.assertEqual(grad, raw_grad, rtol=0, atol=0)
            for buffer, p in zip(buffers, reference):
                self.assertEqual(
                    buffer, optimizer.state[p]["momentum_buffer"], rtol=0, atol=0
                )

    @parametrize("invalid", ["dtype", "shape", "device", "grad_scale"])
    def test_invalid_coefficient(self, device, invalid):
        p = torch.ones(9, device=device)
        grad = torch.ones_like(p)
        coef = torch.ones(1, dtype=torch.float64, device=device)
        scale = None
        if invalid == "dtype":
            coef = coef.float()
        elif invalid == "shape":
            coef = coef.expand(2)
        elif invalid == "device":
            coef = coef.cpu()
        else:
            scale = torch.ones(1, device=device)
        message = "Unscale gradients" if invalid == "grad_scale" else "grad_clip_coef"
        with self.assertRaisesRegex(RuntimeError, message):
            torch.ops.aten._fused_sgd_.grad_clip(
                [p],
                [grad],
                [],
                lr=0.1,
                weight_decay=0.0,
                momentum=0.0,
                dampening=0.0,
                nesterov=False,
                maximize=False,
                is_first_step=True,
                grad_clip_coef=coef,
                grad_scale=scale,
            )
        self.assertEqual(p, torch.ones_like(p))
        self.assertEqual(grad, torch.ones_like(grad))

    @parametrize("tensor_lr", [False, True])
    def test_found_inf_skips_update(self, device, tensor_lr):
        p = torch.ones(9, device=device)
        grad = torch.ones_like(p)
        buffer = torch.full_like(p, 2.0)
        lr = torch.tensor(0.1, device=device) if tensor_lr else 0.1
        op = (
            torch.ops.aten._fused_sgd_.grad_clip_tensor_lr
            if tensor_lr
            else torch.ops.aten._fused_sgd_.grad_clip
        )
        op(
            [p],
            [grad],
            [buffer],
            lr=lr,
            weight_decay=0.01,
            momentum=0.9,
            dampening=0.0,
            nesterov=True,
            maximize=False,
            is_first_step=False,
            grad_clip_coef=torch.tensor(0.5, dtype=torch.float64, device=device),
            found_inf=torch.ones(1, device=device),
        )
        self.assertEqual(p, torch.ones_like(p))
        self.assertEqual(grad, torch.ones_like(grad))
        self.assertEqual(buffer, torch.full_like(buffer, 2.0))


instantiate_device_type_tests(TestFusedSGDClipping, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
