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


class TestFusedAdamClipping(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("optimizer", ["Adam", "AdamW"])
    @parametrize("lr_kind", ["float", "cuda_tensor", "cpu_tensor"])
    @parametrize("amsgrad", [False, True])
    @parametrize("max_norm", [0.0, 1.0, 1e6])
    @parametrize("maximize", [False, True])
    @torch.no_grad()
    def test_matches_separate_clipping(
        self, device, dtype, optimizer, lr_kind, amsgrad, max_norm, maximize
    ):
        params = [torch.randn(n, device=device, dtype=dtype) for n in [37, 64, 65, 0]]
        params.append(torch.randn(3, 5, device=device, dtype=dtype).t())
        reference = [torch.nn.Parameter(p.clone()) for p in params]
        lr = (
            0.01
            if lr_kind == "float"
            else torch.tensor(
                0.01, device=device if lr_kind == "cuda_tensor" else "cpu"
            )
        )
        ref_opt = getattr(torch.optim, optimizer)(
            reference,
            lr=lr,
            fused=True,
            amsgrad=amsgrad,
            weight_decay=0.01,
            maximize=maximize,
        )
        averages = [torch.zeros_like(p) for p in params]
        squares = [torch.zeros_like(p) for p in params]
        maxima = [torch.zeros_like(p) for p in params] if amsgrad else []
        steps = [torch.zeros((), device=device) for _ in params]
        packet = getattr(
            torch.ops.aten, "_fused_adam_" if optimizer == "Adam" else "_fused_adamw_"
        )
        op = packet.grad_clip if lr_kind == "float" else packet.grad_clip_tensor_lr
        for _ in range(3):
            grads = [torch.randn_like(p) for p in params]
            raw_grads = [g.clone() for g in grads]
            for p, g in zip(reference, grads):
                p.grad = g.clone()
            norm = torch.nn.utils.get_total_norm(grads)
            coef = (max_norm / (norm + 1e-6)).clamp(max=1).to(torch.float64)
            torch.nn.utils.clip_grad_norm_(reference, max_norm)
            ref_opt.step()
            torch._foreach_add_(steps, 1)
            op(
                params,
                grads,
                averages,
                squares,
                maxima,
                steps,
                lr=lr,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=amsgrad,
                maximize=maximize,
                grad_clip_coef=coef,
            )
            for i, (p, q, grad, raw_grad) in enumerate(
                zip(params, reference, grads, raw_grads)
            ):
                self.assertEqual(p, q, rtol=0, atol=0)
                self.assertEqual(grad, raw_grad, rtol=0, atol=0)
                self.assertEqual(
                    averages[i], ref_opt.state[q]["exp_avg"], rtol=0, atol=0
                )
                self.assertEqual(
                    squares[i], ref_opt.state[q]["exp_avg_sq"], rtol=0, atol=0
                )
                self.assertEqual(steps[i], ref_opt.state[q]["step"], rtol=0, atol=0)
                if amsgrad:
                    self.assertEqual(
                        maxima[i], ref_opt.state[q]["max_exp_avg_sq"], rtol=0, atol=0
                    )

    @parametrize("optimizer", ["Adam", "AdamW"])
    @parametrize("amsgrad", [False, True])
    @torch.no_grad()
    def test_mixed_precision_states(self, device, optimizer, amsgrad):
        p = torch.randn(65, device=device)
        q = torch.nn.Parameter(p.clone())
        avg = torch.zeros_like(p, dtype=torch.bfloat16)
        square = torch.zeros_like(avg)
        maximum = torch.zeros_like(avg)
        step = torch.zeros((), device=device)
        ref = getattr(torch.optim, optimizer)(
            [q], fused=True, amsgrad=amsgrad, weight_decay=0.01
        )
        ref.state[q] = {
            "step": step.clone(),
            "exp_avg": avg.clone(),
            "exp_avg_sq": square.clone(),
        }
        if amsgrad:
            ref.state[q]["max_exp_avg_sq"] = maximum.clone()
        packet = getattr(
            torch.ops.aten, "_fused_adam_" if optimizer == "Adam" else "_fused_adamw_"
        )
        for _ in range(3):
            grad = torch.randn_like(p)
            raw_grad = grad.clone()
            q.grad = grad.clone()
            coef = (
                (1 / (torch.nn.utils.get_total_norm([grad]) + 1e-6))
                .clamp(max=1)
                .double()
            )
            torch.nn.utils.clip_grad_norm_([q], 1.0)
            ref.step()
            step.add_(1)
            packet.grad_clip(
                [p],
                [grad],
                [avg],
                [square],
                [maximum] if amsgrad else [],
                [step],
                lr=0.001,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=amsgrad,
                maximize=False,
                grad_clip_coef=coef,
            )
            self.assertEqual(p, q, rtol=0, atol=0)
            self.assertEqual(grad, raw_grad, rtol=0, atol=0)
            self.assertEqual(avg, ref.state[q]["exp_avg"], rtol=0, atol=0)
            self.assertEqual(square, ref.state[q]["exp_avg_sq"], rtol=0, atol=0)
            if amsgrad:
                self.assertEqual(
                    maximum, ref.state[q]["max_exp_avg_sq"], rtol=0, atol=0
                )

    @parametrize("optimizer", ["Adam", "AdamW"])
    @parametrize("invalid", ["dtype", "shape", "device", "grad_scale"])
    def test_invalid_coefficient(self, device, optimizer, invalid):
        p = torch.ones(9, device=device)
        grad = torch.ones_like(p)
        avg, square = torch.zeros_like(p), torch.zeros_like(p)
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
        packet = getattr(
            torch.ops.aten, "_fused_adam_" if optimizer == "Adam" else "_fused_adamw_"
        )
        message = "Unscale gradients" if invalid == "grad_scale" else "grad_clip_coef"
        with self.assertRaisesRegex(RuntimeError, message):
            packet.grad_clip(
                [p],
                [grad],
                [avg],
                [square],
                [],
                [torch.ones((), device=device)],
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.0,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
                grad_clip_coef=coef,
                grad_scale=scale,
            )
        self.assertEqual(p, torch.ones_like(p))
        self.assertEqual(grad, torch.ones_like(grad))
        self.assertEqual(avg, torch.zeros_like(avg))
        self.assertEqual(square, torch.zeros_like(square))

    @parametrize("optimizer", ["Adam", "AdamW"])
    @parametrize("tensor_lr", [False, True])
    @parametrize("amsgrad", [False, True])
    def test_found_inf_skips_update(self, device, optimizer, tensor_lr, amsgrad):
        p = torch.ones(9, device=device)
        grad = torch.ones_like(p)
        avg, square, maximum = [torch.full_like(p, 2.0) for _ in range(3)]
        packet = getattr(
            torch.ops.aten, "_fused_adam_" if optimizer == "Adam" else "_fused_adamw_"
        )
        op = packet.grad_clip_tensor_lr if tensor_lr else packet.grad_clip
        op(
            [p],
            [grad],
            [avg],
            [square],
            [maximum] if amsgrad else [],
            [torch.ones((), device=device)],
            lr=torch.tensor(0.1, device=device) if tensor_lr else 0.1,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=amsgrad,
            maximize=False,
            grad_clip_coef=torch.tensor(0.5, dtype=torch.float64, device=device),
            found_inf=torch.ones(1, device=device),
        )
        self.assertEqual(p, torch.ones_like(p))
        self.assertEqual(grad, torch.ones_like(grad))
        for state in [avg, square, maximum]:
            self.assertEqual(state, torch.full_like(state, 2.0))


instantiate_device_type_tests(TestFusedAdamClipping, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
