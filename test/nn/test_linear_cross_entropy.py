# Owner(s): ["module: nn"]

import itertools
from typing import Optional

import torch
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase, run_tests


def _reference_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor],
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    logits = F.linear(input, weight, bias)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.view(-1)
    loss = F.cross_entropy(
        logits_flat,
        target_flat,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    if reduction == "none":
        loss = loss.view(target.shape)
    return loss


class TestLinearCrossEntropyCPU(TestCase):
    def _compare_with_reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        chunking_strategy: str = "auto",
    ) -> None:
        input_clone = input.clone().requires_grad_(input.requires_grad)
        weight_clone = weight.clone().requires_grad_(weight.requires_grad)
        bias_clone = None
        if bias is not None:
            bias_clone = bias.clone().requires_grad_(bias.requires_grad)

        fused = F.linear_cross_entropy(
            input,
            weight,
            target,
            bias,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            chunking_strategy=chunking_strategy,
        )
        ref = _reference_linear_cross_entropy(
            input_clone,
            weight_clone,
            target,
            bias_clone,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        if fused.requires_grad:
            grad_args = [tensor for tensor in (input, weight, bias) if tensor is not None]
            grad_args_ref = [tensor for tensor in (input_clone, weight_clone, bias_clone) if tensor is not None]

            if reduction == "none":
                grad_output = torch.ones_like(fused)
                fused_grads = torch.autograd.grad(
                    fused,
                    grad_args,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    allow_unused=True,
                )
                ref_grads = torch.autograd.grad(
                    ref,
                    grad_args_ref,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    allow_unused=True,
                )
            else:
                fused_grads = torch.autograd.grad(
                    fused,
                    grad_args,
                    retain_graph=False,
                    allow_unused=True,
                )
                ref_grads = torch.autograd.grad(
                    ref,
                    grad_args_ref,
                    retain_graph=False,
                    allow_unused=True,
                )

            for grad_fused, grad_ref, tensor in zip(fused_grads, ref_grads, grad_args):
                if grad_fused is None or grad_ref is None:
                    self.assertTrue(grad_fused is None and grad_ref is None)
                else:
                    self.assertEqual(grad_fused, grad_ref)

        if reduction == "none":
            self.assertEqual(fused.shape, target.shape)
            self.assertEqual(ref.shape, target.shape)
        self.assertEqual(fused, ref)

    def test_forward_backward_matches_reference_auto(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(2, 3, 32, requires_grad=True)
        weight = torch.randn(6000, 32, requires_grad=True)
        bias = torch.randn(6000, requires_grad=True)
        target = torch.randint(0, 6000, (2, 3))
        self._compare_with_reference(input, weight, target, bias, chunking_strategy="auto")

    def test_vocab_chunking(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(4, 16, requires_grad=True)
        weight = torch.randn(5000, 16, requires_grad=True)
        target = torch.randint(0, 5000, (4,))
        self._compare_with_reference(input, weight, target, None, chunking_strategy="vocab")

    def test_batch_chunking(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(1500, 8, requires_grad=True)
        weight = torch.randn(64, 8, requires_grad=True)
        target = torch.randint(0, 64, (1500,))
        self._compare_with_reference(input, weight, target, None, chunking_strategy="batch")

    def test_reduction_and_options(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(3, 4, 8, requires_grad=True)
        weight = torch.randn(16, 8, requires_grad=True)
        bias = torch.randn(16, requires_grad=True)
        target = torch.randint(0, 16, (3, 4))

        for reduction, label_smoothing in itertools.product(["none", "sum", "mean"], [0.0, 0.2]):
            self._compare_with_reference(
                input.clone().requires_grad_(),
                weight.clone().requires_grad_(),
                target,
                bias.clone().requires_grad_(),
                reduction=reduction,
                label_smoothing=label_smoothing,
                ignore_index=-1,
            )

    def test_parameter_validation(self) -> None:
        x = torch.randn(2, 4)
        w = torch.randn(8, 4)
        t = torch.randint(0, 8, (2,))

        with self.assertRaisesRegex(ValueError, "reduction"):
            F.linear_cross_entropy(x, w, t, reduction="invalid")
        with self.assertRaisesRegex(ValueError, "label_smoothing"):
            F.linear_cross_entropy(x, w, t, label_smoothing=-0.1)
        with self.assertRaisesRegex(ValueError, "label_smoothing"):
            F.linear_cross_entropy(x, w, t, label_smoothing=1.1)
        with self.assertRaisesRegex(ValueError, "chunking_strategy"):
            F.linear_cross_entropy(x, w, t, chunking_strategy="other")

    def test_gradcheck(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
        weight = torch.randn(20, 5, dtype=torch.double, requires_grad=True)
        bias = torch.randn(20, dtype=torch.double, requires_grad=True)
        target = torch.randint(0, 20, (3,), dtype=torch.long)

        def func(inp, wgt, b):
            return F.linear_cross_entropy(
                inp,
                wgt,
                target,
                b,
                reduction="mean",
                chunking_strategy="vocab",
            )

        self.assertTrue(torch.autograd.gradcheck(func, (input, weight, bias)))


if __name__ == "__main__":
    run_tests()
