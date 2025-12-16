# Owner(s): ["module: nn"]


import torch
import unittest
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestLinearCrossEntropyCPU(TestCase):
    def test_all_targets_ignored(self) -> None:
        torch.manual_seed(0)
        input = torch.randn(512, 16)
        weight = torch.randn(128, 16)
        bias = torch.randn(128)
        target = torch.full((512,), -1, dtype=torch.long)

        loss_mean = F.linear_cross_entropy(
            input,
            weight,
            target,
            linear_bias=bias,
            reduction="mean",
            ignore_index=-1,
            chunking_strategy="batch",
        )
        self.assertTrue(torch.isnan(loss_mean).item())

        loss_sum = F.linear_cross_entropy(
            input,
            weight,
            target,
            linear_bias=bias,
            reduction="sum",
            ignore_index=-1,
            chunking_strategy="batch",
        )
        self.assertEqual(loss_sum.item(), 0.0)

        loss_none = F.linear_cross_entropy(
            input,
            weight,
            target,
            linear_bias=bias,
            reduction="none",
            ignore_index=-1,
            chunking_strategy="batch",
        )
        self.assertTrue(torch.eq(loss_none, 0).all().item())

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
        with self.assertRaisesRegex(ValueError, "vocab_chunk_size"):
            F.linear_cross_entropy(
                x,
                w,
                t,
                chunking_strategy="vocab",
                vocab_chunk_size=0,
            )
        with self.assertRaisesRegex(ValueError, "batch_chunk_size"):
            F.linear_cross_entropy(
                x,
                w,
                t,
                chunking_strategy="batch",
                batch_chunk_size=-5,
            )

    @skipIfTorchDynamo("gradcheck graph not yet supported under TorchDynamo")
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
                linear_bias=b,
                reduction="mean",
                chunking_strategy="vocab",
            )

        self.assertTrue(torch.autograd.gradcheck(func, (input, weight, bias)))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestLinearCrossEntropyCUDA(TestCase):
    def test_cuda_matches_cpu(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        input = torch.randn(4, 8, device=device)
        weight = torch.randn(16, 8, device=device)
        bias = torch.randn(16, device=device)
        target = torch.randint(0, 16, (4,), device=device)

        def run(strategy: str) -> torch.Tensor:
            return F.linear_cross_entropy(
                input,
                weight,
                target,
                linear_bias=bias,
                reduction="mean",
                chunking_strategy=strategy,
                vocab_chunk_size=8,
                batch_chunk_size=2,
            )

        cpu_input = input.cpu()
        cpu_weight = weight.cpu()
        cpu_bias = bias.cpu()
        cpu_target = target.cpu()

        for strategy in ("none", "vocab", "batch"):
            loss_cuda = run(strategy)
            loss_cpu = F.linear_cross_entropy(
                cpu_input,
                cpu_weight,
                cpu_target,
                linear_bias=cpu_bias,
                reduction="mean",
                chunking_strategy=strategy,
                vocab_chunk_size=8,
                batch_chunk_size=2,
            )
            self.assertTrue(torch.allclose(loss_cuda.cpu(), loss_cpu, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    run_tests()
