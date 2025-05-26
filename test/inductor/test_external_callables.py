# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import TEST_CUDA


class MatMulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.eye(128, 128) * 2, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.matrix)


# torch.add performs better than torch.mm and got choosed during tuning
def matmul_cpu(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_dup(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_cuda(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


class TestInductorExternalCallable(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    def test_matmul_cpu(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_cpu]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cpu}) failed",
        )

    def test_matmul_dup(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        # This should only register the first external call
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_dup, matmul_dup]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_dup}) failed",
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (7, 0),
        "Triton does not support device capability < 7.0",
    )
    def test_matmul_cuda(self):
        device = torch.device("cuda")
        x = (torch.eye(128, 128) * 2).to(device=device)
        opt_fn = torch.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_cuda]},
        )
        opt_fn_golden = torch.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cuda}) failed",
        )


if __name__ == "__main__":
    run_tests()
