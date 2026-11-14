# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import HAS_TRITON


class MatMulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.eye(128, 128) * 2, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.matrix)


# torch.add performs better than torch.mm and got chosen during tuning
def matmul_cpu(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_dup(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_accelerator(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


class TestInductorExternalCallableGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

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


class TestInductorExternalCallableAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_matmul_accelerator(self, device):
        x = (torch.eye(128, 128) * 2).to(device=device)
        opt_fn = torch.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_accelerator]},
        )
        opt_fn_golden = torch.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_accelerator}) failed",
        )


instantiate_device_type_tests(
    TestInductorExternalCallableAccelerator,
    globals(),
    except_for="cpu",
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
