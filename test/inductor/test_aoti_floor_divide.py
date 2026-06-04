# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestAOTIFloorDivide(TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.version.hip,
        "requires CUDA",
    )
    def test_int64_tensor_constant_divisor(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.divisor = torch.tensor(3, dtype=torch.int64)

            def forward(self, x):
                return torch.floor_divide(x, self.divisor)

        x = torch.tensor([-5, -1, 0, 7, 8], dtype=torch.int64, device="cuda")
        model = Model().eval()
        with torch.no_grad():
            expected = model(x)

        ep = torch.export.export(model, (x,), strict=True)
        package_path = torch._inductor.aoti_compile_and_package(ep)
        actual = torch._inductor.aoti_load_package(package_path)(x)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
