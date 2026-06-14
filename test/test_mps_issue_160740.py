"""Reproduce https://github.com/pytorch/pytorch/issues/160740

argmax/argmin fail on non-contiguous (transposed, strided) MPS tensors
with: "view size is not compatible with input tensor's size and stride"
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestArgmaxArgminNonContiguous(TestCase):
    def test_transposed(self):
        x = torch.randn(5, 5, device="mps")
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x.t()).item(), torch.argmax(x_cpu.t()).item())
        self.assertEqual(torch.argmin(x.t()).item(), torch.argmin(x_cpu.t()).item())

    def test_strided(self):
        x = torch.randn(5, 5, device="mps")
        x_cpu = x.cpu()
        self.assertEqual(
            torch.argmax(x[::2, ::2]).item(), torch.argmax(x_cpu[::2, ::2]).item()
        )
        self.assertEqual(
            torch.argmin(x[::2, ::2]).item(), torch.argmin(x_cpu[::2, ::2]).item()
        )

    def test_with_dim_on_transposed(self):
        x = torch.randn(5, 5, device="mps")
        x_cpu = x.cpu()
        self.assertEqual(
            torch.argmax(x.t(), dim=0).cpu(), torch.argmax(x_cpu.t(), dim=0)
        )
        self.assertEqual(
            torch.argmax(x.t(), dim=1).cpu(), torch.argmax(x_cpu.t(), dim=1)
        )
        self.assertEqual(
            torch.argmin(x.t(), dim=0).cpu(), torch.argmin(x_cpu.t(), dim=0)
        )
        self.assertEqual(
            torch.argmin(x.t(), dim=1).cpu(), torch.argmin(x_cpu.t(), dim=1)
        )

    def test_large_sliced(self):
        y = torch.randn(100, 100, device="mps")
        y_cpu = y.cpu()
        col_slice = y[:, ::3]
        col_slice_cpu = y_cpu[:, ::3]
        self.assertEqual(
            torch.argmax(col_slice).item(), torch.argmax(col_slice_cpu).item()
        )
        self.assertEqual(
            torch.argmin(col_slice).item(), torch.argmin(col_slice_cpu).item()
        )

    def test_3d_permuted(self):
        z = torch.randn(4, 8, 16, device="mps")
        z_cpu = z.cpu()
        z_nc = z.permute(2, 0, 1)
        z_nc_cpu = z_cpu.permute(2, 0, 1)
        self.assertEqual(torch.argmax(z_nc).item(), torch.argmax(z_nc_cpu).item())
        self.assertEqual(torch.argmin(z_nc).item(), torch.argmin(z_nc_cpu).item())


if __name__ == "__main__":
    run_tests()
