import torch
from torch.testing import assert_close

from torch.testing._internal.common_utils import TestCase


class TestAdafactorMixedDtype(TestCase):
    def test_default_eps_is_per_dtype(self):
        for foreach in (False, True):
            mixed32 = torch.tensor([1.0, 0.5], dtype=torch.float32)
            mixed64 = torch.tensor([1.0, 0.5], dtype=torch.float64)
            ref32 = mixed32.clone()
            ref64 = mixed64.clone()

            mixed32.requires_grad_()
            mixed64.requires_grad_()
            ref32.requires_grad_()
            ref64.requires_grad_()

            mixed = torch.optim.Adafactor(
                [mixed32, mixed64], lr=1e-2, foreach=foreach
            )
            opt32 = torch.optim.Adafactor([ref32], lr=1e-2, foreach=foreach)
            opt64 = torch.optim.Adafactor([ref64], lr=1e-2, foreach=foreach)

            mixed32.grad = torch.full_like(mixed32, 1e-5)
            mixed64.grad = torch.full_like(mixed64, 1e-5)
            ref32.grad = mixed32.grad.clone()
            ref64.grad = mixed64.grad.clone()

            mixed.step()
            opt32.step()
            opt64.step()

            assert_close(mixed32, ref32)
            assert_close(mixed64, ref64)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
