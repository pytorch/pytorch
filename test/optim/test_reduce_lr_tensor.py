import torch

from torch.testing._internal.common_utils import TestCase, run_tests


class TestReduceLROnPlateauTensorLR(TestCase):
    def test_tensor_lr_stays_tensor(self):
        param = torch.nn.Parameter(torch.ones(1))
        lr = torch.tensor([0.1])
        optimizer = torch.optim.Adam([param], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=0, min_lr=0.001
        )

        scheduler.step(1.0)
        scheduler.step(1.0)

        current_lr = optimizer.param_groups[0]["lr"]
        self.assertIsInstance(current_lr, torch.Tensor)
        self.assertTrue(torch.allclose(current_lr, torch.tensor([0.05])))

    def test_mixed_tensor_and_float_lr(self):
        first = torch.nn.Parameter(torch.ones(1))
        second = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD(
            [
                {"params": [first], "lr": torch.tensor([0.1])},
                {"params": [second], "lr": 0.2},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=0, min_lr=0.001
        )

        scheduler.step(1.0)
        scheduler.step(1.0)

        tensor_lr = optimizer.param_groups[0]["lr"]
        float_lr = optimizer.param_groups[1]["lr"]
        self.assertIsInstance(tensor_lr, torch.Tensor)
        self.assertTrue(torch.allclose(tensor_lr, torch.tensor([0.05])))
        self.assertEqual(float_lr, 0.1)


if __name__ == "__main__":
    run_tests()
