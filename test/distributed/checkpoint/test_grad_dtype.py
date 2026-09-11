import torch

from torch.testing._internal.common_utils import TestCase


class TestGradDtype(TestCase):
    def test_optimizer_state_uses_grad_dtype(self):
        from torch.distributed.checkpoint.state_dict import _init_optim_state

        param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        param.grad_dtype = torch.bfloat16
        optimizer = torch.optim.Adam([param], lr=0.0)

        _init_optim_state(optimizer)

        self.assertIn(param, optimizer.state)
        self.assertEqual(optimizer.state[param]["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(optimizer.state[param]["exp_avg_sq"].dtype, torch.bfloat16)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
