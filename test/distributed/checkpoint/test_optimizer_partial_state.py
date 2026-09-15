import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint._grad_dtype import _init_optim_state


class TestDCPOptimizerState(TestCase):
    def test_partial_state_initialization(self):
        p0 = torch.nn.Parameter(torch.ones(2))
        p1 = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.AdamW([p0, p1])
        p0.grad = torch.ones_like(p0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _init_optim_state(opt)

        self.assertIn(p0, opt.state)
        self.assertIn(p1, opt.state)
        self.assertGreaterEqual(opt.state[p1]["step"].item(), 1)

    def test_custom_grad_dtype(self):
        param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        param.grad_dtype = torch.float64
        opt = torch.optim.AdamW([param])

        _init_optim_state(opt)

        self.assertEqual(opt.state[param]["exp_avg"].dtype, torch.float64)
        self.assertEqual(opt.state[param]["exp_avg_sq"].dtype, torch.float64)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
