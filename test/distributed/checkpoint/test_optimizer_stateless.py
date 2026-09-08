import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


class TestDCPStatelessOptimizer(TestCase):
    def test_flattened_sgd_round_trip(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.Linear(2, 2, bias=False),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 2)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        options = StateDictOptions(flatten_optimizer_state_dict=True)
        state = get_optimizer_state_dict(model, opt, options=options)
        loaded = torch.optim.SGD(model.parameters(), lr=0.1)
        set_optimizer_state_dict(model, loaded, state, options=options)

        self.assertEqual(opt.state_dict()["state"], {})
        self.assertEqual(loaded.state_dict()["state"], {})


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
