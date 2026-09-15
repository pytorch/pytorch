import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


class TestDCPFrozenOptimizerState(TestCase):
    def test_frozen_state_round_trip(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.Linear(2, 2, bias=False),
        )
        opt = torch.optim.AdamW(model.parameters())
        sum(param.sum() for param in model.parameters()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        first, second = model.parameters()
        first.requires_grad_(False)

        state = get_optimizer_state_dict(model, opt)
        loaded = torch.optim.AdamW(model.parameters())
        set_optimizer_state_dict(model, loaded, state)

        self.assertEqual(set(opt.state_dict()["state"]), set(loaded.state_dict()["state"]))

    def test_load_preserves_requires_grad(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.Linear(2, 2, bias=False),
        )
        opt = torch.optim.AdamW(model.parameters())
        sum(param.sum() for param in model.parameters()).backward()
        opt.step()
        state = get_optimizer_state_dict(model, opt)

        first, second = model.parameters()
        first.requires_grad_(False)
        second.requires_grad_(True)
        loaded = torch.optim.AdamW(model.parameters())
        set_optimizer_state_dict(model, loaded, state)

        self.assertFalse(first.requires_grad)
        self.assertTrue(second.requires_grad)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
