import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


class TestDCPFlattenedMissingState(TestCase):
    def test_strict_reports_missing_state(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
        opt = torch.optim.AdamW(model.parameters())
        model(torch.randn(2, 2)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        state = get_optimizer_state_dict(model, opt, options=options)
        for key in list(state):
            if key.startswith("state.1.weight."):
                del state[key]

        with self.assertRaisesRegex(RuntimeError, "Missing optimizer state"):
            set_optimizer_state_dict(model, torch.optim.AdamW(model.parameters()), state, options=options)

    def test_non_strict_skips_missing_state(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
        opt = torch.optim.AdamW(model.parameters())
        model(torch.randn(2, 2)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        options = StateDictOptions(flatten_optimizer_state_dict=True, strict=False)
        state = get_optimizer_state_dict(model, opt, options=options)
        for key in list(state):
            if key.startswith("state.1.weight."):
                del state[key]

        loaded = torch.optim.AdamW(model.parameters())
        set_optimizer_state_dict(model, loaded, state, options=options)
        self.assertNotIn(list(model.parameters())[1], loaded.state)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
