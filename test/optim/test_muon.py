# Owner(s): ["module: optimizer"]

import copy

import torch
from torch import nn, Tensor
from torch.optim import AdamW, Muon

from torch.testing._internal.common_utils import load_tests, TestCase


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class TestMuon(TestCase):
    def test_muon_fqn_set_empty_equals_to_adamw(self):
        model0 = nn.Linear(5, 5)
        model1 = copy.deepcopy(model0)

        lr = 1e-3
        wd = 0.1
        adamw_betas = (0.9, 0.95)
        adamw_eps = 1e-8

        adamw = AdamW(
            model0.parameters(),
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=wd,
        )

        # Muon on model1, with an “empty” FQN so it should fall back to AdamW behavior
        muon_param_fqns = [""]
        muon = Muon(
            model1.named_parameters(),
            lr=lr,
            wd=wd,
            muon_param_fqns=muon_param_fqns,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        torch.manual_seed(0)
        for p0, p1 in zip(model0.parameters(), model1.parameters()):
            g = torch.randn_like(p0)
            p0.grad = g.clone()
            p1.grad = g.clone()

        adamw.step()
        muon.step()

        for p0, p1 in zip(model0.parameters(), model1.parameters()):
            self.assertTrue(
                torch.allclose(p0, p1, atol=1e-6),
                "Muon did not match AdamW for empty-FQN case",
            )


if __name__ == "__main__":
    print("These tests should be run through test/optim/test_muon.py instead")
