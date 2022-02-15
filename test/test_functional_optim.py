# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.distributed.optim.utils import functional_optim_map

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(3, 3, bias=False)
        self.lin2 = nn.Linear(3, 3, bias=False)

    def forward(self, t1):
        return self.lin2(F.relu(self.lin1(t1)))


class TestFunctionalOptimParity(TestCase):
    def _validate_parameters(self, params_1, params_2):
        for p1, p2 in zip(params_1, params_2):
            self.assertEqual(p1, p2)

    def _test_functional_optim_parity(self, optim_cls, *args, **kwargs):
        module_optim = MyModule()
        module_functional = MyModule()
        optim_params = module_optim.parameters()
        functional_params = module_functional.parameters()
        optim = optim_cls(optim_params, *args, **kwargs)
        functional_optim_cls = functional_optim_map.get(optim_cls, None)
        if not functional_optim_cls:
            raise ValueError(f"Functional optimizer not implemented for {optim_cls}")
        optim_functional = functional_optim_cls(
            [], *args, **kwargs, _allow_empty_param_list=True
        )
        if not hasattr(optim_functional, "step_param"):
            raise ValueError(
                f"Functional optimizer class {optim_functional} must implement step_param method."
            )

        # Initial weights should match
        self._validate_parameters(
            module_optim.parameters(), module_functional.parameters()
        )
        # Save old parameters to verify optimizer modifies them.
        old_module_optim_params = [
            param.clone().detach() for param in module_optim.parameters()
        ]
        old_module_functional_params = [
            param.clone().detach() for param in module_functional.parameters()
        ]

        t1 = torch.randn(3, 3)
        for _ in range(10):
            module_optim.zero_grad()
            module_functional.zero_grad()
            # Forward + Backward
            optim_out = module_optim(t1).sum()
            functional_out = module_functional(t1).sum()
            optim_out.backward()
            functional_out.backward()
            # Optimizer step
            optim.step()
            # Functional optimizer step_param
            for param in module_functional.parameters():
                grad = param.grad
                optim_functional.step_param(param, grad)

            # Validate parameters are equal
            for optim_param, functional_param in zip(
                module_optim.parameters(), module_functional.parameters()
            ):
                self.assertEqual(optim_param, functional_param)
            # Validate parameters are modified.
            for i, (optim_param, functional_param) in enumerate(
                zip(module_optim.parameters(), module_functional.parameters())
            ):
                self.assertNotEqual(old_module_optim_params[i], optim_param)
                self.assertNotEqual(old_module_functional_params[i], functional_param)

    def test_functional_optim_parity_sgd(self):
        self._test_functional_optim_parity(SGD, 1e-2, momentum=0.9, weight_decay=0.01)

    def test_functional_optim_parity_adam(self):
        self._test_functional_optim_parity(Adam, 1e-2, betas=(0.9, 0.999), eps=1e-6)

    def test_functional_optim_parity_adam_w(self):
        self._test_functional_optim_parity(AdamW, 1e-2, betas=(0.9, 0.999), eps=1e-6)


if __name__ == "__main__":
    run_tests()
