# Owner(s): ["module: nn"]

import warnings

import torch
import torch.nn.utils.stateless as stateless
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFunctionalCallGenerator(TestCase):
    def _check_generator(self, functional_call):
        module = torch.nn.Linear(2, 2)
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.fill_(1.0)

        x = torch.ones(1, 2)
        parameters = (
            (name, torch.zeros_like(parameter))
            for name, parameter in module.named_parameters()
        )

        result = functional_call(module, parameters, x)
        self.assertEqual(result, torch.zeros_like(result))
        self.assertEqual(module(x), torch.full_like(result, 3.0))

    def test_torch_func_accepts_generator(self):
        self._check_generator(torch.func.functional_call)

    def test_stateless_accepts_generator(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._check_generator(stateless.functional_call)


if __name__ == "__main__":
    run_tests()
