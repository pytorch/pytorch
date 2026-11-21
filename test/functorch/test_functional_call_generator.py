# Owner(s): ["module: functorch"]

import warnings

import torch
import torch.nn.utils.stateless as stateless
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFunctionalCallIterable(TestCase):
    def _check_parameter_pairs(self, functional_call, parameters):
        module = torch.nn.Linear(2, 2)
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.fill_(1.0)

        x = torch.ones(1, 2)
        result = functional_call(module, parameters(module), x)

        self.assertEqual(result, torch.zeros_like(result))
        self.assertEqual(module(x), torch.full_like(result, 3.0))

    def _zero_named_parameters(self, module):
        return (
            (name, torch.zeros_like(parameter))
            for name, parameter in module.named_parameters()
        )

    def _zero_named_parameters_list(self, module):
        return [
            (name, torch.zeros_like(parameter))
            for name, parameter in module.named_parameters()
        ]

    def test_torch_func_accepts_generator(self):
        self._check_parameter_pairs(
            torch.func.functional_call, self._zero_named_parameters
        )

    def test_torch_func_accepts_non_iterator_iterable(self):
        self._check_parameter_pairs(
            torch.func.functional_call, self._zero_named_parameters_list
        )

    def test_stateless_accepts_generator(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._check_parameter_pairs(
                stateless.functional_call, self._zero_named_parameters
            )

    def test_stateless_accepts_non_iterator_iterable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._check_parameter_pairs(
                stateless.functional_call, self._zero_named_parameters_list
            )

    def test_torch_func_iterable_dicts_preserve_duplicate_validation(self):
        module = torch.nn.Linear(2, 2)
        parameters = iter(
            [
                {"weight": torch.zeros_like(module.weight)},
                {"weight": torch.ones_like(module.weight)},
            ]
        )

        with self.assertRaisesRegex(ValueError, "appeared in multiple dictionaries"):
            torch.func.functional_call(module, parameters, torch.ones(1, 2))

    def test_torch_func_rejects_duplicate_parameter_pairs(self):
        module = torch.nn.Linear(2, 2)
        parameters = iter(
            [
                ("weight", torch.zeros_like(module.weight)),
                ("weight", torch.ones_like(module.weight)),
            ]
        )

        with self.assertRaisesRegex(ValueError, "appeared multiple times"):
            torch.func.functional_call(module, parameters, torch.ones(1, 2))


if __name__ == "__main__":
    run_tests()
