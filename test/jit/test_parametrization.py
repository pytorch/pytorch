# Owner(s): ["oncall: jit"]


import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize

from torch.testing._internal.jit_utils import JitTestCase


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestParametrization(JitTestCase):
    # Define some parametrization
    class Symmetric(nn.Module):
        def forward(self, X):
            return X.triu() + X.triu(1).mT

    def test_traceable(self):
        r"""Test the jit scripting and tracing of a parametrized model."""
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        # Check the tracing works. Because traced functions cannot be called
        # directly, we run the comparison on the activations.
        traced_model = torch.jit.trace_module(model, {'forward': x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)

        # Check traced model works with caching
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)

        # Check the tracing throws an error when caching
        with self.assertRaisesRegex(RuntimeError,
                                    'Cannot trace a model while caching'):
            with parametrize.cached():
                traced_model = torch.jit.trace_module(model, {'forward': x})
