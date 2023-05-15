# Owner(s): ["module: bazel"]

"""
This test module contains a minimalistic "smoke tests" for the bazel build.

Currently it doesn't use any testing framework (i.e. pytest)
TODO: integrate this into the existing pytorch testing framework.
"""

import torch

def test_sum():
    assert torch.eq(torch.tensor([[1, 2, 3]]) + torch.tensor([[4, 5, 6]]), torch.tensor([[5, 7, 9]])).all()

test_sum()
