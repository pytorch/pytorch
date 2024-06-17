from __future__ import annotations

import unittest

import torch
from torch.testing._internal.common_utils import TestCase

class TestCustomOperatorsWithAnnotation(TestCase):
    def test_tensor(self):
        @torch.library.custom_op("mylibrary::foo_op", mutates_args={})
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

if __name__ == "__main__":
    run_tests()
