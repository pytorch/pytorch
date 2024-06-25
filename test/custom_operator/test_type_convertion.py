# Owner(s): ["module: pt2-dispatcher"]
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union  # noqa: F401

import torch
from torch._library.infer_schema import tuple_to_list
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTypeConversion(TestCase):
    """In infer_schema(), we try to suggest a correct type when the type annotation is wrong."""

    def setUp(self):
        self.supported_base_types = [
            int,
            float,
            bool,
            str,
            torch.device,
            torch.Tensor,
            torch.dtype,
            torch.types.Number,
        ]

    def test_simple_tuple(self):
        self.assertEqual(List, tuple_to_list(Tuple))

    def test_supported_types(self):
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, t, t])
            self.assertEqual(result_type, List[t])

            result_type = tuple_to_list(Tuple[t])
            self.assertEqual(result_type, List[t])

    def test_optional(self):
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])

            result_type = tuple_to_list(Tuple[t, t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])


if __name__ == "__main__":
    run_tests()
