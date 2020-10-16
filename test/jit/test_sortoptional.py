import os
import sys

import torch
from typing import List
from typing import Optional
from typing import Tuple

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestSortOptional(JitTestCase):
    def test_sort_primitive_optional(self):
        @torch.jit.script
        def sort_int_optional(inputs: List[Optional[int]]):
            inputs.sort()
            return inputs

        self.assertEqual(sort_int_optional([None, 1, 2, 5, None, 8]), [None, None, 1, 2, 5, 8])
        self.assertEqual(sort_int_optional([1, 2, 5, 8]), [1, 2, 5, 8])
        self.assertEqual(sort_int_optional([None]), [None])
        self.assertEqual(sort_int_optional([None, None, None]), [None, None, None])
        self.assertEqual(sort_int_optional([]), [])


    def test_sort_tuple_optional(self):
        @torch.jit.script
        def sort_tuple_optional(inputs: List[Optional[Tuple[Optional[int], Optional[int]]]]):
            inputs.sort()
            return inputs

        self.assertEqual(sort_tuple_optional([None, (1, 2), (5, 8), None, (3, None)]), [None, None, (1, 2), (3, None), (5, 8)])
        self.assertEqual(sort_tuple_optional([(1, 2), (5, 8)]), [(1, 2), (5, 8)])
        self.assertEqual(sort_tuple_optional([None]), [None])
        self.assertEqual(sort_tuple_optional([None, None, None]), [None, None, None])
        self.assertEqual(sort_tuple_optional([]), [])
