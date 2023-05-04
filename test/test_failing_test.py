# Owner(s): ["module: tests"]

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests)

# TODO: remove this
SIZE = 100

class TestSortAndSelect(TestCase):

    def test_topk_arguments(self, device):
        q = torch.randn(10, 2, 10, device=device)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    def test_failing_inputs(self, device):
        self.assertEqual(4, 5, "this shold fail ofc")


    def test_more_failing_inputs(self, device):
        self.assertEqual(6, 5, "this shold fail too")


    def test_success_comp(self, device):
        self.assertEqual(5, 5, "this shold fail...scratch that, it should pass")


instantiate_device_type_tests(TestSortAndSelect, globals())

if __name__ == '__main__':
    run_tests()
