# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestItt(TestCase):
    def setUp(self):
        super(TestItt, self).setUp()

    def tearDown(self):
        super(TestItt, self).tearDown()

    def test_itt(self):
        # Just making sure we can see the symbols
        torch.profiler.itt.range_push("foo")
        torch.profiler.itt.mark("bar")
        torch.profiler.itt.range_pop()

if __name__ == '__main__':
    run_tests()
