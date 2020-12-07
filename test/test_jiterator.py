import torch
from itertools import product
from torch.testing._internal.common_utils import (
    TestCase, run_tests, make_tensor)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCUDA, dtypes, dtypesIfCUDA)

class TestJiterator(TestCase):

    @onlyCUDA
    @dtypes(*product((torch.int, torch.float), (torch.int, torch.float)))
    def test_foo(self, device, dtypes):
        a = make_tensor((10, 10), device=device, dtype=dtypes[0], low=None, high=None)
        b = make_tensor((10, 10), device=device, dtype=dtypes[1], low=None, high=None)
        self.assertEqual(torch.foo(a,b), torch.add(a,b), atol=0., rtol=0.)

instantiate_device_type_tests(TestJiterator, globals())

if __name__ == '__main__':
    run_tests()
