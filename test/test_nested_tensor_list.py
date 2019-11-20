import traceback
import functools
import pdb
import sys
import unittest
from common_utils import TestCase
import random

import traceback
import functools
import pdb
import sys
import unittest
from unittest import TestCase
import random
import torch

def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return (torch.floor(torch.rand(size) * (high - low)) + low).to(torch.int64)


def gen_float_tensor(seed, shape, requires_grad=False):
    torch.manual_seed(seed)
    return torch.rand(shape, requires_grad=requires_grad)


def gen_random_int(seed, low=0, high=2 ** 32):
    """ Returns random integer in [low, high)
    """
    return int(random_int_tensor(seed, (), low=low, high=high))


# TODO: Something occasionally causes a NaN here...
def gen_nested_list(seed, nested_dim, tensor_dim, size_low=1, size_high=10):
    tensors = []
    num_tensors = gen_random_int(
        (seed * nested_dim + seed) * 1024, low=size_low, high=size_high)
    assert nested_dim > 0
    if nested_dim == 1:
        for i in range(num_tensors):
            ran = gen_random_int((seed * nested_dim + seed)
                                 * (1024 * i), low=size_low, high=size_high)
            ran_size = ()
            for _ in range(tensor_dim):
                ran = gen_random_int(ran * 1024, low=size_low, high=size_high)
                ran_size = ran_size + (ran,)

            tensors.append(gen_float_tensor(ran, ran_size))
    else:
        for i in range(num_tensors):
            tensors.append(gen_nested_list(
                num_tensors * seed, nested_dim - 1, tensor_dim))
    return tensors


def nested_map(fn, data):
    if isinstance(data, list):
        return [nested_map(fn, d) for d in data]
    else:
        return fn(data)

def gen_nested_tensor(seed, nested_dim, tensor_dim, size_low=1, size_high=10):
    return torch._ListNestedTensor(gen_nested_list(seed, nested_dim, tensor_dim, size_low=size_low, size_high=size_high))

class Test_ListNestedTensor(TestCase):

    def test_nested_constructor(self):
        num_nested_tensor = 3
        # TODO: Shouldn't be constructable
        nested_tensors = [gen_nested_tensor(i, i, 3)
                          for i in range(1, num_nested_tensor)]
        # nested_tensor = torch.nested_tensor(nested_tensors)

    def test_constructor(self):
        """
        This tests whether _ListNestedTensor stores Variables that share storage with
        the input Variables used for construction.
        """
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
        nested_tensor = torch._ListNestedTensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertTrue((tensors[i] == nested_tensor.unbind()[i]).all())
            self.assertEqual(tensors[i].storage().data_ptr(), nested_tensor.unbind()[i].storage().data_ptr())

    def test_default_constructor(self):
        # self.assertRaises(TypeError, lambda: torch.nested_tensor())
        # nested_dim is 1 and dim is 1 too.
        default_nested_tensor = torch._ListNestedTensor([])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.nested_dim(), 0)
        self.assertEqual(default_nested_tensor.nested_size(), ())
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.requires_grad,
                         default_tensor.requires_grad)
        self.assertEqual(default_nested_tensor.is_pinned(),
                         default_tensor.is_pinned())

    def test_element_size(self):
        nt1 = torch._ListNestedTensor([])
        self.assertEqual(nt1.element_size(), torch.randn(1).element_size())
        a = torch.randn(4).int()
        nt2 = torch._ListNestedTensor([a])
        self.assertEqual(a.element_size(), nt2.element_size())

    def test_nested_dim(self):
        nt = torch._ListNestedTensor([torch.tensor(3)])
        self.assertTrue(nt.nested_dim() == 1)
        for i in range(2, 5):
            nt = gen_nested_tensor(i, i, 3)
            self.assertTrue(nt.nested_dim() == i)

    def test_nested_size(self):
        a = torch._ListNestedTensor(
            [torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
        self.assertEqual(a.nested_size(), na)

    def test_nested_stride(self):
        tensors = [torch.rand(1, 2, 4)[:, :, 0], torch.rand(2, 3, 4)[:, 1, :], torch.rand(3, 4, 5)[1, :, :]]
        a = torch._ListNestedTensor(tensors)
        na = tuple(t.stride() for t in tensors)
        self.assertEqual(a.nested_stride(), na)


    def test_pin_memory(self):
        # Check if it can be applied widely
        nt = gen_nested_tensor(1, 4, 3)
        nt1 = nt.pin_memory()

        # Make sure it's actually a copy
        self.assertFalse(nt.is_pinned())
        self.assertTrue(nt1.is_pinned())
        a1 = torch.randn(1, 2)
        a2 = torch.randn(2, 3)
        nt2 = torch._ListNestedTensor([a1, a2])
        self.assertFalse(a1.is_pinned())
        self.assertFalse(a2.is_pinned())

        # Double check property transfers
        nt3 = nt2.pin_memory()
        self.assertFalse(nt2.is_pinned())
        self.assertTrue(nt3.is_pinned())

        # Check whether pinned memory is applied to constiuents
        # and relevant constiuents only.
        a3, a4 = nt3.unbind()
        a5, a6 = nt2.unbind()
        self.assertFalse(a1.is_pinned())
        self.assertFalse(a2.is_pinned())
        self.assertTrue(a3.is_pinned())
        self.assertTrue(a4.is_pinned())
        self.assertFalse(a5.is_pinned())
        self.assertFalse(a6.is_pinned())




    def test_len(self):
        a = torch._ListNestedTensor([torch.tensor([1, 2]),
                                 torch.tensor([3, 4]),
                                 torch.tensor([5, 6]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 4)
        a = torch._ListNestedTensor([torch.tensor([1, 2]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 2)
        a = torch._ListNestedTensor([torch.tensor([1, 2])])
        self.assertEqual(len(a), 1)



    def test_unbind(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt = torch._ListNestedTensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue((a == a1).all())
        self.assertTrue((b == b1).all())

        a = gen_float_tensor(1, (2, 3)).add_(1)
        nt = torch._ListNestedTensor([a])
        self.assertTrue((a == nt.unbind()[0]).all())

    def test_contiguous(self):
        a = torch._ListNestedTensor([torch.tensor([1, 2]),
                                 torch.tensor([3, 4]),
                                 torch.tensor([5, 6]),
                                 torch.tensor([7, 8])])
        self.assertTrue(not a.is_contiguous())

if __name__ == "__main__":
    unittest.main()
