import torch
import unittest
from common_utils import TestCase


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


# TODO: Something occasionally causes a NaN here?
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
    return torch.nestedtensor._ListNestedTensor(gen_nested_list(seed, nested_dim, tensor_dim, size_low=size_low, size_high=size_high))


def _test_property(self, fn):
    num_nested_tensor = 3
    nested_tensor_lists = [gen_nested_list(i, i, 3)
                           for i in range(1, num_nested_tensor)]
    first_tensors = [get_first_tensor(ntl) for ntl in nested_tensor_lists]
    nested_tensors = [torch.nestedtensor._ListNestedTensor(ntl) for ntl in nested_tensor_lists]
    for nested_tensor, first_tensor in zip(nested_tensors, first_tensors):
        self.assertEqual(fn(nested_tensor), fn(first_tensor))


def get_first_tensor(nested_list):
    if isinstance(nested_list, list):
        return get_first_tensor(nested_list[0])
    else:
        return nested_list


# TODO: Test backward
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
        nested_tensor = torch.nestedtensor._ListNestedTensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertTrue((tensors[i] == nested_tensor.unbind()[i]).all())
            self.assertEqual(tensors[i].storage().data_ptr(), nested_tensor.unbind()[i].storage().data_ptr())

    def test_default_constructor(self):
        # self.assertRaises(TypeError, lambda: torch.nested_tensor())
        # nested_dim is 1 and dim is 1 too.
        default_nested_tensor = torch.nestedtensor._ListNestedTensor([])
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
        nt1 = torch.nestedtensor._ListNestedTensor([])
        self.assertEqual(nt1.element_size(), torch.randn(1).element_size())
        a = torch.randn(4).int()
        nt2 = torch.nestedtensor._ListNestedTensor([a])
        self.assertEqual(a.element_size(), nt2.element_size())

    def test_nested_dim(self):
        nt = torch.nestedtensor._ListNestedTensor([torch.tensor(3)])
        self.assertTrue(nt.nested_dim() == 1)
        for i in range(2, 5):
            nt = gen_nested_tensor(i, i, 3)
            self.assertTrue(nt.nested_dim() == i)

    def test_nested_size(self):
        a = torch.nestedtensor._ListNestedTensor(
            [torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
        self.assertEqual(a.nested_size(), na)

    def test_nested_stride(self):
        tensors = [torch.rand(1, 2, 4)[:, :, 0], torch.rand(2, 3, 4)[:, 1, :], torch.rand(3, 4, 5)[1, :, :]]
        a = torch.nestedtensor._ListNestedTensor(tensors)
        na = tuple(t.stride() for t in tensors)
        self.assertEqual(a.nested_stride(), na)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not enabled.")
    def test_pin_memory(self):
        # Check if it can be applied widely
        nt = gen_nested_tensor(1, 4, 3)
        nt1 = nt.pin_memory()

        # Make sure it's actually a copy
        self.assertFalse(nt.is_pinned())
        self.assertTrue(nt1.is_pinned())
        a1 = torch.randn(1, 2)
        a2 = torch.randn(2, 3)
        nt2 = torch.nestedtensor._ListNestedTensor([a1, a2])
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
        a = torch.nestedtensor._ListNestedTensor([torch.tensor([1, 2]),
                                                  torch.tensor([3, 4]),
                                                  torch.tensor([5, 6]),
                                                  torch.tensor([7, 8])])
        self.assertEqual(len(a), 4)
        a = torch.nestedtensor._ListNestedTensor([torch.tensor([1, 2]),
                                                  torch.tensor([7, 8])])
        self.assertEqual(len(a), 2)
        a = torch.nestedtensor._ListNestedTensor([torch.tensor([1, 2])])
        self.assertEqual(len(a), 1)

    def test_dtype(self):
        _test_property(self, lambda x: x.dtype)

    def test_device(self):
        _test_property(self, lambda x: x.device)

    def test_layout(self):
        _test_property(self, lambda x: x.layout)

    def test_requires_grad(self):
        _test_property(self, lambda x: x.requires_grad)

    def test_unbind(self):
        # This is the most important operation. We want to make sure
        # that the Tensors we use for construction can be retrieved
        # and used independently while still being kept track of.

        # In fact _ListNestedTensor behave just like a list. Any
        # list of torch.Tensors you initialize it with will be
        # unbound to have the same id. That is, they are indeed
        # the same Variable, since each torch::autograd::Variable has
        # assigned to it a unique PyObject* by construction.

        # TODO: Check that unbind returns torch.Tensors when nested_dim is 1

        a = torch.tensor([1, 2])
        b = torch.tensor([7, 8])
        nt = torch.nestedtensor._ListNestedTensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is a1)
        self.assertTrue(b is b1)

        c = torch.tensor([3, 4])
        d = torch.tensor([5, 6])
        e = torch.tensor([6, 7])

        nt1 = torch.nestedtensor._ListNestedTensor([[c, d], [e]])
        nt11, nt12 = nt1.unbind()
        c1, d1 = nt11.unbind()
        e1 = nt12.unbind()[0]

        self.assertTrue(c is c1)
        self.assertTrue(d is d1)
        self.assertTrue(e is e1)

    def test_contiguous(self):
        a = torch.nestedtensor._ListNestedTensor([torch.tensor([1, 2]),
                                                  torch.tensor([3, 4]),
                                                  torch.tensor([5, 6]),
                                                  torch.tensor([7, 8])])
        self.assertTrue(not a.is_contiguous())


if __name__ == "__main__":
    unittest.main()
