import torch
import unittest
from common_utils import TestCase

torch = torch.nested.monkey_patch(torch)


def _shape_prod(shape_):
    shape = tuple(shape_)
    start = 1
    for s in shape:
        start = start * s
    return start

# From torchaudio by jamarshon


def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32,
                        requires_grad=False):
    """ Generates random tensors given a seed and size
    https://en.wikipedia.org/wiki/Linear_congruential_generator
    X_{n + 1} = (a * X_n + c) % m
    Using Borland C/C++ values
     The tensor will have values between [0,1)
    Inputs:
        seed (int): an int
        size (Tuple[int]): the size of the output tensor
        a (int): the multiplier constant to the generator
        c (int): the additive constant to the generator
        m (int): the modulus constant to the generator
    """
    num_elements = 1
    for s in size:
        num_elements *= s

    arr = [(a * seed + c) % m]
    for i in range(num_elements - 1):
        arr.append((a * arr[i] + c) % m)

    return torch.tensor(arr, requires_grad=requires_grad).float().view(size) / m


def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low


def gen_float_tensor(seed, shape, requires_grad=False):
    return random_float_tensor(seed, shape, requires_grad=requires_grad)


class TestNestedTensor(TestCase):

    def test_nested_constructor(self):
        def _gen_nested_tensor():
            tensors = []
            num_tensors = 4
            for i in range(num_tensors):
                tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
            return torch.nested_tensor(tensors)
        num_nested_tensor = 3
        nested_tensors = [_gen_nested_tensor() for _ in range(num_nested_tensor)]
        nested_tensor = torch.nested_tensor(nested_tensors)
        nested_tensor.cos_()

    def test_constructor(self):
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
        nested_tensor = torch.nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertTrue((tensors[i] != nested_tensor._tensors[i]).all())
        self.assertRaises(ValueError, lambda: torch.nested_tensor([]))
        self.assertRaises(ValueError, lambda: torch.nested_tensor(torch.tensor([3.0])))

    def test_nested_size(self):
        a = torch.nested_tensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
        self.assertEqual(a.nested_size(), na)

    def test_len(self):
        a = torch.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([3, 4]),
                                 torch.tensor([5, 6]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 4)
        a = torch.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 2)
        a = torch.nested_tensor([torch.tensor([1, 2])])
        self.assertEqual(len(a), 1)

    def test_equal(self):
        a1 = torch.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a2 = torch.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a3 = torch.nested_tensor([torch.tensor([3, 4]),
                                  torch.tensor([5, 6])])
        # Just exercising them until we have __bool__, all() etc.
        self.assertTrue((a1 == a2).all())
        self.assertTrue((a1 != a3).all())
        self.assertTrue(not (a1 != a2).any())
        self.assertTrue(not (a1 == a3).any())

    def test_unary(self):
        for func in torch.nested.codegen.extension.get_unary_functions():
            data = [gen_float_tensor(1, (2, 3)) - 0.5,
                    gen_float_tensor(2, (2, 3)) - 0.5]
            if func in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
                data = list(map(lambda x: x.abs(), data))
            a1 = torch.nested_tensor(data)
            a2 = torch.nested_tensor(list(map(lambda x: getattr(torch, func)(x), data)))
            self.assertTrue((getattr(torch, func)(a1) == a2).all())
            self.assertTrue((getattr(a1, func)() == a2).all())
            self.assertTrue((getattr(a1, func + "_")() == a2).all())
            self.assertTrue((a1 == a2).all())

    def test_binary(self):
        for func in torch.nested.codegen.extension.get_binary_functions():
            a = gen_float_tensor(1, (2, 3))
            b = gen_float_tensor(2, (2, 3))
            c = gen_float_tensor(3, (2, 3))
            # The constructor is supposed to copy!
            a1 = torch.nested_tensor([a, b])
            a2 = torch.nested_tensor([b, c])
            a3 = torch.nested_tensor([getattr(torch, func)(a, b),
                                      getattr(torch, func)(b, c)])
            self.assertTrue((a3 == getattr(torch, func)(a1, a2)).all())
            self.assertTrue((a3 == getattr(a1, func)(a2)).all())
            self.assertTrue((a3 == getattr(a1, func + "_")(a2)).all())
            self.assertTrue((a3 == a1).all())

    def test_detach(self):
        data = [gen_float_tensor(1, (10, 10)),
                gen_float_tensor(2, (10, 10)),
                gen_float_tensor(3, (10, 10))]
        ones_data = [torch.ones(10, 10),
                     torch.ones(10, 10),
                     torch.ones(10, 10)]
        # We don't support scalar arguments yet (broadcasting)
        # This will be part of NestedTensor 0.0.3
        twos_data = [torch.ones(10, 10) * 2,
                     torch.ones(10, 10) * 2,
                     torch.ones(10, 10) * 2]
        fours_data = [torch.ones(10, 10) * 4,
                      torch.ones(10, 10) * 4,
                      torch.ones(10, 10) * 4]
        ones = torch.nested_tensor(ones_data).to(torch.float)
        twos = torch.nested_tensor(twos_data).to(torch.float)
        fours = torch.nested_tensor(fours_data).to(torch.float)
        x = torch.nested_tensor(data, requires_grad=True)
        y = x + twos
        y = y.detach()
        z = y * fours + twos
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.nested_tensor(data, requires_grad=True)
        y = x * twos
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertRaises(NotImplementedError, lambda: y.grad_fn)
        z = x + y
        z.sum().backward()

        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertTrue((x.grad.data == ones).all())

        # in-place detach
        x = torch.nested_tensor(data, requires_grad=True)
        y = torch.nested_tensor(data, requires_grad=True)
        a = x * twos
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertTrue((x.grad.data == ones * twos).all())
        self.assertTrue((y.grad.data == ones * twos).all())

        # TODO: view semantics will be defined by NestedTensor 0.0.3 or 0.0.4
        # in-place deatch on a view raises an exception
        # view = x.narrow(0, 1, 4)
        # self.assertRaisesRegex(RuntimeError, 'view', lambda: view.detach_())

    def test_detach_base(self):
        x = torch.nested_tensor([torch.randn(10, 10)], requires_grad=True)
        x.detach_()
        self.assertFalse(x.requires_grad)


if __name__ == "__main__":
    # unittest.main()
    def _gen_nested_tensor():
        tensors = []
        num_tensors = 4
        for i in range(num_tensors):
            tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
        return torch.nested_tensor(tensors)
    num_nested_tensor = 3
    nested_tensors = [_gen_nested_tensor() for _ in range(num_nested_tensor)]
    nested_tensor = torch.nested_tensor(nested_tensors)
    nested_tensor.cos_()
