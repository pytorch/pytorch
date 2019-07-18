import torch
import unittest
from common_utils import TEST_WITH_NESTEDTENSORS
import torch.tensortypes.nestedtensor.codegen as codegen

if TEST_WITH_NESTEDTENSORS:
    import torch.tensortypes.nestedtensor as nestedtensor
    from common_utils import TestCase
    NestedTensor = nestedtensor.NestedTensor
else:
    print('NestedTensors not available, skipping tests')
    TestCase = object


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


class TestNestedTensor(TestCase):

    def gen_float_tensor(self, seed, shape, requires_grad=False):
        return random_float_tensor(seed, shape, requires_grad=requires_grad)

    def test_constructor(self):
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(self.gen_float_tensor(i, (16, 128, 128)))
        nested_tensor = torch.nestedtensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            assert (tensors[i] != nested_tensor.tensors[i]).all()
        self.assertRaises(ValueError, lambda: torch.nestedtensor([]))
        self.assertRaises(ValueError, lambda: torch.nestedtensor(torch.tensor([3.0])))

    def test_nested_size(self):
        a = torch.nestedtensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
        assert a.nested_size() == na


    def test_len(self):
        a = torch.nestedtensor([torch.tensor([1, 2]), 
                                torch.tensor([3, 4]), 
                                torch.tensor([5, 6]), 
                                torch.tensor([7, 8])])
        assert(len(a) == 4)
        a = torch.nestedtensor([torch.tensor([1, 2]), 
                                torch.tensor([7, 8])])

        assert(len(a) == 2)
        a = torch.nestedtensor([torch.tensor([1, 2])])
        assert(len(a) == 1)


    def test_unbind(self):
        data = [self.gen_float_tensor(1, (2, 2)),
                self.gen_float_tensor(2, (2, 2)),
                self.gen_float_tensor(3, (2, 2))]
        a = torch.nestedtensor(data)
        b = a.unbind()
        c = torch.nestedtensor([data_i + 1 for data_i in data])
        for t in b:
            t.add_(1)
        assert (a == c).all()


    def test_equal(self):
        a1 = torch.nestedtensor([torch.tensor([1, 2]), 
                                 torch.tensor([7, 8])])
        a2 = torch.nestedtensor([torch.tensor([1, 2]), 
                                 torch.tensor([7, 8])])
        a3 = torch.nestedtensor([torch.tensor([3, 4]), 
                                 torch.tensor([5, 6])])
        # Just exercising them until we have __bool__, all() etc.
        assert (a1 == a2).all()
        assert (a1 != a3).all()
        assert not (a1 != a2).any()
        assert not (a1 == a3).any()


    def test_unary(self):
        for func in codegen.extension.get_unary_functions():
            data = [self.gen_float_tensor(1, (2, 3)) - 0.5,
                    self.gen_float_tensor(2, (2, 3)) - 0.5]
            if func in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
                data = list(map(lambda x: x.abs(), data))
            a1 = torch.nestedtensor(data)
            a2 = torch.nestedtensor(list(map(lambda x: getattr(torch, func)(x), data)))
            assert (getattr(torch, func)(a1) == a2).all()
            assert (getattr(a1, func)() == a2).all()
            assert (getattr(a1, func + "_")() == a2).all()
            assert (a1 == a2).all()


    def test_binary(self):
        for func in codegen.extension.get_binary_functions():
            a = self.gen_float_tensor(1, (2, 3))
            b = self.gen_float_tensor(2, (2, 3))
            c = self.gen_float_tensor(3, (2, 3))
            # The constructor is supposed to copy!
            a1 = torch.nestedtensor([a, b])
            a2 = torch.nestedtensor([b, c])
            a3 = torch.nestedtensor([getattr(torch, func)(a, b),
                                     getattr(torch, func)(b, c)])
            assert (a3 == getattr(torch, func)(a1, a2)).all()
            assert not (a3 == a1).any()
            assert not (a3 == a2).any()
            assert (a3 == getattr(a1, func)(a2)).all()
            assert not (a3 == a1).any()
            assert not (a3 == a2).any()
            assert (a3 == getattr(a1, func + "_")(a2)).all()
            assert (a3 == a1).all()

    def test_detach(self):
        data = [self.gen_float_tensor(1, (10, 10)),
                self.gen_float_tensor(2, (10, 10)),
                self.gen_float_tensor(3, (10, 10))]
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
        ones = torch.nestedtensor(ones_data).to(torch.float)
        twos = torch.nestedtensor(twos_data).to(torch.float)
        fours = torch.nestedtensor(fours_data).to(torch.float)
        x = torch.nestedtensor(data, requires_grad=True)
        y = x + twos
        y = y.detach()
        z = y * fours + twos
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.nestedtensor(data, requires_grad=True)
        print('1 x.requires_grad')
        print(x.requires_grad)
        y = x * twos
        print('2 x.requires_grad')
        print(x.requires_grad)
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        print('3 x.requires_grad')
        print(x.requires_grad)
        z.sum().backward()

        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        print('4 x.requires_grad')
        print(x.requires_grad)
        print('x.grad')
        print(x.grad)
        # print('x.buffer().grad')
        # print(x.buffer().grad)
        self.assertTrue((x.grad.data == ones).all())

        # in-place detach
        x = torch.nestedtensor(data, requires_grad=True)
        y = torch.nestedtensor(data, requires_grad=True)
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
        "detaching base does not detach view"
        x = torch.nestedtensor([torch.randn(10, 10)], requires_grad=True)
        x.detach_()
        self.assertFalse(x.requires_grad)

if __name__ == "__main__":
    unittest.main()
