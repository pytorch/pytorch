import torch
import unittest
from common_utils import TEST_WITH_NESTEDTENSORS

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
def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32):
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

    return torch.tensor(arr).float().view(size) / m


def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low


class TestNestedTensor(TestCase):

    def gen_float_tensor(self, seed, shape):
        return random_float_tensor(seed, shape)

    def test_constructor(self):
        tensors = []
        for i in range(16):
            tensors.append(self.gen_float_tensor(i, (16, 128, 128)))
        nested_tensor = torch.nestedtensor(tensors)
        for i in range(16):
            tensors[i].mul_(i + 2)
        for i in range(16):
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
        a1 = torch.nestedtensor([self.gen_float_tensor(1, (2,)),
                                 self.gen_float_tensor(2, (2,))])
        a2 = torch.nestedtensor([self.gen_float_tensor(1, (2,)).exp_(),
                                 self.gen_float_tensor(2, (2,)).exp_()])
        assert (torch.exp(a1) == a2).all()
        assert not (a1 == a2).any()
        assert (a1.exp() == a2).all()
        assert not (a1 == a2).any()
        assert (a1.exp_() == a2).all()
        assert (a1 == a2).all()


    def test_binary(self):
        a = self.gen_float_tensor(1, (2,))
        b = self.gen_float_tensor(2, (2,))
        c = self.gen_float_tensor(3, (2,))
        # The constructor is suppoed to copy!
        a1 = torch.nestedtensor([a, b])
        a2 = torch.nestedtensor([b, c])
        a3 = torch.nestedtensor([a + b, b + c])
        assert (a3 == torch.add(a1, a2)).all()
        assert not (a3 == a1).any()
        assert not (a3 == a2).any()
        assert (a3 == a1.add(a2)).all()
        assert not (a3 == a1).any()
        assert not (a3 == a2).any()
        assert (a3 == a1.add_(a2)).all()
        assert (a3 == a1).all()

if __name__ == "__main__":
    unittest.main()
