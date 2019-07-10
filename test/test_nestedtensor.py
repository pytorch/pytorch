import torch.prototypes.nestedtensor as nestedtensor

NestedTensor = nestedtensor.NestedTensor

import torch
import unittest

def _shape_prod(shape_):
    shape = tuple(shape_)
    start = 1
    for s in shape:
        start = start  * s
    return start

def gen_float_tensor(seed, shape):
    data = []
    for data_i in range(_shape_prod(shape)):
        data.append(data_i + seed)
    ret = torch.tensor(data)
    return ret.reshape(shape).float()

class NestedTensorTest(unittest.TestCase):

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
        a = torch.nestedtensor([])
        assert(len(a) == 0)
    
    
    def test_unbind(self):
        data = [gen_float_tensor(1, (2, 2)),
                gen_float_tensor(2, (2, 2)),
                gen_float_tensor(3, (2, 2))]
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
        a1 = torch.nestedtensor([gen_float_tensor(1, (2,)),
                                 gen_float_tensor(2, (2,))])
        a2 = torch.nestedtensor([gen_float_tensor(1, (2,)).exp_(),
                                 gen_float_tensor(2, (2,)).exp_()])
    
        assert (torch.exp(a1) == a2).all()
        assert not (a1 == a2).any()
        assert (a1.exp() == a2).all()
        assert not (a1 == a2).any()
        assert (a1.exp_() == a2).all()
        assert (a1 == a2).all()
    
    def test_binary(self):
        a = gen_float_tensor(1, (2,))
        b = gen_float_tensor(2, (2,))
        c = gen_float_tensor(3, (2,))
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

# TODO: Carefully test reference passing vs. value passing for each function
# TODO: Add more tests for variable length examples
# TODO: Need constructor tests
if __name__ == "__main__":
    unittest.main()
