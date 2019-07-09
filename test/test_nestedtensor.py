import torch.prototypes.nestedtensor as nestedtensor

NestedTensor = nestedtensor.NestedTensor

from torch import cat
from torch import stack
from torch.nn.functional import embedding
import torch

def test_cat():
    # Test concatenation of NestedTensors
    # This is a core operation used in stack and append

    def test_two_scalars():
        # Concating two scalars
        a = NestedTensor(torch.tensor(1.0),
                          torch.tensor(1))
        b = NestedTensor(torch.tensor(2.0),
                          torch.tensor(1))
        try:
            cat([a, b])
            raise Exception("It does not make sense to"
                            " concat two numbers")
        except AssertionError:
            pass

    def test_scalar_and_list():
        # Concating a scalar and a list
        a = NestedTensor(torch.tensor([]),
                          torch.tensor([]).type(torch.int64))
        b = NestedTensor(torch.tensor(2.0),
                          torch.tensor(1))
        try:
            cat([a, b])
            raise Exception("It does not make sense to"
                            " concat a number and a list")
        except AssertionError:
            pass

    def test_empty_lists():
        # Concating two empty lists
        a = NestedTensor(torch.tensor([]),
                          torch.tensor([]).type(torch.int64))
        b = NestedTensor(torch.tensor([]),
                          torch.tensor([]).type(torch.int64))
        c = cat([a, b])
        assert c.tensor.numel() == 0
        assert c.mask.numel() == 0
        assert c.tensor.shape == (0,)
        assert c.mask.shape == (0,)

    def test_empty_list_single_list():
        # Concating an empty list and a single element list
        a = NestedTensor(torch.tensor([]),
                          torch.tensor([]).type(torch.int64))
        b = NestedTensor(torch.tensor([2.0]),
                          torch.tensor([1]))
        c = cat([a, b])
        assert (c.tensor == torch.tensor([2.0])).all()
        assert (c.mask == torch.tensor([1])).all()

    def test_single_element_lists():
        # Concating a single element list and a single element list
        a = NestedTensor(torch.tensor([3.0]),
                          torch.tensor([1]))
        b = NestedTensor(torch.tensor([2.0]),
                          torch.tensor([1]))
        c = cat([a, b])
        assert (c.tensor == torch.tensor([3.0, 2.0])).all()
        assert (c.mask == torch.tensor([1, 1])).all()

    def test_empty_list_lists():
        # Concating a list of an empty list and a list of an empty list
        a = NestedTensor(torch.tensor([[]]),
                          torch.tensor([[]]).type(torch.int64))
        b = NestedTensor(torch.tensor([[]]),
                          torch.tensor([[]]).type(torch.int64))
        c = cat([a, b])
        assert (c.tensor == torch.tensor([[], []])).all()
        assert (c.mask == torch.tensor([[], []]).type(torch.float)).all()

    def test_empty_list_empty_list_list():
        # Concating an empty list and a list of an empty list
        # must throw an error, because the dimensionality of their
        # entries does not align. NOTE: In this future this might
        # be relaxed! For now this contraints significantly
        # eases development.
        a = NestedTensor(torch.tensor([]),
                          torch.tensor([]).type(torch.int64))
        b = NestedTensor(torch.tensor([[]]),
                          torch.tensor([[]]).type(torch.int64))
        try:
            cat([a, b])
            raise Exception("The entries of an empty list have different"
                            " dimensionality than the entires of a list with"
                            " an empty list")
        except AssertionError:
            pass

    def test_two_element_single_element_lists():
        # Concating a two element list and a single element list
        a = NestedTensor(torch.tensor([3.0, 4.0]),
                          torch.tensor([1, 1]))

        b = NestedTensor(torch.tensor([5.0]),
                          torch.tensor([1]))
        c = cat([a, b])
        assert (c.tensor == torch.tensor([3.0, 4.0, 5.0])).all()
        assert (c.mask == torch.tensor([1, 1, 1])).all()

    test_empty_list_empty_list_list()
    test_empty_list_lists()
    test_empty_list_single_list()
    test_empty_lists()
    test_scalar_and_list()
    test_single_element_lists()
    test_two_element_single_element_lists()
    test_two_scalars()


def test_nested_size():
    a = torch.nestedtensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
    print(a.nested_size())
    na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
    assert a.nested_size() == na


def test_len():
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


# TODO: Add view test. Very rigorous
def test_unbind():
    a = torch.nestedtensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
    print(a)
    print(a.unbind())


def test_equal():
    a1 = torch.nestedtensor([torch.tensor([1, 2]), 
                             torch.tensor([7, 8])])
    a2 = torch.nestedtensor([torch.tensor([1, 2]), 
                             torch.tensor([7, 8])])
    a3 = torch.nestedtensor([torch.tensor([3, 4]), 
                             torch.tensor([5, 6])])
    # Just exercising them until we have __bool__, all() etc.
    print(a1 == a2)
    print(a1 != a3)
    print(a1 != a2)
    print(a1 == a3)

def test_float():
    a1 = torch.nestedtensor([torch.tensor([1, 2]), 
                             torch.tensor([7, 8])])
    a2 = a1.to(torch.float)

    a3 = torch.nestedtensor([torch.tensor([3, 4]), 
                             torch.tensor([5, 6])])
    a4 = a3.to(torch.float)

    print(a2)
    import pdb; pdb.set_trace()


def test_unary():
    a1 = torch.nestedtensor([torch.tensor([1, 2]), 
                             torch.tensor([7, 8])])
    a2 = a1.to(torch.float)

    print("a2")
    print(a2)
    print("--- torch.exp")
    print(torch.exp(a2))
    print(a2)
    print("--- exp")
    print(a2.exp())
    print(a2)
    print("--- exp_")
    print(a2.exp_())
    print(a2)
    print("---")

def test_binary():
    a1 = torch.nestedtensor([torch.tensor([1, 2]), 
                             torch.tensor([7, 8])])
    a2 = a1.to(torch.float)
    a3 = torch.nestedtensor([torch.tensor([3, 4]), 
                             torch.tensor([5, 6])])
    a4 = a3.to(torch.float)

    print("--- binary")
    print("a2")
    print(a2)
    print("a4")
    print(a4)
    print("--- torch.add")
    print(torch.add(a2, a4))
    print("a2")
    print(a2)
    print("a4")
    print(a4)
    print("--- add")
    print(a2.add(a4))
    print("a2")
    print(a2)
    print("a4")
    print(a4)
    print("--- add_")
    print(a2.add_(a4))
    print("a2")
    print(a2)
    print("a4")
    print(a4)
    print("--- binary end")

# TODO: Carefully test reference passing vs. value passing for each function
# TODO: Add more tests for variable length examples
if __name__ == "__main__":
    test_unbind()
    test_nested_size()
    test_equal()
    test_float()
    # test_embedding_monkey()
    # test_nested_cross_entropy_loss()
    # test_nested_linear()
    test_len()
    test_unary()
    test_binary()
    # test_nested_lstm()
