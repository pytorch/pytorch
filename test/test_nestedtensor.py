import torch.prototypes.nestedtensor as nestedtensor

NestedTensor = nestedtensor.NestedTensor

from torch import cat
from torch import stack
from torch.nn.functional import embedding
import torch

def test_init():
    # This is a scalar. It aligns with Torch.
    # These cannot be concatenated with anything else.
    # They function as a building block for lists.
    a = NestedTensor(torch.tensor(1.0),
                      torch.tensor(1))
    assert a.tensor.dim() == 0
    assert a.mask.dim() == 0
    assert a.tensor.numel() == 1
    assert a.mask.numel() == 1
    assert a.tensor.shape == ()
    assert a.mask.shape == ()


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


def test_constructors():
    a = torch.nestedtensor([torch.tensor(2.0)])


def test_embedding_monkey():
    a = torch.tensor([1, 2, 2])
    b = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    d = torch.tensor([[7, 8, 9], [10, 11, 12], [10, 11, 12]])
    c = embedding(a, b)
    assert (c == d).all()

    a = torch.nestedtensor([torch.tensor([1, 2, 2])])
    b = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    c = embedding(a, b)
    d = torch.nestedtensor([torch.tensor([[7, 8, 9], [10, 11, 12], [10, 11, 12]])])
    assert (c.tensor == d.tensor).all()
    assert (c.mask == d.mask).all()

    a = torch.nestedtensor([torch.tensor([[1, 2], [0, 0]])])
    b = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    c = embedding(a, b)
    d = torch.nestedtensor([torch.tensor([[[7, 8, 9], [10, 11, 12]], [[4, 5, 6], [4, 5, 6]]])])
    assert (c.tensor == d.tensor).all()
    assert (c.mask == d.mask).all()

    return
    a = torch.tensor([[]])
    b = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    c = embedding(a, b)

    a = tensor([[0]])
    b = torch.tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    d = tensor([[[7, 8, 9], [10, 11, 12]], [[4, 5, 6], [4, 5, 6]]])
    c = embedding(a, b)

    # TODO: Deal with (1, 0, 3) shaped empty tensor


def test_nested_linear():
    blinear = torch.nn.Linear(2, 3, bias=False)
    blinear.weight.data.mul_(0).add_(1)
    a1 = torch.nestedtensor([torch.tensor([2.0]), torch.tensor([2.0, 2.0])])
    a1.tensor[0][0] = 3.0
    a2 = torch.nestedtensor([torch.tensor([2.0]), torch.tensor([2.0, 2.0])])
    b1 = blinear(a1)
    b2 = blinear(a2)


def test_nested_lstm():

    def _test_two(bsz, seq_len, ninp, nhid, nlayers):
        dropout = 0.0  # Must be 0, otherwise you get random output
        hidden = (torch.randn(nlayers, bsz, nhid),
                  torch.randn(nlayers, bsz, nhid))
        nested_hidden = (torch.nestedtensor([hidden[0]]),
                         torch.nestedtensor([hidden[1]]))
        # nested_rnn = nestedLSTM(ninp, nhid, nlayers,
        #                           dropout=dropout, batch_first=True)
        rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout,
                            batch_first=True)

        flat_weights = []
        # i = 0
        for idx, p in enumerate(rnn.parameters()):
            x = torch.randn_like(p.data)
            flat_weights.append(x)
            p.data.copy_(x)
        # for idx, p in enumerate(nested_rnn.parameters()):
        #     p.data.copy_(flat_weights[i])
        #     i += 1

        input_tensor = torch.randn(bsz, seq_len, ninp)
        nested_input_tensor = torch.nestedtensor([input_tensor])

        assert (nested_input_tensor.tensor == input_tensor).all()
        assert (nested_hidden[0].tensor == hidden[0]).all()
        assert (nested_hidden[1].tensor == hidden[1]).all()

        result, result_hidden = rnn(input_tensor, hidden)
        nested_result, nested_result_hidden = \
            rnn(nested_input_tensor,
                nested_hidden)
        torch.testing.assert_allclose(nested_result.tensor, result)
        torch.testing.assert_allclose(nested_result_hidden[0].tensor,
                                      result_hidden[0])
        torch.testing.assert_allclose(nested_result_hidden[1].tensor,
                                      result_hidden[1])

    def _test_one_cat(bsz, seq_len, ninp, nhid, nlayers):

        assert bsz == 4
        dropout = 0.0  # Must be 0, otherwise you get random output
        hidden = (torch.randn(nlayers, bsz, nhid),
                  torch.randn(nlayers, bsz, nhid))

        nested_hidden_0 = torch.randn(nlayers, bsz + 1, nhid)
        nested_hidden_1 = torch.randn(nlayers, bsz + 1, nhid)
        nested_hidden_0[:, :2, :] = hidden[0][:, :2, :]
        nested_hidden_0[:, 3:, :] = hidden[0][:, 2:, :]
        nested_hidden_1[:, :2, :] = hidden[1][:, :2, :]
        nested_hidden_1[:, 3:, :] = hidden[1][:, 2:, :]
        nested_hidden = (torch.nestedtensor([torch.tensor(nested_hidden_0)]),
                         torch.nestedtensor([torch.tensor(nested_hidden_1)]))

        rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout,
                            batch_first=True)

        flat_weights = []
        for idx, p in enumerate(rnn.parameters()):
            x = torch.randn_like(p.data)
            flat_weights.append(x)
            p.data.copy_(x)

        input_tensor = torch.randn(bsz, seq_len, ninp)
        nested_input_tensor_0 = torch.nestedtensor([torch.tensor(input_tensor[:2, :, :])])
        nested_input_tensor_1 = torch.nestedtensor([torch.tensor(torch.randn(
            1, seq_len, ninp))])
        nested_input_tensor_2 = torch.nestedtensor([torch.tensor(input_tensor[2:, :, :])])

        nested_input_tensor = cat([nested_input_tensor_0,
                                   nested_input_tensor_1,
                                   nested_input_tensor_2])

        pairs = [(0, 0), (1, 1), (2, 3), (3, 4)]
        for rnn_i, brnn_i in pairs:
            assert (nested_input_tensor.tensor[brnn_i] ==
                    input_tensor[rnn_i]).all()
            for h in [0, 1]:
                assert (nested_hidden[h].tensor[:, brnn_i, :] ==
                        hidden[h][:, rnn_i, :]).all()

        nested_result, nested_result_hidden = \
            rnn(nested_input_tensor,
                nested_hidden)
        result, result_hidden = rnn(input_tensor, hidden)
        for rnn_i, brnn_i in pairs:
            torch.testing.assert_allclose(nested_result.tensor[brnn_i],
                                          result[rnn_i])
            torch.testing.assert_allclose(nested_result_hidden[0].
                                          tensor[:, brnn_i, :],
                                          result_hidden[0][:, rnn_i, :])
            for h in [0, 1]:
                torch.testing.assert_allclose(nested_result_hidden[h].
                                              tensor[:, brnn_i, :],
                                              result_hidden[h][:, rnn_i, :])

    for nlayers in [1, 2, 4, 6]:
        _test_two(1,  1,  1,  1,  nlayers)
        _test_two(10, 1,  1,  1,  nlayers)
        _test_two(10, 10, 1,  1,  nlayers)
        _test_two(10, 10, 10, 1,  nlayers)
        _test_two(10, 10, 10, 10, nlayers)
        _test_two(1,  10, 10, 10, nlayers)
        _test_two(1,  1,  10, 10, nlayers)
        _test_two(1,  1,  1,  10, nlayers)
        _test_one_cat(4, 1,  1,  1, nlayers)
        _test_one_cat(4, 10, 1,  1, nlayers)
        _test_one_cat(4, 10, 10, 1, nlayers)
        _test_one_cat(4, 10, 10, 10, nlayers)
        _test_one_cat(4, 1,  10, 10, nlayers)
        _test_one_cat(4, 1,  1,  10, nlayers)


def test_nested_size():
    a = torch.nestedtensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
    print(a.nested_size())
    na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
    assert a.nested_size() == na


# TODO: Write a test
def test_nested_cross_entropy_loss():
    a = torch.tensor([[0., 4., 3.], [1., 2., 3.]])
    b = torch.tensor([1, 2])
    n = torch.nn.CrossEntropyLoss()
    ba = torch.nestedtensor([torch.tensor(a)])
    bb = torch.nestedtensor([torch.tensor(b)])
    print(n(a, b))
    print(n(ba, bb))
    print("--")
    a = torch.tensor([[0., 4., 3.], [1., 2., 3.]])
    b = torch.tensor([1, 2])
    ba = torch.nestedtensor([torch.tensor([[[0., 4., 3.], [1., 2., 3.]]])])
    bb = torch.nestedtensor([torch.tensor([[1], [2]])])
    print(n(a, b))
    print(n(ba, bb))


# TODO: Write a test
def test_fill_masked():
    a = torch.nestedtensor([torch.tensor([0., 4.]), torch.tensor([1., 2., 3., 4., 1., 0.])])
    a.fill_masked(3)


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


# TODO: Carefully test reference passing vs. value passing for each function
# TODO: Add more tests for variable length examples
if __name__ == "__main__":
    test_init()
    test_cat()
    # TODO: Why does stack and cat behave the same for a list
    # of torch.tensor(scalar)s
    test_constructors()
    test_unbind()
    test_nested_size()
    test_embedding_monkey()
    test_fill_masked()
    test_nested_cross_entropy_loss()
    test_nested_linear()
    test_len()
    import pdb; pdb.set_trace()
    test_nested_lstm()
