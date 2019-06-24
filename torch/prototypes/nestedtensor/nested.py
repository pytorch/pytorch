import torch
import torch.nn.functional as F
import numbers

# Use this to check mask consistency
# Add check for 1, 1, 1, ..., 1, 0, 0, 0, ..., 0 format
DEBUG_LEVEL = 0

# TODO: tensor will carry autograd information


# TODO: Might need a squeeze to remove empty lists
# or require all operations to return a squeezed version
# or make operations fail if something will come about as empty


# Only entires at the end of lists can be hidden.


def tensor_scalar(a):
    assert isinstance(a, numbers.Number)
    ret_tensor = torch.tensor(a)
    return make_tensor(ret_tensor)


def tensor_tensor(t):
    assert isinstance(t, torch.Tensor)
    return NestedTensor(t, torch.ones_like(t))


def tensor_list(lst):
    def _tensor_list(l):
        if not isinstance(l, list):
            return make_tensor(l)
        impls = []
        for entry in l:
            impls.append(_tensor_list(entry))
        return stack(impls)

    assert isinstance(lst, list), "Is " + str(type(lst))
    if len(lst) == 0:
        return make_tensor(torch.tensor([]))
    return _tensor_list(lst)


# TODO: Clone value if NestedTensor?
# Standardize on torch behavior here
# Equivalent to torch.tensor?
def make_tensor(value):
    if isinstance(value, NestedTensor):
        return value
    elif isinstance(value, numbers.Number):
        return tensor_scalar(value)
    elif isinstance(value, torch.Tensor):
        return tensor_tensor(value)
    elif isinstance(value, list):
        return tensor_list(value)
    else:
        assert "Given value is of unsupported type"


orig_embedding = torch.nn.functional.embedding
orig_dropout = torch.nn.functional.dropout
orig_cross_entropy = torch.nn.functional.cross_entropy
orig_linear = torch.nn.functional.linear

orig_vf_lstm = torch.nn._VF.lstm
orig_nn_lstm_forward = torch.nn.LSTM.forward


def embedding_monkey(input, weight, padding_idx=None, max_norm=None,
                     norm_type=2., scale_grad_by_freq=False, sparse=False):
    if isinstance(input, NestedTensor):
        ret_tensor = orig_embedding(input.tensor, weight, padding_idx,
                                    max_norm, norm_type, scale_grad_by_freq,
                                    sparse)

        ret_mask = input.mask.clone()
        ret_mask = ret_mask.reshape(tuple(ret_mask.size()) + (1,))
        ret_mask = ret_mask.expand_as(ret_tensor).clone()

        return NestedTensor(ret_tensor, ret_mask)
    else:
        ret = orig_embedding(input, weight, padding_idx, max_norm, norm_type,
                             scale_grad_by_freq, sparse)
    return ret


def dropout_monkey(input, p=0.5, training=True, inplace=False):
    if isinstance(input, NestedTensor):
        ret_tensor = orig_dropout(input.tensor, p, training, inplace)
        assert(ret_tensor.size() == input.tensor.size())
        return NestedTensor(ret_tensor, input.mask.clone())
    else:
        return orig_dropout(input, p, training, inplace)


def cross_entropy_monkey(input_, target_, weight=None, size_average=None,
                         ignore_index=-100, reduce=None, reduction='mean'):
    if isinstance(input_, NestedTensor):
        assert isinstance(target_, NestedTensor)
        # assert input_.tensor.dim() == 3
        # assert target_.tensor.dim() == 2
        target_tensor = target_.fill_masked(ignore_index).tensor
        target_tensor = target_tensor.clone()
        target_tensor = target_tensor.view(-1)
        input_tensor = input_.tensor.view(-1, input_.tensor.size(-1))
        max_last_dim = target_.nested_size().max(-1)[0].item()
        target_tensor = target_.tensor.narrow(-1, 0, max_last_dim)
        target_tensor = target_tensor.contiguous().view(-1)
        # TODO: Off by a scaling factor to match non-batched version
        ret_val = F.cross_entropy(input_tensor,
                                  target_tensor,
                                  weight=weight,
                                  ignore_index=ignore_index,
                                  reduction=reduction)
        return ret_val
    else:
        return orig_cross_entropy(input_, target_, weight, size_average,
                                  ignore_index, reduce, reduction)


def linear_monkey(input_, weight, bias=None):
    if isinstance(input_, NestedTensor):

        input = input_.fill_masked(0)
        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            output = torch.addmm(bias, input.tensor, weight.t())
        else:
            # if input.mask.numel() > input.mask.sum():
            #     import pdb; pdb.set_trace()
            output = input.tensor.matmul(weight.t())
            if bias is not None:
                output += bias

        if DEBUG_LEVEL > 0:
            assert input.mask.dtype == torch.float
        # TODO: This can be dodged by allowing NestedTensor
        # to act as a Parameter and overwriting nn.Linear's
        # weight with the NestedTensor version on initialization
        _mask = torch.ones_like(weight)
        input_mask = input.mask
        # Cast to make use of much faster kernel
        if input.mask.type() != _mask.type():
            input_mask = input_mask.type(_mask.type())
        result_mask = input_mask.matmul(_mask.t())
        result_mask = _normalize_mask(result_mask)
        result = NestedTensor(output, result_mask)
        return result
    else:
        return orig_linear(input, weight, bias)


def nn_lstm_forward_monkey(self, input, hx=None):
    if isinstance(input, NestedTensor):
        assert hx is not None

        if input.is_empty():
            return input.clone(), hx

        def _forward_impl(self, input, hx, batch_sizes, max_batch_size,
                          sorted_indices):
            assert hx is not None
            assert batch_sizes is not None
            hx = self.permute_hidden(hx, sorted_indices)

            self.check_forward_args(input, hx, batch_sizes)
            result = torch.nn._VF.lstm(input, batch_sizes, hx,
                                       self._get_flat_weights(), self.bias,
                                       self.num_layers, self.dropout,
                                       self.training, self.bidirectional)
            output = result[0]
            hidden = result[1:]

            return output, hidden

        def _forward_packed(self, input, hx):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)

            assert len(hx) == 2
            hx_tensors = (hx[0].tensor, hx[1].tensor)

            output, hidden = _forward_impl(self, input, hx_tensors,
                                           batch_sizes, max_batch_size,
                                           sorted_indices)

            output = torch.nn.utils.rnn.get_packed_sequence(output,
                                                            batch_sizes,
                                                            sorted_indices,
                                                            unsorted_indices)
            return output, self.permute_hidden(hidden, unsorted_indices)

        for i in range(len(hx)):
            assert isinstance(hx[i], NestedTensor)
        # TODO: Investigate this as part of rewrite
        assert input.dim() > 0

        input_lengths = input.mask.sum(1)[:, 0]
        nonzero_input_indicies = input_lengths.nonzero()[:, 0]
        nonzero_input_lengths = input_lengths[nonzero_input_indicies]
        nonzero_input_tensor = input.tensor[nonzero_input_indicies]

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            nonzero_input_tensor, nonzero_input_lengths, batch_first=True,
            enforce_sorted=False)

        packed_output, hidden = _forward_packed(self, packed_input, hx)

        tensors, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)

        # NOTE: This is necessary for now because of the per level shape
        # constraint
        empty_output_tensor = torch.tensor([])
        empty_output_shape = (0,) * (input.dim() - 1)
        empty_output_tensor = empty_output_tensor.reshape(empty_output_shape)
        empty_output_tensor = empty_output_tensor.type(input.tensor.type())
        bts = len(input.tensor) * [make_tensor(empty_output_tensor.clone())]
        i = 0
        for x in nonzero_input_indicies:
            x = x.item()
            bts[x] = make_tensor(tensors[i][:lengths[i]])
            i += 1

        output = stack(bts)
        hidden = (make_tensor(hidden[0]).pad_to_shape(
            hx[0].tensor.size()),
                  make_tensor(hidden[1]).pad_to_shape(
            hx[1].tensor.size()))
        return output, hidden
    else:
        return orig_nn_lstm_forward(self, input, hx)


def _cat(nts, skip_empty):
    if len(nts) == 0:
        raise RuntimeError("expected a non-empty list of NestedTensors")

    def _max_shape(tups):
        if len(tups) == 0:
            return ()
        result = len(tups[0]) * [0]
        for i in range(len(tups)):
            for j in range(len(result)):
                result[j] = max(result[j], tups[i][j])
        return tuple(result)

    assert len(nts) > 0, "Can't concatenate less than 1 Tensors"
    # It makes no sense to concatenate a number to something
    for nt in nts:
        assert(nt.dim() > 0)
    # For now we only support the concatenation of
    # NestedTensor with matching dimensions.
    for i, nt in enumerate(nts):
        if i + 1 < len(nts):
            assert(nt.dim() == nts[i + 1].dim())
    max_shape = _max_shape([tuple(nt.tensor.size()) for nt in nts])

    tensors = []
    masks = []
    all_zero_numel = True
    for i in range(len(nts)):
        # Skip empty tensors akin to torch.cat
        if nts[i].is_empty() and skip_empty:
            continue
        all_zero_numel = False
        goal_shape = 90
        if nts[i].dim() > 0:
            goal_shape = (nts[i].tensor.size(0),)
        if len(max_shape) > 1:
            goal_shape = goal_shape + max_shape[1:]
        nts[i] = nts[i].pad_to_shape(goal_shape)
        tensors.append(nts[i].tensor)
        masks.append(nts[i].mask)

    # For torch.concat empty tensors are being ignored
    # unless the entire list consists of empty tensors
    # An empty tensor here is defined as the result from
    # torch.tensor([])
    if all_zero_numel:
        return make_tensor([])

    tensor = torch.cat(tensors)
    mask = torch.cat(masks)
    return NestedTensor(tensor, mask)


# All shapes, but first dim must match
# Empty or not, doesn't matter
def stack(nts_):
    if len(nts_) == 0:
        raise RuntimeError("expected a non-empty list of NestedTensors")
    nts = []
    # Raise dimensionality by 1
    for entry in nts_:
        new_tensor = entry.tensor
        new_shape = (1, ) + tuple(new_tensor.size())
        new_tensor = new_tensor.reshape(new_shape)
        new_mask = entry.mask
        new_mask = new_mask.reshape(new_shape)
        nts.append(NestedTensor(new_tensor, new_mask))
    return _cat(nts, False)


# All shapes must match period.
# Deprecated behavior supports insane catting of non-shape matching
# empty tensors but we don't want that. Don't support this here and
# throw an Error.
def cat(nts):
    return _cat(nts, True)


def _normalize_mask(mask):
    assert (mask >= 0).sum() == mask.numel()
    mask = (mask > 0)
    return mask


def _check_mask(mask):
    assert (mask.numel() == ((mask == 0).sum() +
                             (mask == 1).sum()))


# NOTE: MAJOR CONSTRAINT!
# AT EACH LEVEL, THE DIMENSIONALITY OF ALL ENTRIES
# MUST MATCH!
# THIS MAKES THIS MUCH EASIER (FOR NOW).
# IN THE FUTURE A SHAPE FIELD MIGHT BE ABLE TO ALLOW
# SEMANTICS THAT CLOSER REFLECT NESTED LISTS
# NOTE: The major value this structure represents
# is the ability to represent entries of different lengths.
# Semantically this is like a list of Tensors. We don't support
# deletion. It's a fixed sized list. The entries however may
# be edited.
# NOTE: Another constraint: The number of entries, except of vectors
# must match in a given list.
class NestedTensor():
    # TODO: Introduce a nested shape?
    # XXX: For performance resaons we think about this as a wrapper
    # around these two tensors and don't clone them.
    def __init__(self, tensor, mask):
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        self.tensor = tensor
        # TODO: Use float mask for efficiency?
        # Align with tensor dtype?
        self.mask = mask
        if DEBUG_LEVEL > 0:
            _check_mask(self.mask)

    def __getattribute__(self, attr):
        if attr == 'shape':
            raise NotImplementedError()
        if attr == 'dtype':
            raise NotImplementedError()
        return super().__getattribute__(attr)

    # Requires dim to reduce number of features
    # and make it easier to guarantee correctness
    def squeeze(self, dim):
        tensor = self.tensor.squeeze(dim)
        mask = self.mask.squeeze(dim)
        return NestedTensor(tensor, mask)

    def fill_masked(self, value):
        masked_out = self.mask.view(-1) == 0
        if DEBUG_LEVEL > 0:
            assert self.mask.dtype == torch.float
        new_tensor = self.tensor.clone()
        new_tensor = new_tensor.view(-1).masked_fill(masked_out, value)
        new_tensor = new_tensor.view(self.tensor.size())
        return NestedTensor(new_tensor, self.mask)

    # XXX: This select differs from the regular Tensor
    # narrow in that it doesn't throw an error if you're
    # out of bounds. Entries which are
    # shorter are simply represented by an empty list.
    def narrow(self, dim, start, length):
        assert isinstance(dim, numbers.Number)
        assert isinstance(start, numbers.Number)
        assert isinstance(length, numbers.Number)
        assert length >= 0
        assert start >= 0
        if start + length >= self.tensor.size(dim):
            length = max(0, self.tensor.size(dim) - start)
        # If this is the case length doesn't matter, since
        # we're entirely out of bounds
        if start >= self.tensor.size(dim):
            start = 0
            length = 0
        tensor = torch.narrow(self.tensor, dim, start, length)
        mask = torch.narrow(self.mask, dim, start, length)
        return NestedTensor(tensor, mask)

    # A multi narrow is like applying narrow in the given dim
    # with different lengths for each entry instead of a single length.
    def multi_narrow(self, dim, start, lengths):
        # A non-zero start creates masks that are much harder to deal with
        assert start == 0
        assert dim > 0  # There aren't variable length entries at dim 0
        # A 1-dim Tensor can only be narrow with 1 length
        # That's what narrow is for
        assert self.dim() > 1
        assert isinstance(dim, numbers.Number)
        dim = int(dim)
        assert isinstance(start, numbers.Number)
        start = int(start)
        assert isinstance(lengths, tuple)
        int_lengths = []
        for end in lengths:
            assert isinstance(end, numbers.Number)
            int_lengths.append(int(end))
        lengths = tuple(int_lengths)

        new_mask = self.mask.clone()
        assert self.tensor.size(dim - 1) == len(lengths)
        for i in range(self.tensor.size(dim - 1)):
            new_mask.select(dim - 1, i).fill_(0)
            new_mask.select(dim - 1, i).narrow(0, start, lengths[i]).fill_(1)
        return NestedTensor(self.tensor, new_mask)

    def __len__(self):
        # A 0-dim Tensors can't have a length
        assert self.dim() > 0
        if self.dim() == 1:
            return int(self.mask.sum())
        else:
            # Note: Only vectors can hide entries.
            # Higher dim tensors might still have varibly
            # sized entries, but they can't be hidden.
            # Even [[]] has length 1.
            return len(self.mask)

    def __str__(self):
        return self.tolist().__str__()

    # TODO: Use a tolist that doesn't iterate over all entries
    def __repr__(self):
        return self.tolist().__repr__()

    # Ops specific to NestedTensor

    def pad_to_shape(self, goal_shape):
        def pad_tensor_to_shape(t, goal_shape):
            padd = ()
            tup = tuple(t.size())
            assert(t.dim() == len(goal_shape))
            for i in range(len(tup)):
                padd = (0, goal_shape[i] - tup[i]) + padd
            new_tensor = F.pad(t, padd)
            new_tensor = new_tensor.reshape(goal_shape)
            return new_tensor

        assert self.tensor.size() == self.mask.size()
        new_tensor = pad_tensor_to_shape(self.tensor, goal_shape)
        new_mask = pad_tensor_to_shape(self.mask, goal_shape)
        return NestedTensor(new_tensor, new_mask)

    def is_empty(self):
        return self.mask.sum() == 0

    # Tensor ops

    def detach(self):
        tensor = self.tensor.detach()
        mask = self.mask.detach()
        return NestedTensor(tensor, mask)

    def backward(self):
        self.tensor.backward()

    def mul(self, other):
        tensor = self.tensor.mul(other)
        return NestedTensor(tensor, self.mask)

    def div(self, other):
        tensor = self.tensor.div(other)
        return NestedTensor(tensor, self.mask)

    def clone(self):
        new_tensor = self.tensor.clone()
        new_mask = self.mask.clone()
        return NestedTensor(new_tensor, new_mask)

    def item(self):
        # Nested lists don't have items
        assert self.dim() == 1
        if self.mask.sum() == 1:
            return self.tensor.item()
        else:
            assert self.mask.sum() == 0
            return torch.tensor([]).type(self.tensor.type())

    # There is nothing special about a NestedTensor's dim
    def dim(self):
        return self.tensor.dim()

    def type(self, dtype):
        return_tensor = self.tensor.type(dtype)
        return_mask = self.mask.type(dtype)
        return NestedTensor(return_tensor, return_mask)

    def to(self, device):
        self.tensor = self.tensor.to(device)
        self.mask = self.mask.to(device)
        return self

    # Only lists of vectors will have a varying number of elements.
    def nested_size(self):
        # This isn't defined for a 0-dim tensor, because
        # you can't index into it.
        assert self.dim() > 0
        if self.dim() == 1:
            return torch.tensor(int(self.mask.sum().item()))
        else:
            sizes = []
            for i in range(len(self)):
                data = self.narrow(0, i, 1).squeeze(0)
                sizes.append(data.nested_size())
            return torch.stack(sizes)

    def tolist(self):
        if self.dim() == 0:
            # There can be no such thing as a masked scalar
            assert self.mask.item() == 1
            return self.tensor.item()
        if self.dim() == 1:
            return self.tensor.narrow(0, 0, int(self.mask.sum())).tolist()
        lst = []
        for i in range(self.tensor.size(0)):
            tmp = self.narrow(0, i, 1).squeeze(0).tolist()
            # NOTE: Assumes mask is of form 1, 1, 1, ..., 1, 0, 0, 0, ..., 0
            if tmp is None:
                break
            lst.append(tmp)
        return lst
