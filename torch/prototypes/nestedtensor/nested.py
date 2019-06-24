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

# TODO: use torch.is_tensor

def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


orig_embedding = torch.nn.functional.embedding
orig_dropout = torch.nn.functional.dropout
orig_cross_entropy = torch.nn.functional.cross_entropy
orig_linear = torch.nn.functional.linear

orig_vf_lstm = torch.nn._VF.lstm
orig_nn_lstm_forward = torch.nn.LSTM.forward

orig_cat = torch.cat
orig_stack = torch.stack

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
        target_nested_size = target_.nested_size()
        max_last_dim = 0
        for s in target_nested_size:
            max_last_dim = max(max_last_dim, s[-1])
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
        bts = len(input.tensor) * [make_nested_tensor_from_tensor(empty_output_tensor)]
        i = 0
        for x in nonzero_input_indicies:
            x = x.item()
            bts[x] = make_nested_tensor_from_tensor(tensors[i][:lengths[i]])
            i += 1

        output = stack(bts)
        hidden = (make_nested_tensor_from_tensor(hidden[0]).pad_to_shape(
            hx[0].tensor.size()),
                  make_nested_tensor_from_tensor(hidden[1]).pad_to_shape(
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
        return make_nested_tensor_from_tensor(torch.tensor([]))

    tensor = orig_cat(tensors)
    mask = orig_cat(masks)
    return NestedTensor(tensor, mask)


# All shapes, but first dim must match
# Empty or not, doesn't matter
def stack(nts_):
    if not (len(nts_) > 0 and is_nested_tensor(nts_[0])):
        return orig_stack(nts_)
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

def make_nested_tensor(obj):
    if is_nested_tensor(obj):
        return NestedTensor(obj.tensor.clone().detach(), obj.mask.clone().detach())
    elif torch.is_tensor(obj):
        return obj.clone().detach()
    elif isinstance(obj, list):
        if len(obj) == 0:
            return NestedTensor(torch.tensor([]), torch.tensor([]))
        for obj_ in obj:
            assert(torch.is_tensor(obj_))
        dim = obj[0].dim()
        layout = obj[0].layout
        device = obj[0].device
        for obj_ in obj:
            assert(dim == obj_.dim())
            assert(layout == obj_.layout)
            assert(device == obj_.device)
        tensors = []
        for obj_ in obj:
            tensors.append(NestedTensor(obj_.clone().detach(),
                torch.ones_like(obj_)))
        return stack(tensors)
    else:
        assert "Given value is of unsupported type"

def make_nested_tensor_from_tensor(tensor):
    assert isinstance(tensor, torch.Tensor)
    return NestedTensor(tensor, torch.ones_like(tensor))


# All shapes must match period.
# Deprecated behavior supports insane catting of non-shape matching
# empty tensors but we don't want that. Don't support this here and
# throw an Error.
def cat(nts):
    if len(nts) > 0 and is_nested_tensor(nts[0]):
        return _cat(nts, True)
    return orig_cat(nts)


def _normalize_mask(mask):
    assert (mask >= 0).sum() == mask.numel()
    mask = (mask > 0)
    return mask


def _check_mask(mask):
    assert (mask.numel() == ((mask == 0).sum() +
                             (mask == 1).sum()))

# Given a mask get the sizes of the constituent Tensors
# TODO: This is a very important function!
# More tests!
def _mask_to_size(mask):
    def __mask_to_size(mask_):
        if mask_.dim() == 0:
            return tuple()
        sum1 = mask_.sum(-1)
        sum2 = int(sum1.view(-1)[0].item())
        sum1 = _normalize_mask(sum1)
        return __mask_to_size(sum1) + (sum2,)
    return __mask_to_size(mask)


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

    def fill_masked(self, value):
        masked_out = self.mask.view(-1) == 0
        if DEBUG_LEVEL > 0:
            assert self.mask.dtype == torch.float
        new_tensor = self.tensor.clone()
        new_tensor = new_tensor.view(-1).masked_fill(masked_out, value)
        new_tensor = new_tensor.view(self.tensor.size())
        return NestedTensor(new_tensor, self.mask)

    # We're constraining this to a list of Tensors
    def __len__(self):
        return len(self.tensor)

    def __str__(self):
        tensors = self.unbind()
        result = "nestedtensor([\n"
        for tensor in tensors:
            result += "  " + tensor.__str__() + "\n"
        result += "])"
        return result

    def __repr__(self):
        tensors = self.unbind()
        result = "nestedtensor([\n"
        for tensor in tensors:
            result += "  " + tensor.__repr__() + "\n"
        result += "])"
        return result

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
        tensors = self.unbind()
        sizes = []
        for tensor in tensors:
            sizes.append(tensor.size())
        return tuple(sizes)

    def unbind(self):
        tensors_ = self.tensor.unbind()
        masks = self.mask.unbind()
        tensors = []
        for i in range(len(tensors_)):
            tensor = tensors_[i]
            mask = masks[i]
            t = _mask_to_size(mask)
            for j in range(len(t)):
                band = t[j]
                tensor = tensor.narrow(j, 0, band)
            tensors.append(tensor)
        return tensors
