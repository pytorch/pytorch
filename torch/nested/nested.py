import torch
import torch.nn.functional as F

from . import masking

# Set this flag to true, if you want to enable additional verifications.
DEBUG = False

# This implementation is based on NestedTensor 0.0.1
# NOTE: This is experimental code! Don't use this in production!
# RFC: https://github.com/pytorch/pytorch/issues/22169


orig_cat = torch.cat


def cat(*args, **kwargs):
    if is_nested_tensor(args[0][0]):
        # Assuming 1 level of nesting
        dim = kwargs.get('dim', None) - 1
        assert 'out' not in kwargs
        ret = []
        all_tensors = list(args[0][i]._tensors for i in range(len(args[0])))
        for tensors in zip(*all_tensors):
            ret.append(orig_cat(tensors, dim=dim))
        return NestedTensor(ret)
    else:
        return orig_cat(*args, **kwargs)


orig_mv = torch.mv


def mv(*args, **kwargs):
    if is_nested_tensor(args[0]):
        # Assuming 1 level of nesting
        ret = []
        for tensor1, tensor2 in zip(args[0]._tensors, args[1]._tensors):
            ret.append(orig_mv(tensor1, tensor2))
        return NestedTensor(ret)
    else:
        return orig_mv(*args, **kwargs)


def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


# Given a tensor of size N x T x D and mask of size N x T
# return a NestedTensor of nested size ((t_1 x D), ..., (t_N x D))
# If mask[i][j] is 1, tensor[i][j] is an included Vector
def tensor_mask_to_nested_tensor(tensor, mask):
    if tensor.dim() != 3:
        raise NotImplementedError("Only support tensor arguments of dimension 3")
    if mask.dim() == 1:
        raise NotImplementedError("Not implemented for masks of dimension 1 yet.")
    if mask.dim() == 2:
        matrices = tensor.unbind()
        lengths = list(map(sum, mask.unbind()))
        return nested_tensor([matrices[i][:lengths[i]] for i in range(len(lengths))])
    raise NotImplementedError("Not implemented for masks of dimension 3 or more yet.")

# Arguments match torch.tensor


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if is_nested_tensor(data):
        # This is consistent with torch.tensor(torch.Tensor)
        # but errors out.
        raise ValueError("To copy construct from a NestedTensor, "
                         "use sourceTensor.clone().detach() or "
                         "sourceTensor.clone().detach().requires_grad_(True), "
                         "rather than torch.tensor(sourceTensor).")
    elif torch.is_tensor(data):
        # The user has the right to expect a NestedTensor from this
        # function, but we can't meaningfully provide one if passed a Tensor
        raise ValueError("Can't construct a NestedTensor from a Tensor")
    else:
        if not (isinstance(data, list) or isinstance(data, tuple)):
            raise ValueError("Pass a list or tuple to construct a NestedTensor. Got {} instead.".format(type(data)))

        nested_tensors = []
        for data_ in data:
            if is_nested_tensor(data_):
                nested_tensors.append(data_.clone().detach())

        if len(nested_tensors) == 0:
            for data_ in data:
                if isinstance(data_, list) or isinstance(data_, tuple):
                    nested_tensors.append(nested_tensor(data_))

        if len(nested_tensors) > 0:
            if len(nested_tensors) != len(data):
                raise ValueError("All entries of the passed list must either be Tensors or NestedTensors")
            return NestedTensor(nested_tensors)

        for data_ in data:
            if not torch.is_tensor(data_):
                raise ValueError("Each element of the tuple or list must "
                                 "be a torch.Tensor")
        tensors = []
        for data_ in data:
            # torch.tensor copies on construction
            new_data = data_.clone().detach()
            new_data = new_data.to(dtype=dtype, device=device)
            new_data = new_data.requires_grad_(requires_grad)
            if pin_memory:
                new_data = new_data.pin_memory()
            tensors.append(new_data)

        return NestedTensor(tensors)


def as_nested_tensor(data, dtype=None, device=None):
    ret = NestedTensor(data)
    if dtype is not None:
        ret = ret.to(dtype)
    if device is not None:
        ret = ret.to(device)
    return ret


def _nested_apply_return(f):
    def decorator(self, *args, **kwargs):
        if not(torch.is_tensor(self) or is_nested_tensor(self)):
            raise ValueError("First argument must be Tensor or NestedTensor")
        if self.nested_dim == 1:
            return f(self, *args, **kwargs)
        else:
            components = self.unbind()
            if not all(components[0].nested_dim == component.nested_dim for component in components):
                raise ValueError("All NestedTensors must have the same nested dimension")
            return NestedTensor([f(component, *args, **kwargs) for component in components])
    return decorator


def _nested_apply(f):
    def decorator(self, *args, **kwargs):
        if not(torch.is_tensor(self) or is_nested_tensor(self)):
            raise ValueError("First argument must be Tensor or NestedTensor")
        if self.nested_dim == 1:
            f(self, *args, **kwargs)
        else:
            for component in self.unbind():
                f(component, *args, **kwargs)
    return decorator


def _nary_gen(out_dtype=None):
    # Follows signature of torch nary functions
    def _nary(*args, **kwargs):
        func_name = args[0]
        func = args[1]
        inputs = args[2:]
        out = kwargs.get('out', None)

        def _nary_tensors(inputs, out):
            if out is None:
                out_tensor = func(*inputs)
                if out_dtype is not None:
                    out_tensor = out_tensor.to(out_dtype)
                return out_tensor
            else:
                if out_dtype is not None:
                    out = out.to(out_dtype)
                func(*inputs, out=out)
                return out

        if torch.is_tensor(inputs[0]):
            return _nary_tensors(inputs, out)
        else:
            unbound_inputs = [inp.unbind() for inp in inputs]
            result = []
            if out:
                unbound_out = out.unbind()
                for i in range(len(inputs[0])):
                    result.append(_nary(*([func_name, func] + [unbound_inputs[j][i]
                                                               for j in range(len(unbound_inputs))]), out=unbound_out[i]))
            else:
                for i in range(len(inputs[0])):
                    result.append(_nary(*([func_name, func] + [unbound_inputs[j][i]
                                                               for j in range(len(unbound_inputs))])))
            return as_nested_tensor(result)

    return _nary


@_nested_apply
def _verify_tensors(obj):
    tensors = obj.unbind()
    for tensor in tensors:
        assert torch.is_tensor(tensor)
    if len(tensors):
        dim = tensors[0].dim()
        layout = tensors[0].layout
        device = tensors[0].device
        dtype = tensors[0].dtype
        requires_grad = tensors[0].requires_grad
        is_pinned = tensors[0].is_pinned()
        for tensor in tensors:
            if not (dim == tensor.dim() and
                    layout == tensor.layout and
                    device == tensor.device and
                    dtype == tensor.dtype and
                    requires_grad == tensor.requires_grad and
                    is_pinned == tensor.is_pinned()):
                raise ValueError("Each passed Tensor "
                                 "must match in dim, layout, "
                                 "device, dtype and requires_grad")
    else:
        # Carrying around information as member variables vs.
        # checking one entry of the owned Tensors is annoying
        # and error-prone. Carrying around an is_empty attribute
        # to hide the fact that we carry around a list with a
        # single empty Tensor is also annoying and error-prone.
        # Both are not worth it for a minor feature.
        raise ValueError("We do not support empty lists for now.")


class NestedTensor(object):
    # The attributes must match across all constiuents
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # The passed lists of tensors must be non-empty for now.
    #
    # Attributes:
    #     dim
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned
    def __init__(self, tensors):
        self._tensors = tensors
        _verify_tensors(self)

    # TODO: Create level of nesting function from tuples
    @property
    def nested_dim(self):
        if torch.is_tensor(self._tensors[0]):
            return 1
        else:
            return (self._tensors[0]).nested_dim + 1

    @property
    def dim(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].dim

    @property
    def dtype(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].dtype

    @property
    def layout(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].layout

    @property
    def device(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].device

    @property
    def requires_grad(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].requires_grad

    def __len__(self):
        return len(self._tensors)

    def __bool__(self):
        raise NotImplementedError(
            "This has not been covered by NestedTensor 0.0.1")

    def __str__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__str__() + ",\n"
        result += "])"
        return result

    def __repr__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__repr__() + ",\n"
        result += "])"
        return result

    def __iadd__(self, other):
        for i in range(len(self)):
            self._tensors[i].add_(other._tensors[i])
        return self

    def __apply(self, fn):
        return [fn(tensor) for tensor in self._tensors]

    def nested_size(self):
        if self.nested_dim == 1:
            return tuple(t.size() for t in self._tensors)
        else:
            return tuple(t.nested_size() for t in self.unbind())

    def size(self):
        if self.nested_dim == 1:
            sizes = [t.size() for t in self._tensors]
        else:
            sizes = [t.size() for t in self.unbind()]
        if len(sizes) > 0:
            size_0 = sizes[0]
            result_size = list(size_0)
            for i in range(len(size_0)):
                for size in sizes:
                    result_size[i] = size_0[i] if result_size[i] == size[i] else None
            return (len(self),) + tuple(result_size)
        else:
            return ()

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.

    def all(self):
        return all(t.all() for t in self._tensors)

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    def any(self):
        return any(t.any() for t in self._tensors)

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    def sum(self, dim=None):
        # We currently assume len(self._tensors) is always non-zero
        if dim is None:
            return torch.stack(tuple(t.sum() for t in self._tensors)).sum()
        else:
            if dim > self.nested_dim - 1:
                return torch.as_nested_tensor(tuple(t.sum(dim - 1) for t in self._tensors))
            else:
                raise NotImplementedError("Reductions over NestedTensor dimension not defined")

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    # TODO: This needs indicies!!! - not clear
    def argmax(self, dim=None):
        # We currently asmaxe len(self._tensors) is always non-zero
        if dim is None:
            raise NotImplementedError("Full reduction currently not supported")
        else:
            if dim > self.nested_dim - 1:
                return torch.as_nested_tensor(tuple(t.argmax(dim - 1) for t in self._tensors))
            else:
                raise NotImplementedError("Reductions over NestedTensor dimension not defined")

    # Tensor ops
    def detach(self):
        return NestedTensor(self.__apply(lambda x: x.detach()))

    def clone(self):
        return NestedTensor(self.__apply(lambda x: x.clone()))

    def numel(self):
        all_numel = 0
        for tensor in self._tensors:
            all_numel += tensor.numel()
        return all_numel

    def unbind(self):
        return tuple(self._tensors)

    def to_tensor(self):
        if None in self.size():
            raise ValueError("Cannot convert irreguarly shaped NestedTensor into a Tensor")
        else:
            if self.nested_dim == 1:
                return torch.stack(self.unbind())
            else:
                return torch.stack(list(map(lambda x: x.to_tensor(), self.unbind())))

    def to_list(self):
        if self.nested_dim == 1:
            return self._tensors
        else:
            return list(map(lambda x: x.to_list(), self.unbind()))

    def to_tensor_mask(self):
        tensor, mask = masking.make_tensor_mask(self.to_list())
        mask = mask.sum(-1)
        mask = (mask > 0)
        return tensor, mask
