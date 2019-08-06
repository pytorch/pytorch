import torch
import torch.nn.functional as F

# Set this flag to true, if you want to enable additional verifications.
DEBUG = False

# This implementation is based on NestedTensor 0.0.1
# NOTE: This is experimental code! Don't use this in production!
# RFC: https://github.com/pytorch/pytorch/issues/22169



orig_interpolate = F.interpolate

def interpolate(*args, **kwargs):
    if is_nested_tensor(args[0]):
        # feat_shape = inner_lateral.shape[-2:]
        # inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
        ret_list = []
        input_ = args[0]
        size = kwargs['size']
        # TODO: Implement size parameter
        for i in range(len(input_)):
            ret = F.interpolate(input_._tensors[i].view((1,) + input_._tensors[i].size()),
                          size=size[i],
                          mode="nearest")
            ret_list.append(ret.view(ret.size()[1:]))
        return as_nested_tensor(ret_list)
    else:
        return orig_interpolate(*args, **kwargs)

orig_max_pool2d = torch.max_pool2d

def max_pool2d(*args, **kwargs):
    if is_nested_tensor(args[0]):
        ret = []
        for tensor_ in args[0]._tensors:
            tensor = tensor_.view(*((1,) + tensor_.size()))
            args_ = (tensor,) + args[1:]
            ret_ = orig_max_pool2d(*args_)
            ret.append(ret_.view(*(ret_.size()[1:])))
        return NestedTensor(ret)
    else:
        return orig_max_pool2d(*args, **kwargs)

orig_conv2d = F.conv2d

def conv2d(input, weight, bias, stride, padding, dilation, groups):
    if is_nested_tensor(input):
        ret = []
        for tensor_ in input._tensors:
            tensor = tensor_.view(*((1,) + tensor_.size()))
            ret_ = orig_conv2d(tensor, weight, bias, stride,
                               padding, dilation, groups)
            ret.append(ret_.view(*(ret_.size()[1:])))
        return NestedTensor(ret)
    else:
        return orig_conv2d(input, weight, bias, stride,
                           padding, dilation, groups)
orig_relu = F.relu

def relu(input, inplace=False):
    if is_nested_tensor(input):
        ret = []
        for tensor_ in input._tensors:
            ret.append(orig_relu(tensor_, inplace))
        return NestedTensor(ret)
    else:
        return orig_relu(input, inplace)

def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)

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
            raise ValueError("Pass a list or tuple to construct NestedTensor.")

        nested_tensors = []
        for data_ in data:
            if is_nested_tensor(data_):
                nested_tensors.append(data_.clone().detach())
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

        return NestedTensor(tensors).contiguous()

def as_nested_tensor(data, dtype=None, device=None):
    ret = NestedTensor(data)
    if dtype is not None:
        ret = ret.to(dtype)
    if device is not None:
        ret = ret.to(device)
    return ret

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
    def grad(self):
        grads = [t.grad for t in self._tensors]
        if any(grad is None for grad in grads):
            assert all(grad is None for grad in grads)
            return None
        else:
            return NestedTensor(grads)

    @property
    def data(self):
        return NestedTensor([t.data for t in self._tensors])

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
    def shape(self):
        raise NotImplementedError()

    @property
    def requires_grad(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].requires_grad

    @property
    def grad_fn(self):
        raise NotImplementedError(
            "We don't support grad_fn as a user-facing construct.")

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

    def is_empty(self):
        # This condition can never be true, since we disallow an empty list for now.
        raise ValueError("self._tensors cannot be empty under current constraints.")

    def __apply(self, fn):
        return [fn(tensor) for tensor in self._tensors]

    def nested_size(self):
        return tuple(t.size() for t in self._tensors)

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
            if dim > 0:
                return torch.as_nested_tensor(tuple(t.sum(dim - 1) for t in self._tensors))
            else:
                raise NotImplementedError("Reductions over NestedTensor dimension not defined")

    # Tensor ops
    def detach(self):
        return NestedTensor(self.__apply(lambda x: x.detach()))

    def detach_(self):
        return NestedTensor(self.__apply(lambda x: x.detach_()))

    def clone(self):
        return NestedTensor(self.__apply(lambda x: x.clone()))

    def to(self, *args, **kwargs):
        return NestedTensor(self.__apply(lambda x: x.to(*args, **kwargs)))

    def requires_grad_(self, *args, **kwargs):
        return NestedTensor(self.__apply(lambda x: x.requires_grad_(*args, **kwargs)))

    def backward(self, *args, **kwargs):
        self.__apply(lambda x: x.backward(*args, **kwargs))

    # The overhead on this function is very heavy
    def is_contiguous(self):
        first_data_ptr = self._tensors[0].data_ptr()
        current_offset = 0
        is_cont = hasattr(self, 'buffer_')
        for tensor in self._tensors:
            if not is_cont:
                return False
            test_data_ptr = first_data_ptr + current_offset
            is_cont = is_cont and tensor.data_ptr() == test_data_ptr
            is_cont = is_cont and tensor.is_contiguous()
            current_offset += tensor.numel() * tensor.element_size()
        return is_cont

    def contiguous(self):
        flat_tensors = []
        for tensor in self._tensors:
            flat_tensors.append(tensor.view(-1))
        self.buffer_ = torch.cat(flat_tensors)
        current_offset = 0
        for i in range(len(self._tensors)):
            # This is an unnecessary allocation
            new_tensor = torch.empty_like(self._tensors[i],
                    dtype=self.dtype, layout=self.layout, device=self.device)
            with torch.no_grad():
                new_tensor.set_(self.buffer_.storage(),
                                storage_offset=current_offset,
                                size=self._tensors[i].size(),
                                stride=self._tensors[i].stride())
            new_tensor.requires_grad_(self.requires_grad)
            self._tensors[i] = new_tensor
            current_offset += self._tensors[i].numel()
        return self

    def numel(self):
        all_numel = 0
        for tensor in self._tensors:
            all_numel += tensor.numel()
        return all_numel

    def unbind(self):
        return tuple(self._tensors)
