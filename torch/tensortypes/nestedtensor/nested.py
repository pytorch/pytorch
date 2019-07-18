import torch

# This implementation is based on NestedTensor 0.0.1
# RFC: https://github.com/pytorch/pytorch/issues/22169

def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


def _verify_tensors(tensors):
    if not isinstance(tensors, list):
        raise ValueError("Must pass a list.")
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            raise ValueError("List must consist of Tensors")
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


# Arguments match torch.tensor
def make_nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if is_nested_tensor(data):
        # This is consistent with torch.tensor(torch.Tensor)
        # but errors out.
        raise ValueError("UserWarning: To copy construct from a NestedTensor, "
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
        for data_ in data:
            if not torch.is_tensor(data_):
                raise ValueError("Each element of the tuple or list must "
                                 "be a torch.Tensor")
        ret = as_nestedtensor(data).clone().detach()
        ret = ret.to(dtype)
        ret = ret.to(device)
        ret.requires_grad_(requires_grad)
        if pin_memory:
            ret.pin_memory()
        return ret #.contiguous()

def as_nestedtensor(data, dtype=None, device=None):
    ret = NestedTensor(data)
    if dtype is not None:
        ret = ret.to(dtype)
    if device is not None:
        ret = ret.to(device)
    return ret

class NestedTensor():
    # The attributes must match across all constiuents
    # and default to the empty Tensor's if the given list
    # is empty.
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # Attributes:
    #     dim
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned
    def __init__(self, tensors):
        _verify_tensors(tensors)
        self.tensors = tensors

    def __getattribute__(self, attr):
        if attr == 'shape':
            raise NotImplementedError()
        if attr == 'requires_grad':
            return self.tensors[0].requires_grad
        if attr == 'dtype':
            return self.tensors[0].dtype
        if attr == 'layout':
            return self.tensors[0].layout
        if attr == 'device':
            return self.tensors[0].device
        if attr == 'grad_fn':
            # We're assuming len(self.tensors) is non-zero
            if self.tensors[0].grad_fn is None:
                for i in range(1, len(self.tensors)):
                    assert self.tensors[i].grad_fn is None
                return None

            def create_grad_fn(nested_tensor):
                def _func(*args, **kwargs):
                    for tensor in nested_tensor.tensors:
                        tensor.grad_fn(*args, **kwargs)
                return _func
            return create_grad_fn(self)
        if attr == 'grad':
            grads = []
            for tensor in self.tensors:
                grads.append(tensor.grad)
            print('grads')
            print(grads)
            if all(grad is None for grad in grads):
                return None
            else:
                return NestedTensor(grads)
        if attr == 'data':
            data = []
            for tensor in self.tensors:
                data.append(tensor.data)
            return NestedTensor(data)
        return super().__getattribute__(attr)

    def __len__(self):
        return len(self.tensors)

    def __bool__(self):
        raise NotImplementedError(
            "This has not been covered by NestedTensor 0.0.1")

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

    def is_empty(self):
        # This condition can never be true, since we disallow an empty list for now.
        raise ValueError("self.tensors cannot be empty under current constraints.")

    def __loop__apply(self, fn):
        tensors = []
        for tensor in self.tensors:
            tensors.append(fn(tensor))
        return tensors

    def nested_size(self):
        tensors = self.unbind()
        sizes = []
        for tensor in tensors:
            sizes.append(tensor.size())
        return tuple(sizes)

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    def all(self):
        ret = True
        for tensor in self.tensors:
            ret = ret and tensor.all()
        return ret

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    def any(self):
        ret = False
        for tensor in self.tensors:
            ret = ret or tensor.any()
        return ret

    # TODO: Not covered by RFC! NestedTensor 0.0.2 will talk about reductions.
    def sum(self):
        sums = []
        # We currently assume len(self.tensors) is always non-zero
        for tensor in self.tensors:
            sums.append(tensor.sum())
        return torch.stack(sums).sum()

    # Tensor ops
    def detach(self):
        return NestedTensor(self.__loop__apply(lambda x: x.detach()))

    def detach_(self):
        return NestedTensor(self.__loop__apply(lambda x: x.detach_()))

    def pin_memory(self, *args, **kwargs):
        return NestedTensor(self.__loop__apply(lambda x: x.pin_memory(*args, **kwargs)))

    def backward(self, *args, **kwargs):
        self.tensors = self.__loop__apply(lambda x: x.backward(*args, **kwargs))

    def clone(self):
        return NestedTensor(self.__loop__apply(lambda x: x.clone()))

    def type(self, dtype):
        return NestedTensor(self.__loop__apply(lambda x: x.type(dtype)))

    def to(self, *args, **kwargs):
        return NestedTensor(self.__loop__apply(lambda x: x.to(*args, **kwargs)))

    def requires_grad_(self, *args, **kwargs):
        self.__loop__apply(lambda x: x.requires_grad_(*args, **kwargs))

    def unbind(self):
        return tuple(self.tensors)

    def is_contiguous(self):
        first_data_ptr = self.tensors[0].data_ptr()
        current_offset = 0
        is_cont = True
        for tensor in self.tensors:
            test_data_ptr = first_data_ptr + current_offset
            is_cont = is_cont and tensor.data_ptr() == test_data_ptr
            is_cont = is_cont and tensor.is_contiguous()
            current_offset += tensor.numel() * tensor.element_size()
        return is_cont

    def contiguous(self):
        flat_tensors = []
        for tensor in self.tensors:
            flat_tensors.append(tensor.view(-1))
        self.buffer_ = torch.cat(flat_tensors)
        current_offset = 0
        for i in range(len(self.tensors)):
            new_tensor = torch.empty_like(self.tensors[i], dtype=self.dtype, layout=self.layout, device=self.device)
            new_tensor.set_(self.buffer_.storage(),
                            storage_offset=current_offset,
                            size=self.tensors[i].size(),
                            stride=self.tensors[i].stride())
            new_tensor.copy_(self.tensors[i])
            new_tensor.requires_grad_(self.requires_grad)
            self.tensors[i] = new_tensor
            current_offset += self.tensors[i].numel()
        return self

    def numel(self):
        all_numel = 0
        for tensor in self.tensors:
            all_numel += tensor.numel()
        return all_numel

    # This does allow the user to shoot herself in the foot
    # by modifying shared memory, but then it also doesn't
    # prevent her from being smart.
    def buffer(self):
        return self.buffer_
