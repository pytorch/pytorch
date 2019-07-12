import torch


def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


def _verify_tensors(tensors):
    for tensor in tensors:
        assert torch.is_tensor(tensor)
    if len(tensors):
        dim = tensors[0].dim()
        layout = tensors[0].layout
        device = tensors[0].device
        dtype = tensors[0].dtype
        for tensor in tensors:
            assert(dim == tensor.dim())
            assert(layout == tensor.layout)
            assert(device == tensor.device)
            assert(dtype == tensor.dtype)


def is_contiguous_tensors(tensors):
    assert len(tensors)
    first_data_ptr = tensors[0].data_ptr()
    current_offset = 0
    is_cont = True
    for tensor in tensors:
        test_data_ptr = first_data_ptr + current_offset
        is_cont = is_cont and tensor.data_ptr() == test_data_ptr
        current_offset += tensor.numel() * tensor.element_size()
    return is_cont


def make_contiguous_tensors(tensors):
    _verify_tensors(tensors)
    assert len(tensors)
    dtype = tensors[0].dtype
    device = tensors[0].device
    all_numel = 0
    for tensor in tensors:
        all_numel += tensor.numel()
    memory = tensor.new_empty((all_numel,), dtype=dtype, device=device)
    current_numel = 0
    new_tensors = []
    for tensor in tensors:
        new_tensor = memory.narrow(0, current_numel, tensor.numel())
        new_tensor = new_tensor.view(tensor.shape)
        new_tensor.copy_(tensor)
        new_tensors.append(new_tensor)
        current_numel += tensor.numel()
    assert is_contiguous_tensors(new_tensors)
    return new_tensors


def make_nested_tensor(obj):
    if is_nested_tensor(obj):
        return obj.clone().detach()
    elif torch.is_tensor(obj):
        return obj.clone().detach()
    else:
        assert isinstance(obj, list)
        if len(obj) == 0:
            return NestedTensor([])
        _verify_tensors(obj)
        return NestedTensor(make_contiguous_tensors(obj))


class NestedTensor():
    def __init__(self, tensors):
        _verify_tensors(tensors)
        self.tensors = tensors
        reference_tensor = torch.Tensor([])
        if len(tensors):
            reference_tensor = tensors[0]
        self.dim = reference_tensor.dim()
        self.layout = reference_tensor.layout
        self.device = reference_tensor.device
        self.dtype = reference_tensor.dtype

    def __getattribute__(self, attr):
        if attr == 'shape':
            raise NotImplementedError()
        return super().__getattribute__(attr)

    def __len__(self):
        return len(self.tensors)

    def __bool__(self):
        # Explicitly punt on this until we have fully
        # specificed reduction semantics.
        raise NotImplementedError()

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
        return len(self.tensors) == 0

    def __loop__apply(self, fn):
        tensors = []
        for tensor in self.tensors:
            tensors.append(fn(tensor))
        return NestedTensor(tensors)

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

    # Tensor ops
    def detach(self):
        return self.__loop__apply(lambda x: x.detach())

    def backward(self):
        for tensor in self.tensors:
            tensor.backward()

    def clone(self):
        return self.__loop__apply(lambda x: x.clone())

    def type(self, dtype):
        return self.__loop__apply(lambda x: x.type(dtype))

    def to(self, device):
        return self.__loop__apply(lambda x: x.to(device))

    def unbind(self):
        return tuple(self.tensors)
