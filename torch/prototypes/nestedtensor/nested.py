import torch


def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


def make_nested_tensor(obj):
    if is_nested_tensor(obj):
        return obj.clone().detach()
    elif torch.is_tensor(obj):
        return obj.clone().detach()
    else:
        assert isinstance(obj, list)
        if len(obj) == 0:
            return NestedTensor([])
        for obj_ in obj:
            assert(torch.is_tensor(obj_))
        tensors = []
        for obj_ in obj:
            tensors.append(obj_.clone().detach())
        return NestedTensor(tensors)


class NestedTensor():
    def __init__(self, tensors):
        for tensor in tensors:
            assert torch.is_tensor(tensor)
        self.tensors = tensors
        if len(tensors):
            self.dim = tensors[0].dim()
            self.layout = tensors[0].layout
            self.device = tensors[0].device
            self.dtype = tensors[0].dtype
            for tensor in tensors:
                if not (self.dim == tensor.dim() and
                        self.layout == tensor.layout and
                        self.device == tensor.device and
                        self.dtype == tensor.dtype):
                    raise ValueError("Each passed Tensor "
                            "must match in dim, layout, "
                            "device and dtype")
        else:
            empty_tensor = torch.Tensor([])
            self.dim = empty_tensor.dim()
            self.layout = empty_tensor.layout
            self.device = empty_tensor.device
            self.dtype = empty_tensor.dtype

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
