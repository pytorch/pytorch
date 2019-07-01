import torch
import torch.nn.functional as F

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
        dim = obj[0].dim()
        layout = obj[0].layout
        device = obj[0].device
        for obj_ in obj:
            assert(dim == obj_.dim())
            assert(layout == obj_.layout)
            assert(device == obj_.device)
        tensors = []
        for obj_ in obj:
            tensors.append(obj_.clone().detach())
        return NestedTensor(tensors)


class NestedTensor():
    def __init__(self, tensors):
        for tensor in tensors:
            assert torch.is_tensor(tensor)
        self.tensors = tensors

    def __getattribute__(self, attr):
        if attr == 'shape':
            raise NotImplementedError()
        if attr == 'dtype':
            raise NotImplementedError()
        return super().__getattribute__(attr)

    def __len__(self):
        return len(self.tensors)

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

    # Tensor ops
    def detach(self):
        return __loop__apply(lambda x: x.detach())

    def backward(self):
        for tensor in self.tensors:
            tensor.backward()

    def clone(self):
        return __loop__apply(lambda x: x.clone())

    def type(self, dtype):
        return __loop__apply(lambda x: x.type(dtype))

    def to(self, device):
        return __loop__apply(lambda x: x.to(device))

    def unbind(self):
        return tuple(self.tensors)
