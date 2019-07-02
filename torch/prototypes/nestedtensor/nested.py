import torch
import torch.nn.functional as F

orig_embedding = torch.nn.functional.embedding

def embedding_monkey(input, weight, padding_idx=None, max_norm=None,
                     norm_type=2., scale_grad_by_freq=False, sparse=False):
    if isinstance(input, NestedTensor):
        ret_tensors = []
        for tensor in input.tensors:
            ret_tensors.append(orig_embedding(tensor, weight, padding_idx,
                                        max_norm, norm_type, scale_grad_by_freq,
                                        sparse))
        return NestedTensor(ret_tensors)
    else:
        return orig_embedding(input, weight, padding_idx, max_norm, norm_type,
                             scale_grad_by_freq, sparse)
    return ret

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
        # TODO: Empty NestedTensor behavior
        if len(tensors) == 0:
            return self
        self.dim = tensors[0].dim()
        self.layout = tensors[0].layout
        self.device = tensors[0].device
        self.dtype = tensors[0].dtype
        for tensor in tensors:
            assert(self.dim == tensor.dim())
            assert(self.layout == tensor.layout)
            assert(self.device == tensor.device)
            assert(self.dtype == tensor.dtype)
        self.tensors = tensors

    def __getattribute__(self, attr):
        if attr == 'shape':
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
