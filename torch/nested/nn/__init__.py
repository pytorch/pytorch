import torch
from . import functional

def max_pool2d(*args, **kwargs):
    if is_nested_tensor(args[0]):
        ret = []
        for tensor_ in args[0]._tensors:
            tensor = tensor_.view(*((1,) + tensor_.size()))
            args_ = (tensor,) + args[1:]
            ret_ = torch.max_pool2d(*args_)
            ret.append(ret_.view(*(ret_.size()[1:])))
        return NestedTensor(ret)
    else:
        torch.max_pool2d(*args, **kwargs)
