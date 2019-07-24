import torch
import torch.nn.functional as F

def conv2d(input, weight, bias, stride, padding, dilation, groups):
    if is_nested_tensor(input):
        ret = []
        for tensor_ in input._tensors:
            tensor = tensor_.view(*((1,) + tensor_.size()))
            ret_ = F.conv2d(tensor, weight, bias, stride,
                               padding, dilation, groups)
            ret.append(ret_.view(*(ret_.size()[1:])))
        return NestedTensor(ret)
    else:
        return F.conv2d(input, weight, bias, stride,
                           padding, dilation, groups)

def relu(input, inplace=False):
    if is_nested_tensor(input):
        ret = []
        for tensor_ in input._tensors:
            ret.append(F.relu(tensor_, inplace))
        return NestedTensor(ret)
    else:
        return orig_relu(input, inplace)
