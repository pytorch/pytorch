import torch
from ..backends.thnn import backend as thnn_backend
from torch.autograd import Variable


class Module(object):

    def __init__(self):
        self._backend = thnn_backend

    def __call__(self, *input):
        raise NotImplementedError

    def type(self, type, *forwarded_args):
        # Find all tensors and convert them
        for key, value in self.__dict__.items():
            if isinstance(value, Variable):
                # Variables stored in modules are graph leaves,
                # and we don't want to create copy nodes.
                value.data = value.data.type(type, *forwarded_args)
            elif torch.isTensor(value):
                setattr(self, key, value.type(type, *forwarded_args))
            elif isinstance(value, Module):
                value.type(type, *forwarded_args)
        return self

    def cuda(self, device_id=None):
        import torch.cuda
        if device_id is not None:
            return self.type(torch.cuda.FloatTensor, device_id)
        else:
            return self.type(torch.cuda.FloatTensor)

    def float(self):
        return self.type(torch.FloatTensor)

    def double(self):
        return self.type(torch.DoubleTensor)
