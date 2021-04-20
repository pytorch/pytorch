import torch
import torch._C.key as key

HANDLED_FUNCTIONS = {}
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        import pdb; pdb.set_trace()
        return NotImplemented
x = ScalarTensor(5, 1)
out = key.makePython(x)
out*2
# import pdb; pdb.set_trace()