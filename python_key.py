import torch
import torch._C.key as key

HANDLED_FUNCTIONS = {}
class PythonTensor(object):
    def __init__(self, shape):
        self.value = torch.empty(shape)

    def __repr__(self):
        return f"PythonTensor({tuple(self.value.shape)})"

    def tensor(self):
        return self.value

    def __torch_function__(self, func, types, args=(), kwargs={}):
        args = [i.value if isinstance(i, PythonTensor) else i for i in args]
        print(func, args)
        out = func(*args)
        if isinstance(out, torch.Tensor):
            return PythonTensor(out.shape)
        else:
            return out
x = PythonTensor((5, 5))
def f(x):
    return x*2*3
out = f(key.makePython(x))
# print(torch.jit.trace(torch.vmap(f),(x,y)))
print(out)
# import pdb; pdb.set_trace()