import torch
import torch._C.key as key

HANDLED_FUNCTIONS = {}
class PythonTensor(torch.Tensor):
    def __init__(self, shape):
        self.value = torch.empty(shape)

    def __repr__(self):
        return f"PythonTensor({tuple(self.value.shape)})"

    def tensor(self):
        return self.value
    func_mapping = {
        'aten::_unsafe_view': torch.Tensor.view,
    }
    def __torch_function__(self, func, types, args=(), kwargs={}):
        print(func)
        # if isinstance(func, str):
        #     if func in self.func_mapping:
        #         func = self.func_mapping[func]
        #     else:
        #         import pdb; pdb.set_trace()
        out = kwargs['val']
        if isinstance(out, torch.Tensor):
            return PythonTensor(out.shape)
        else:
            return out

x = PythonTensor((5, 5))
# def f(x):
#     return x*2
def f(x):
    out = (x*2).sum()
    out.backward()
    # return x.view((25,))
    # return torch.dot(x, torch.ones(5))
grad_x = key.addKey(x)
grad_x.requires_grad = True
out = f(grad_x)
# print(torch.vmap(f)(key.addKey(x)))
# print(torch.jit.trace(torch.vmap(f),(x)))
# print(key.removeKey(out))
# import pdb; pdb.set_trace()