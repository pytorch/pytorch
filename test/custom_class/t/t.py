import torch
torch.ops.load_library("build/t.cpython-37m-darwin.so")

@torch.jit.script
class Bar:
    def __init__(self):
        self.x = 1
        self.y = 2


# @torch.jit.script
def f():
    print(torch._C.Foo().display())

f()
# print(f.graph)
