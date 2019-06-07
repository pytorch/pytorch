import torch
torch.ops.load_library("build/t.cpython-37m-darwin.so")
# print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32)))

@torch.jit.script
class Bar:
    def __init__(self):
        self.x = 1
        self.y = 2
    def display(self):
        print(self.x, self.y)

@torch.jit.script
def f():
    val = torch._C.Foo()
    val.display()
    print(val)

print(f.graph)
f()
