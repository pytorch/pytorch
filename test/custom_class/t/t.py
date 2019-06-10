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
def f(x):
    val = torch._C.Foo(5, 3)
    val2 = torch._C.Foo(100, 0)
    print(val, val2)
    val.display()
    print(val.add(3))

print(f.graph)
f(torch.randn(32, 32))
