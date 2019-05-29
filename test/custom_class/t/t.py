import torch
torch.ops.load_library("build/libt.dylib")

@torch.jit.script
class Bar:
    def __init__(self):
        self.x = 1
        self.y = 2


@torch.jit.script
def f():
    a = Foo()
    print(a)

print(f.graph)
f()
