import torch
torch.ops.load_library("build/t.cpython-37m-darwin.so")

# @torch.jit.script
# class Bar:
#     def __init__(self):
#         self.x = 1
#         self.y = 2


print(dir(torch._C))
print(torch._C.Foo())
# @torch.jit.script
def f():
    print(torch.ops.my_ops.warp_perspective(torch.randn(5, 5)))
    print(torch.ops.my_class.Foo())

print(f.graph)
f()
