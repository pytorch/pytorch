import torch
torch.ops.load_library("build/t.cpython-37m-darwin.so")
# print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32)))

@torch.jit.script
class Bar:
    def __init__(self):
        self.x = 1
        self.y = 2

    # def __init__(self, x_, y_):
    #     # type: (int, int) -> None
    #     self.x = x_
    #     self.y = y_

    def display(self):
        print(self.x, self.y)

@torch.jit.script
def f():
    val = torch.classes.Foo(5, 3)
    print(val)
    print(val)
    val.display()
    # # val2 = Bar(100, 0)
    val2 = torch.classes.Foo(100, 0)
    val.display()
    val2.display()
    # # val3 = val.combine(val2)
    # # val3.display()
    # return val

print(f())
print(f.graph)
