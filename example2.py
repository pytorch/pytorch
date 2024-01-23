import torch
import torch._dynamo


def foo():
    def h(x):
        a = x.cos()
        print(x)
        b = a.cos()
        return b

    def g(x):
        x = x.sin()
        x = h(x)
        # print(x)
        x = x.tan()
        return x

    def f(x):
        j = x.cos()
        # print(j)
        t = g(j)
        x = t.cos()
        return x

    def my_backend(gm, inputs):
        print(gm)
        return gm

    x = torch.randn([2, 2])
    eager = f(x)
    f = torch._dynamo.optimize(my_backend)(f)
    # breakpoint()
    compiled = f(x)
    # breakpoint()
    assert torch.equal(eager, compiled)


foo()
