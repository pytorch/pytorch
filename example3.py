from functools import partial
import depyf
import torch
import torch._dynamo

def bar(x, **kwargs):
    return x + x

@torch.compile(backend="eager", dynamic=True)
def foo(x, i):

    def inner():
        print("this is a graph_break")
        return op(x)

    op = partial(bar, dim=10)
    x = inner()
    # op = partial(bar, other=10)
    return inner() + x


# with depyf.prepare_debug("./dump_src_dir"):
foo(torch.rand(1), 10)
