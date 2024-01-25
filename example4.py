from functools import partial
import depyf
import torch
import torch._dynamo

def bar(x, **kwargs):
    return x + x

@torch.compile(backend="eager", dynamic=True)
def foo(x, y):
    one = lambda x, y: x + y

    def inner():
        # Force no inlining
        torch._dynamo.graph_break()
        return one(x, y)

    res = inner()
    one = lambda x, y: x - y
    res += inner()
    return res


# with depyf.prepare_debug("./dump_src_dir"):
foo(torch.rand(1), 10)
