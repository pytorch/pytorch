import torch


o = ValueError('abcd')
# o = object()


def bar(e = o):
    return e


@torch.compile(backend="eager", fullgraph=True)
def foo(x):
    # if o == bar(o):
    if o is bar(o):
        return x + 1
    else:
        raise RuntimeError('error')

x = torch.randn(2)
y = foo(x)
print(y)