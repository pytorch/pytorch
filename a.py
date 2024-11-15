import torch
import contextlib


@contextlib.contextmanager
def whoo(x):
    y = x.sin() + x.cos()
    try:
    #     raise ValueError
    # except ValueError:
        yield y
    finally:
        pass


@torch.compile(backend='eager')
def fn(x):
    y = x.sin()
    with whoo(x) as z:
        y += z.cos()
    y += y.atanh()
    return y


x = torch.randn(2)
fn(x)