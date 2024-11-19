import contextlib

import torch


# @contextlib.contextmanager
# def whoo(x):
#     y = x.sin() + x.cos()
#     try:
#         #     raise ValueError
#         # except ValueError:
#         yield y
#     finally:
#         pass


# @torch.compile(backend="eager")
# def fn(x):
#     y = x.sin()
#     try:
#         with whoo(x) as z:
#             y += z.cos()
#     except ValueError:
#         breakpoint()
#         assert False
#     y += y.atanh()
#     return y


def gen():
    while True:
        yield 1


@torch.compile(backend='eager', fullgraph=True)
def fn(x):
    return list(zip(range(10), gen()))



x = torch.randn(2)
y = fn(x)
print(y)
# import dis
# dis.dis(fn)
