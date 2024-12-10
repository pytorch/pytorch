# import functools
# import torch


# def decorator(func):
#     # @functools.wraps(func)
#     def helper(*args):
#         return func(*args)
#     return helper


# def g(x):
#     @decorator
#     def h():
#         return x * 100
#     return h


# def run(h):
#     return h()


# @torch.compile(fullgraph=True)
# def fn(x):
#     h = g(x)
#     return run(h)


# x = torch.randn(1)
# y = fn(x)
# print(x, y)

def chain(*iterables):
    for iterable in iterables:
        yield from iterable

a = [1, 2]
b = [3, 4]

c = chain(a, b)
print(c)