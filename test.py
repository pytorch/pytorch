import torch
from functorch import make_fx
import custom_ops3
from custom_ops3 import my_sin, my_sin_cos, my_sin_inplace
from library import Function

# =====================================================================
# Test cases
# =====================================================================

# class MyNorm(Function):
#     @staticmethod
#     def forward(point):
#         return {'z': point["x"] * point["y"]}
#
#     @staticmethod
#     def post_forward(ctx, args, result):
#         point, = args
#         ctx.save_for_backward(point['x'], point['y'])
#
#     @staticmethod
#     def backward(ctx, grad):
#         x, y = ctx.saved_tensors
#         grad_z = grad['z']
#         return {'x': grad_z * y * 2, 'y': grad_z * x * 2}
#
#
# x = torch.tensor(2., requires_grad=True)
# y = 1
# p = {'x': x, 'y': y}
# z = MyNorm.apply(p)
# z['z'].backward()
# assert torch.allclose(x.grad, y * 2)
# assert torch.allclose(y.grad, x * 2)

# =====================================================================
# my_sin Basic
x = torch.tensor([0.2, 0.4, 0.5])
y = my_sin(x)
assert torch.allclose(y, x.sin())

# =====================================================================
# my_sin Autograd
x = torch.randn(3, requires_grad=True)
y = my_sin(x)
y.sum().backward()
assert torch.allclose(x.grad, x.cos())

# =====================================================================
# my_sin make_fx
def f(x):
    return my_sin(x)


gm = make_fx(f, tracing_mode="fake")(x)
result = gm.code.strip()
print(result)
expected = """
def forward(self, x_1):
    numpy_sin = torch.ops.mangled2__custom_ops3.numpy_sin.default(x_1);  x_1 = None
    return numpy_sin
""".strip()
assert result == expected

# =====================================================================
# my_sin_cos Basic
x = torch.tensor([0.1, 0.2, 0.3])
y = my_sin_cos(x)
assert torch.allclose(y, x.sin().cos())

# =====================================================================
# my_sin_cos make_fx
def f(x):
    return my_sin_cos(x)


gm = make_fx(f, tracing_mode="fake")(x)
result = gm.code.strip()
print(result)
expected = """
def forward(self, x_1):
    my_sin_cos3 = torch.ops.mangled2__custom_ops2.MySinCos3.default(x_1);  x_1 = None
    getitem = my_sin_cos3[0];  my_sin_cos3 = None
    return getitem
""".strip()
# assert result == expected

x = torch.randn(3)
x_version = x._version
my_sin_inplace(x)
new_x_version = x._version
# TODO: need to fix.
# assert x_version < new_x_version, (x_version, new_x_version)


# @torch.compile(backend="inductor")
# def f(x, y):
#     return my_add(x, y)
#
# x = torch.randn(5, device='cuda')
# y = torch.randn(5, device='cuda')
# z = f(x, y)
# assert torch.allclose(z, x + y)


