import torch
from functorch import make_fx
from custom_ops import my_sin, my_sin_cos

# =====================================================================
# Test cases 
# =====================================================================

# =====================================================================
# my_sin Basic
x = torch.randn(3)
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
expected = """
def forward(self, x_1):
    my_sin = torch.ops.mangled2__custom_ops.MySin.default(x_1);  x_1 = None
    return my_sin
""".strip()
assert result == expected

# =====================================================================
# my_sin_cos Basic
x = torch.randn(3)
y = my_sin_cos(x)
assert torch.allclose(y, x.sin().cos())

# =====================================================================
# my_sin_cos make_fx
def f(x):
    return my_sin_cos(x)


gm = make_fx(f, tracing_mode="fake")(x)
result = gm.code.strip()
expected = """
def forward(self, x_1):
    my_sin_cos = torch.ops.mangled2__custom_ops.MySinCos.default(x_1);  x_1 = None
    return my_sin_cos
""".strip()
assert result == expected
