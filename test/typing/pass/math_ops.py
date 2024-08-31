# flake8: noqa
import math

import torch


a = torch.randn(4)
b = torch.randn(4)
t = torch.tensor([-1, -2, 3], dtype=torch.int8)

# abs/absolute
torch.abs(torch.tensor([-1, -2, 3]))
torch.absolute(torch.tensor([-1, -2, 3]))

# acos/arccos
torch.acos(a)
torch.arccos(a)

# acosh/arccosh
torch.acosh(a.uniform_(1, 2))

# add
torch.add(a, 20)
torch.add(a, torch.randn(4, 1), alpha=10)
torch.add(a + 1j, 20 + 1j)
torch.add(a + 1j, 20, alpha=1j)

# addcdiv
torch.addcdiv(torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1)

# addcmul
torch.addcmul(torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1)

# angle
torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])) * 180 / 3.14159

# asin/arcsin
torch.asin(a)
torch.arcsin(a)

# asinh/arcsinh
torch.asinh(a)
torch.arcsinh(a)

# atan/arctan
torch.atan(a)
torch.arctan(a)

# atanh/arctanh
torch.atanh(a.uniform_(-1, 1))
torch.arctanh(a.uniform_(-1, 1))

# atan2
torch.atan2(a, a)

# bitwise_not
torch.bitwise_not(t)

# bitwise_and
torch.bitwise_and(t, torch.tensor([1, 0, 3], dtype=torch.int8))
torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))

# bitwise_or
torch.bitwise_or(t, torch.tensor([1, 0, 3], dtype=torch.int8))
torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))

# bitwise_xor
torch.bitwise_xor(t, torch.tensor([1, 0, 3], dtype=torch.int8))

# ceil
torch.ceil(a)

# clamp/clip
torch.clamp(a, min=-0.5, max=0.5)
torch.clamp(a, min=0.5)
torch.clamp(a, max=0.5)
torch.clip(a, min=-0.5, max=0.5)

# conj
torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))

# copysign
torch.copysign(a, 1)
torch.copysign(a, b)

# cos
torch.cos(a)

# cosh
torch.cosh(a)

# deg2rad
torch.deg2rad(torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]]))

# div/divide/true_divide
x = torch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637])
torch.div(x, 0.5)
p = torch.tensor(
    [
        [-0.3711, -1.9353, -0.4605, -0.2917],
        [0.1815, -1.0111, 0.9805, -1.5923],
        [0.1062, 1.4581, 0.7759, -1.2344],
        [-0.1830, -0.0313, 1.1908, -1.4757],
    ]
)
q = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
torch.div(p, q)
torch.divide(p, q, rounding_mode="trunc")
torch.divide(p, q, rounding_mode="floor")

# digamma
torch.digamma(torch.tensor([1, 0.5]))

# erf
torch.erf(torch.tensor([0, -1.0, 10.0]))

# erfc
torch.erfc(torch.tensor([0, -1.0, 10.0]))

# erfinv
torch.erfinv(torch.tensor([0, 0.5, -1.0]))

# exp
torch.exp(torch.tensor([0, math.log(2.0)]))

# exp2
torch.exp2(torch.tensor([0, math.log2(2.0), 3, 4]))

# expm1
torch.expm1(torch.tensor([0, math.log(2.0)]))

# fake_quantize_per_channel_affine
x = torch.randn(2, 2, 2)
scales = (torch.randn(2) + 1) * 0.05
zero_points = torch.zeros(2).to(torch.long)
torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)

# fake_quantize_per_tensor_affine
torch.fake_quantize_per_tensor_affine(a, 0.1, 0, 0, 255)

# float_power
torch.float_power(torch.randint(10, (4,)), 2)
torch.float_power(torch.arange(1, 5), torch.tensor([2, -3, 4, -5]))

# floor
torch.floor(a)

# floor_divide
torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))
torch.floor_divide(torch.tensor([4.0, 3.0]), 1.4)

# fmod
torch.fmod(torch.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
torch.fmod(torch.tensor([1, 2, 3, 4, 5]), 1.5)

# frac
torch.frac(torch.tensor([1, 2.5, -3.2]))

# imag
torch.randn(4, dtype=torch.cfloat).imag

# ldexp
torch.ldexp(torch.tensor([1.0]), torch.tensor([1]))
torch.ldexp(torch.tensor([1.0]), torch.tensor([1, 2, 3, 4]))

# lerp
start = torch.arange(1.0, 5.0)
end = torch.empty(4).fill_(10)
torch.lerp(start, end, 0.5)
torch.lerp(start, end, torch.full_like(start, 0.5))

# lgamma
torch.lgamma(torch.arange(0.5, 2, 0.5))

# log
torch.log(torch.arange(5) + 10)

# log10
torch.log10(torch.rand(5))

# log1p
torch.log1p(torch.randn(5))

# log2
torch.log2(torch.rand(5))

# logaddexp
torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))

# logaddexp2
torch.logaddexp2(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp2(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp2(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))

# logical_and
torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
r = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
s = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
torch.logical_and(r, s)
torch.logical_and(r.double(), s.double())
torch.logical_and(r.double(), s)
torch.logical_and(r, s, out=torch.empty(4, dtype=torch.bool))

# logical_not
torch.logical_not(torch.tensor([True, False]))
torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
torch.logical_not(torch.tensor([0.0, 1.5, -10.0], dtype=torch.double))
torch.logical_not(
    torch.tensor([0.0, 1.0, -10.0], dtype=torch.double),
    out=torch.empty(3, dtype=torch.int16),
)

# logical_or
torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
torch.logical_or(r, s)
torch.logical_or(r.double(), s.double())
torch.logical_or(r.double(), s)
torch.logical_or(r, s, out=torch.empty(4, dtype=torch.bool))

# logical_xor
torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
torch.logical_xor(r, s)
torch.logical_xor(r.double(), s.double())
torch.logical_xor(r.double(), s)
torch.logical_xor(r, s, out=torch.empty(4, dtype=torch.bool))

# logit
torch.logit(torch.rand(5), eps=1e-6)

# hypot
torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0]))

# i0
torch.i0(torch.arange(5, dtype=torch.float32))

# igamma/igammac
a1 = torch.tensor([4.0])
a2 = torch.tensor([3.0, 4.0, 5.0])
torch.igamma(a1, a2)
torch.igammac(a1, a2)

# mul/multiply
torch.mul(torch.randn(3), 100)
torch.multiply(torch.randn(4, 1), torch.randn(1, 4))
torch.mul(torch.randn(3) + 1j, 100 + 1j)

# mvlgamma
torch.mvlgamma(torch.empty(2, 3).uniform_(1, 2), 2)

# nan_to_num
w = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
torch.nan_to_num(x)
torch.nan_to_num(x, nan=2.0)
torch.nan_to_num(x, nan=2.0, posinf=1.0)

# neg/negative
torch.neg(torch.randn(5))

# nextafter
eps = torch.finfo(torch.float32).eps
torch.nextafter(torch.tensor([1, 2]), torch.tensor([2, 1])) == torch.tensor(
    [eps + 1, 2 - eps]
)

# polygamma
torch.polygamma(1, torch.tensor([1, 0.5]))
torch.polygamma(2, torch.tensor([1, 0.5]))
torch.polygamma(3, torch.tensor([1, 0.5]))
torch.polygamma(4, torch.tensor([1, 0.5]))

# pow
torch.pow(a, 2)
torch.pow(torch.arange(1.0, 5.0), torch.arange(1.0, 5.0))

# rad2deg
torch.rad2deg(torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]]))

# real
torch.randn(4, dtype=torch.cfloat).real

# reciprocal
torch.reciprocal(a)

# remainder
torch.remainder(torch.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
torch.remainder(torch.tensor([1, 2, 3, 4, 5]), 1.5)

# round
torch.round(a)

# rsqrt
torch.rsqrt(a)

# sigmoid
torch.sigmoid(a)

# sign
torch.sign(torch.tensor([0.7, -1.2, 0.0, 2.3]))

# sgn
torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j]).sgn()

# signbit
torch.signbit(torch.tensor([0.7, -1.2, 0.0, 2.3]))

# sin
torch.sin(a)

# sinc
torch.sinc(a)

# sinh
torch.sinh(a)

# sqrt
torch.sqrt(a)

# square
torch.square(a)

# sub/subtract
torch.sub(torch.tensor((1, 2)), torch.tensor((0, 1)), alpha=2)
torch.sub(torch.tensor((1j, 2j)), 1j, alpha=2)
torch.sub(torch.tensor((1j, 2j)), 10, alpha=2j)

# tan
torch.tan(a)

# tanh
torch.tanh(a)

# trunc/fix
torch.trunc(a)

# xlogy
f = torch.zeros(
    5,
)
g = torch.tensor([-1, 0, 1, float("inf"), float("nan")])
torch.xlogy(f, g)

f = torch.tensor([1, 2, 3])
g = torch.tensor([3, 2, 1])
torch.xlogy(f, g)
torch.xlogy(f, 4)
torch.xlogy(2, g)
