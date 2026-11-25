# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math as pymath
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from .triton_compat import (
    _log2,
    builtins_use_semantic_kwarg,
    JITFunction,
    libdevice,
    math,
    tl,
    triton,
)


_T = TypeVar("_T")
_LOG_2_E: tl.constexpr = tl.constexpr(pymath.log2(pymath.e))
_skip_gpu_driver_setup: ContextVar[bool] = ContextVar(
    "_skip_gpu_driver_setup", default=False
)


@contextmanager
def skip_gpu_driver_setup():
    # Scoped no-op for set_driver_to_gpu(). ContextVar keeps nested/thread-local
    # uses isolated.
    token = _skip_gpu_driver_setup.set(True)
    try:
        yield
    finally:
        _skip_gpu_driver_setup.reset(token)


def set_driver_to_cpu():
    driver = triton.runtime.driver
    if backend := triton.backends.backends.get("cpu", None):
        if isinstance(driver.active, backend.driver):
            # Don't re-initialize backend if it is already active
            return
        driver.set_active(backend.driver())
        return
    # This can be a hard error once triton-cpu is merged into fbcode
    warnings.warn(
        "Could not find an active CPU backend. Generated kernels will not be executable!"
    )


def _is_backend_active(name, backend):
    if backend.driver.is_active():
        return True
    # Triton may fail to detect the GPU in subprocess workers when using
    # ctypes-based driver detection (triton-lang/triton#9578). Fall back
    # to torch's own device checks which are more reliable in these environments.
    if name == "nvidia":
        import torch

        return torch.cuda.is_available() and torch.version.hip is None
    if name == "amd":
        import torch

        return torch.cuda.is_available() and torch.version.hip is not None
    return False


def set_driver_to_gpu():
    if _skip_gpu_driver_setup.get():
        return

    driver = triton.runtime.driver
    for name, backend in triton.backends.backends.items():
        if _is_backend_active(name, backend) and name != "cpu":
            # After https://github.com/triton-lang/triton/commit/b844d519bc5e86edf00fe6b3c6c2d1badcd509a4,
            # `driver.active` can be of `LazyProxy` type and the sign of this - `_obj` attribute.
            if (
                isinstance(driver.active, backend.driver)
                or hasattr(driver.active, "_obj")
                and isinstance(driver.active._obj, backend.driver)
            ):
                # Don't re-initialize backend if it is already active
                return
            driver.set_active(backend.driver())
            return
    raise RuntimeError("Could not find an active GPU backend")


def get_backend_options_for_target(target, options=None):
    options = {} if options is None else dict(options)
    backend = triton.compiler.compiler.make_backend(target)
    return backend.parse_options(options).__dict__


def get_backend_options():
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return get_backend_options_for_target(target)


def _is_concrete_backend_option_value(value: object) -> bool:
    import sympy

    import torch

    if isinstance(
        value,
        (
            torch.Tensor,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            sympy.Expr,
        ),
    ):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_concrete_backend_option_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_concrete_backend_option_value(key)
            and _is_concrete_backend_option_value(item)
            for key, item in value.items()
        )
    return True


def try_filter_backend_options_for_target(target, options, kernel_arg_names=()):
    parsed_options = get_backend_options_for_target(target)
    kernel_arg_names = tuple(kernel_arg_names)
    filtered_options = {
        name: value for name, value in options.items() if name in parsed_options
    }
    invalid_options = [
        name
        for name in options
        if name not in parsed_options and name not in kernel_arg_names
    ]
    if invalid_options:
        raise RuntimeError(
            "Triton launch kwargs must be kernel parameters or valid backend options: "
            f"{sorted(invalid_options)!r}."
        )
    dynamic_options = [
        name
        for name, value in filtered_options.items()
        if not _is_concrete_backend_option_value(value)
    ]
    if dynamic_options:
        raise RuntimeError(
            "Triton backend options must be concrete values: "
            f"{sorted(dynamic_options)!r}."
        )
    return filtered_options


def get_constexprs(kernel: JITFunction) -> list[int]:
    return [p.num for p in kernel.params if p.is_constexpr]


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def fp8e4m3fn_to_float32(x):
    x_u32 = x.to(tl.uint32)
    sign = (x_u32 & 0x80) << 24
    exp = (x_u32 >> 3) & 0xF
    mant = x_u32 & 0x7

    normal_bits = sign | ((exp + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    subnormal_abs = mant.to(tl.float32) * 0.001953125
    subnormal_bits = subnormal_abs.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)

    nan = (sign | 0x7FF00000).to(tl.float32, bitcast=True)
    result = tl.where(exp == 0, subnormal, normal)
    return tl.where((exp == 0xF) & (mant == 0x7), nan, result)


@triton.jit
def div_floor_integer(a, b):
    # NOTE: a // b is C division, but we want floor division
    # Based on c10::div_floor_integer
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)


@triton.jit
def remainder_integer(a, b):
    # NOTE: a % b matches C division, not floor division
    remainder = a % b
    return tl.where((remainder != 0) & ((a < 0) != (b < 0)), remainder + b, remainder)


@triton.jit
def pow_integer(base, exponent):
    # Triton has no exact integer pow primitive; use repeated squaring for
    # nonnegative integer exponents so integral scalar pow does not round
    # through libdevice.pow before casting back to int.
    exponent_dtype: tl.constexpr = tl.core.get_int_dtype(
        exponent.dtype.primitive_bitwidth, signed=False
    )
    exp = exponent.to(exponent_dtype)
    result = tl.full(base.shape, 1, base.dtype)
    for _ in tl.static_range(exponent_dtype.primitive_bitwidth):
        result = tl.where((exp & 1) != 0, result * base, result)
        exp = exp >> 1
        base = base * base
    return result


@triton.jit
def _aten_bessel_poly(t, coefficients: tl.constexpr, monic: tl.constexpr = False):
    # Native numerators start at zero; monic denominators start at one.
    # Fold the first numerator coefficient into the initial value.
    result = tl.full(t.shape, 1.0 if monic else coefficients[0], t.dtype)
    for i in tl.static_range(0 if monic else 1, len(coefficients)):
        result = tl.fma(result, t, tl.full(t.shape, coefficients[i], t.dtype))
    return result


@triton.jit
def aten_bessel_j0(x):
    """Cephes bessel_j0_forward, as eager's CUDA jiterator evaluates it.

    Port of ``bessel_j0_string`` in aten/src/ATen/native/cuda/Math.cuh; the
    branches there become selects, so every arm is evaluated for every lane.
    """
    pp: tl.constexpr = (
        7.96936729297347051624e-04,
        8.28352392107440799803e-02,
        1.23953371646414299388e00,
        5.44725003058768775090e00,
        8.74716500199817011941e00,
        5.30324038235394892183e00,
        9.99999999999999997821e-01,
    )
    pq: tl.constexpr = (
        9.24408810558863637013e-04,
        8.56288474354474431428e-02,
        1.25352743901058953537e00,
        5.47097740330417105182e00,
        8.76190883237069594232e00,
        5.30605288235394617618e00,
        1.00000000000000000218e00,
    )
    qp: tl.constexpr = (
        -1.13663838898469149931e-02,
        -1.28252718670509318512e00,
        -1.95539544257735972385e01,
        -9.32060152123768231369e01,
        -1.77681167980488050595e02,
        -1.47077505154951170175e02,
        -5.14105326766599330220e01,
        -6.05014350600728481186e00,
    )
    qq: tl.constexpr = (
        6.43178256118178023184e01,
        8.56430025976980587198e02,
        3.88240183605401609683e03,
        7.24046774195652478189e03,
        5.93072701187316984827e03,
        2.06209331660327847417e03,
        2.42005740240291393179e02,
    )
    rp: tl.constexpr = (
        -4.79443220978201773821e09,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    )
    rq: tl.constexpr = (
        4.99563147152651017219e02,
        1.73785401676374683123e05,
        4.84409658339962045305e07,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    )
    ax = tl.where(x < 0.0, -x, x)
    xx = ax * ax

    small = libdevice.div_rn(
        (xx - tl.full(x.shape, 5.78318596294678452118, x.dtype))
        * (xx - tl.full(x.shape, 3.04712623436620863991e01, x.dtype))
        * _aten_bessel_poly(xx, rp),
        _aten_bessel_poly(xx, rq, True),
    )
    small = tl.where(
        ax < tl.full(x.shape, 0.00001, x.dtype), tl.fma(xx, -0.25, 1.0), small
    )

    t = libdevice.div_rn(tl.full(x.shape, 25.0, x.dtype), xx)
    w = ax - tl.full(x.shape, 0.785398163397448309615660845819875721, x.dtype)
    a = libdevice.div_rn(
        _aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)
    ) * libdevice.cos(w)
    b = libdevice.div_rn(tl.full(x.shape, 5.0, x.dtype), ax) * libdevice.div_rn(
        _aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq, True)
    )
    # ptxas contracts eager's `a - b * sin(w)` into a single FFMA; strict numerics
    # compiles Triton with enable_fp_fusion=False, so spell the fma out.
    large = libdevice.div_rn(
        tl.fma(-b, libdevice.sin(w), a)
        * tl.full(x.shape, 0.797884560802865355879892119868763737, x.dtype),
        libdevice.sqrt_rn(ax),
    )
    return tl.where(ax <= 5.0, small, large)


@triton.jit
def aten_bessel_j1(x):
    """Cephes bessel_j1_forward, as eager's CUDA jiterator evaluates it.

    Port of ``bessel_j1_string`` in aten/src/ATen/native/cuda/Math.cuh; the
    branches there become selects, so every arm is evaluated for every lane.
    """
    pp: tl.constexpr = (
        7.62125616208173112003e-04,
        7.31397056940917570436e-02,
        1.12719608129684925192e00,
        5.11207951146807644818e00,
        8.42404590141772420927e00,
        5.21451598682361504063e00,
        1.00000000000000000254e00,
    )
    pq: tl.constexpr = (
        5.71323128072548699714e-04,
        6.88455908754495404082e-02,
        1.10514232634061696926e00,
        5.07386386128601488557e00,
        8.39985554327604159757e00,
        5.20982848682361821619e00,
        9.99999999999999997461e-01,
    )
    qp: tl.constexpr = (
        5.10862594750176621635e-02,
        4.98213872951233449420e00,
        7.58238284132545283818e01,
        3.66779609360150777800e02,
        7.10856304998926107277e02,
        5.97489612400613639965e02,
        2.11688757100572135698e02,
        2.52070205858023719784e01,
    )
    qq: tl.constexpr = (
        7.42373277035675149943e01,
        1.05644886038262816351e03,
        4.98641058337653607651e03,
        9.56231892404756170795e03,
        7.99704160447350683650e03,
        2.82619278517639096600e03,
        3.36093607810698293419e02,
    )
    rp: tl.constexpr = (
        -8.99971225705559398224e08,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    )
    rq: tl.constexpr = (
        6.20836478118054335476e02,
        2.56987256757748830383e05,
        8.35146791431949253037e07,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    )
    ax = tl.where(x < 0.0, -x, x)
    xx = ax * ax

    small = (
        libdevice.div_rn(_aten_bessel_poly(xx, rp), _aten_bessel_poly(xx, rq, True))
        * ax
        * (xx - tl.full(x.shape, 1.46819706421238932572e01, x.dtype))
        * (xx - tl.full(x.shape, 4.92184563216946036703e01, x.dtype))
    )

    # j1 reduces in 5/x, j0 in 25/(x*x); the two round differently, so keep both.
    u = libdevice.div_rn(tl.full(x.shape, 5.0, x.dtype), ax)
    t = u * u
    w = ax - tl.full(x.shape, 2.356194490192344928846982537459627163, x.dtype)
    a = libdevice.div_rn(
        _aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)
    ) * libdevice.cos(w)
    b = u * libdevice.div_rn(_aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq, True))
    large = libdevice.div_rn(
        tl.fma(-b, libdevice.sin(w), a)
        * tl.full(x.shape, 0.797884560802865355879892119868763737, x.dtype),
        libdevice.sqrt_rn(ax),
    )

    result = tl.where(ax <= 5.0, small, large)
    # Eager reflects odd-symmetrically with PTX neg. Triton's unary minus lowers
    # to `0 - v`, leaving +0.0 positive, so flip the bit directly and keep the
    # subtraction only for the NaN case.
    bits: tl.constexpr = tl.uint64 if x.dtype == tl.float64 else tl.uint32
    sign: tl.constexpr = 0x8000000000000000 if x.dtype == tl.float64 else 0x80000000
    negated = tl.where(
        result == result,
        (result.to(bits, bitcast=True) ^ tl.full(result.shape, sign, bits)).to(
            x.dtype, bitcast=True
        ),
        0.0 - result,
    )
    return tl.where(x < 0.0, negated, result)


@triton.jit
def aten_bessel_y0(x):
    """Cephes bessel_y0_forward, as eager's CUDA jiterator evaluates it.

    Port of ``bessel_y0_string`` in aten/src/ATen/native/cuda/Math.cuh; the
    branches there become selects, so every arm is evaluated for every lane.
    """
    pp: tl.constexpr = (
        7.96936729297347051624e-04,
        8.28352392107440799803e-02,
        1.23953371646414299388e00,
        5.44725003058768775090e00,
        8.74716500199817011941e00,
        5.30324038235394892183e00,
        9.99999999999999997821e-01,
    )
    pq: tl.constexpr = (
        9.24408810558863637013e-04,
        8.56288474354474431428e-02,
        1.25352743901058953537e00,
        5.47097740330417105182e00,
        8.76190883237069594232e00,
        5.30605288235394617618e00,
        1.00000000000000000218e00,
    )
    qp: tl.constexpr = (
        -1.13663838898469149931e-02,
        -1.28252718670509318512e00,
        -1.95539544257735972385e01,
        -9.32060152123768231369e01,
        -1.77681167980488050595e02,
        -1.47077505154951170175e02,
        -5.14105326766599330220e01,
        -6.05014350600728481186e00,
    )
    qq: tl.constexpr = (
        6.43178256118178023184e01,
        8.56430025976980587198e02,
        3.88240183605401609683e03,
        7.24046774195652478189e03,
        5.93072701187316984827e03,
        2.06209331660327847417e03,
        2.42005740240291393179e02,
    )
    yp: tl.constexpr = (
        1.55924367855235737965e04,
        -1.46639295903971606143e07,
        5.43526477051876500413e09,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    )
    yq: tl.constexpr = (
        1.04128353664259848412e03,
        6.26107330137134956842e05,
        2.68919633393814121987e08,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    )
    xx = x * x

    # Math.cuh:1645 writes a bare `NAN;` where it means `return NAN;`, so x < 0
    # reaches this arm and comes back NaN only because log(x) does. Reproducing
    # eager means reproducing that, not the intent. NVVM (not ptxas) contracts
    # the trailing `+ (2/pi * log(x)) * J0(x)`, so spell the fma out.
    small = tl.fma(
        libdevice.log(x)
        * tl.full(x.shape, 0.636619772367581343075535053490057448, x.dtype),
        aten_bessel_j0(x),
        libdevice.div_rn(_aten_bessel_poly(xx, yp), _aten_bessel_poly(xx, yq, True)),
    )
    small = tl.where(x == 0.0, float("-inf"), small)

    t = libdevice.div_rn(tl.full(x.shape, 25.0, x.dtype), xx)
    w = x - tl.full(x.shape, 0.785398163397448309615660845819875721, x.dtype)
    a = libdevice.div_rn(
        _aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)
    ) * libdevice.sin(w)
    b = libdevice.div_rn(tl.full(x.shape, 5.0, x.dtype), x) * libdevice.div_rn(
        _aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq, True)
    )
    large = libdevice.div_rn(
        tl.fma(b, libdevice.cos(w), a)
        * tl.full(x.shape, 0.797884560802865355879892119868763737, x.dtype),
        libdevice.sqrt_rn(x),
    )
    return tl.where(x <= 5.0, small, large)


@triton.jit
def aten_bessel_y1(x):
    """Cephes bessel_y1_forward, as eager's CUDA jiterator evaluates it.

    Port of ``bessel_y1_string`` in aten/src/ATen/native/cuda/Math.cuh; the
    branches there become selects, so every arm is evaluated for every lane.
    """
    pp: tl.constexpr = (
        7.62125616208173112003e-04,
        7.31397056940917570436e-02,
        1.12719608129684925192e00,
        5.11207951146807644818e00,
        8.42404590141772420927e00,
        5.21451598682361504063e00,
        1.00000000000000000254e00,
    )
    pq: tl.constexpr = (
        5.71323128072548699714e-04,
        6.88455908754495404082e-02,
        1.10514232634061696926e00,
        5.07386386128601488557e00,
        8.39985554327604159757e00,
        5.20982848682361821619e00,
        9.99999999999999997461e-01,
    )
    qp: tl.constexpr = (
        5.10862594750176621635e-02,
        4.98213872951233449420e00,
        7.58238284132545283818e01,
        3.66779609360150777800e02,
        7.10856304998926107277e02,
        5.97489612400613639965e02,
        2.11688757100572135698e02,
        2.52070205858023719784e01,
    )
    qq: tl.constexpr = (
        7.42373277035675149943e01,
        1.05644886038262816351e03,
        4.98641058337653607651e03,
        9.56231892404756170795e03,
        7.99704160447350683650e03,
        2.82619278517639096600e03,
        3.36093607810698293419e02,
    )
    yp: tl.constexpr = (
        1.26320474790178026440e09,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    )
    yq: tl.constexpr = (
        5.94301592346128195359e02,
        2.35564092943068577943e05,
        7.34811944459721705660e07,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    )
    xx = x * x

    # Unlike y0, Math.cuh returns NAN for x < 0. FP32 log already yields that
    # encoding; FP64 needs the explicit conversion of the float NAN macro.
    small = tl.fma(
        x,
        libdevice.div_rn(_aten_bessel_poly(xx, yp), _aten_bessel_poly(xx, yq, True)),
        tl.fma(
            libdevice.log(x),
            aten_bessel_j1(x),
            libdevice.div_rn(tl.full(x.shape, -1.0, x.dtype), x),
        )
        * tl.full(x.shape, 0.636619772367581343075535053490057448, x.dtype),
    )
    if x.dtype == tl.float64:
        nan = tl.full(x.shape, 0x7FFFFFFF, tl.uint32).to(tl.float32, bitcast=True)
        small = tl.where(x < 0.0, nan.to(x.dtype), small)
    small = tl.where(x == 0.0, float("-inf"), small)

    u = libdevice.div_rn(tl.full(x.shape, 5.0, x.dtype), x)
    t = u * u
    w = x - tl.full(x.shape, 2.356194490192344928846982537459627163, x.dtype)
    a = libdevice.div_rn(
        _aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)
    ) * libdevice.sin(w)
    b = u * libdevice.div_rn(_aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq, True))
    large = libdevice.div_rn(
        tl.fma(b, libdevice.cos(w), a)
        * tl.full(x.shape, 0.797884560802865355879892119868763737, x.dtype),
        libdevice.sqrt_rn(x),
    )
    return tl.where(x <= 5.0, small, large)


@triton.jit
def aten_hypot(x, y):
    # Match CUDA's FP32 hypot, including its correctly rounded square root.
    abs_mask = tl.full([], 0x7FFFFFFF, tl.uint32)
    x_bits = x.to(tl.uint32, bitcast=True) & abs_mask
    y_bits = y.to(tl.uint32, bitcast=True) & abs_mask
    lo_bits = tl.minimum(x_bits, y_bits)
    hi_bits = tl.maximum(x_bits, y_bits)
    lo = lo_bits.to(tl.float32, bitcast=True)
    hi = hi_bits.to(tl.float32, bitcast=True)

    exponent = hi_bits & tl.full([], 0xFE000000, tl.uint32)
    scale_down_bits = tl.full([], 0x7E800000, tl.uint32) - exponent
    scale_down = scale_down_bits.to(tl.float32, bitcast=True)
    scaled_lo = lo * scale_down
    scaled_hi = hi * scale_down
    lo_square = scaled_lo * scaled_lo
    square_sum = tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[scaled_hi, scaled_hi, lo_square],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    magnitude = tl.sqrt_rn(square_sum)
    scale_up_bits = exponent | tl.full([], 0x00800000, tl.uint32)
    scale_up = scale_up_bits.to(tl.float32, bitcast=True)
    result = magnitude * scale_up

    result = tl.where(lo_bits == 0, hi, result)
    inf_bits = tl.full([], 0x7F800000, tl.uint32)
    nan = abs_mask.to(tl.float32, bitcast=True)
    result = tl.where(hi_bits > inf_bits, nan, result)
    is_inf = (x_bits == inf_bits) | (y_bits == inf_bits)
    return tl.where(is_inf, inf_bits.to(tl.float32, bitcast=True), result)


@triton.jit
def aten_log_ndtr(x):
    sqrt_half = tl.full(x.shape, 0.707106781186547524400844362104849039, x.dtype)
    half = tl.full(x.shape, 0.5, x.dtype)
    t = x * sqrt_half
    log_term = libdevice.log(libdevice.erfcx(-t) * half)
    left = tl.fma(-t, t, log_term)
    right = libdevice.log1p(-libdevice.erfc(t) * half)
    result = tl.where(x < -1.0, left, right)
    negative = tl.full(x.shape, -1.0, x.dtype)
    return tl.where(libdevice.isnan(x), result, libdevice.copysign(result, negative))


@triton.jit
def _aten_erfcx_y100(y100):
    coefficients: tl.constexpr = (
        (
            0.70878032454106438663e-3,
            0.71234091047026302958e-3,
            0.35779077297597742384e-5,
            0.17403143962587937815e-7,
            0.81710660047307788845e-10,
            0.36885022360434957634e-12,
            0.15917038551111111111e-14,
        ),
        (
            0.21479143208285144230e-2,
            0.72686402367379996033e-3,
            0.36843175430938995552e-5,
            0.18071841272149201685e-7,
            0.85496449296040325555e-10,
            0.38852037518534291510e-12,
            0.16868473576888888889e-14,
        ),
        (
            0.36165255935630175090e-2,
            0.74182092323555510862e-3,
            0.37948319957528242260e-5,
            0.18771627021793087350e-7,
            0.89484715122415089123e-10,
            0.40935858517772440862e-12,
            0.17872061464888888889e-14,
        ),
        (
            0.51154983860031979264e-2,
            0.75722840734791660540e-3,
            0.39096425726735703941e-5,
            0.19504168704300468210e-7,
            0.93687503063178993915e-10,
            0.43143925959079664747e-12,
            0.18939926435555555556e-14,
        ),
        (
            0.66457513172673049824e-2,
            0.77310406054447454920e-3,
            0.40289510589399439385e-5,
            0.20271233238288381092e-7,
            0.98117631321709100264e-10,
            0.45484207406017752971e-12,
            0.20076352213333333333e-14,
        ),
        (
            0.82082389970241207883e-2,
            0.78946629611881710721e-3,
            0.41529701552622656574e-5,
            0.21074693344544655714e-7,
            0.10278874108587317989e-9,
            0.47965201390613339638e-12,
            0.21285907413333333333e-14,
        ),
        (
            0.98039537275352193165e-2,
            0.80633440108342840956e-3,
            0.42819241329736982942e-5,
            0.21916534346907168612e-7,
            0.10771535136565470914e-9,
            0.50595972623692822410e-12,
            0.22573462684444444444e-14,
        ),
        (
            0.11433927298290302370e-1,
            0.82372858383196561209e-3,
            0.44160495311765438816e-5,
            0.22798861426211986056e-7,
            0.11291291745879239736e-9,
            0.53386189365816880454e-12,
            0.23944209546666666667e-14,
        ),
        (
            0.13099232878814653979e-1,
            0.84167002467906968214e-3,
            0.45555958988457506002e-5,
            0.23723907357214175198e-7,
            0.11839789326602695603e-9,
            0.56346163067550237877e-12,
            0.25403679644444444444e-14,
        ),
        (
            0.14800987015587535621e-1,
            0.86018092946345943214e-3,
            0.47008265848816866105e-5,
            0.24694040760197315333e-7,
            0.12418779768752299093e-9,
            0.59486890370320261949e-12,
            0.26957764568888888889e-14,
        ),
        (
            0.16540351739394069380e-1,
            0.87928458641241463952e-3,
            0.48520195793001753903e-5,
            0.25711774900881709176e-7,
            0.13030128534230822419e-9,
            0.62820097586874779402e-12,
            0.28612737351111111111e-14,
        ),
        (
            0.18318536789842392647e-1,
            0.89900542647891721692e-3,
            0.50094684089553365810e-5,
            0.26779777074218070482e-7,
            0.13675822186304615566e-9,
            0.66358287745352705725e-12,
            0.30375273884444444444e-14,
        ),
        (
            0.20136801964214276775e-1,
            0.91936908737673676012e-3,
            0.51734830914104276820e-5,
            0.27900878609710432673e-7,
            0.14357976402809042257e-9,
            0.70114790311043728387e-12,
            0.32252476000000000000e-14,
        ),
        (
            0.21996459598282740954e-1,
            0.94040248155366777784e-3,
            0.53443911508041164739e-5,
            0.29078085538049374673e-7,
            0.15078844500329731137e-9,
            0.74103813647499204269e-12,
            0.34251892320000000000e-14,
        ),
        (
            0.23898877187226319502e-1,
            0.96213386835900177540e-3,
            0.55225386998049012752e-5,
            0.30314589961047687059e-7,
            0.15840826497296335264e-9,
            0.78340500472414454395e-12,
            0.36381553564444444445e-14,
        ),
        (
            0.25845480155298518485e-1,
            0.98459293067820123389e-3,
            0.57082915920051843672e-5,
            0.31613782169164830118e-7,
            0.16646478745529630813e-9,
            0.82840985928785407942e-12,
            0.38649975768888888890e-14,
        ),
        (
            0.27837754783474696598e-1,
            0.10078108563256892757e-2,
            0.59020366493792212221e-5,
            0.32979263553246520417e-7,
            0.17498524159268458073e-9,
            0.87622459124842525110e-12,
            0.41066206488888888890e-14,
        ),
        (
            0.29877251304899307550e-1,
            0.10318204245057349310e-2,
            0.61041829697162055093e-5,
            0.34414860359542720579e-7,
            0.18399863072934089607e-9,
            0.92703227366365046533e-12,
            0.43639844053333333334e-14,
        ),
        (
            0.31965587178596443475e-1,
            0.10566560976716574401e-2,
            0.63151633192414586770e-5,
            0.35924638339521924242e-7,
            0.19353584758781174038e-9,
            0.98102783859889264382e-12,
            0.46381060817777777779e-14,
        ),
        (
            0.34104450552588334840e-1,
            0.10823541191350532574e-2,
            0.65354356159553934436e-5,
            0.37512918348533521149e-7,
            0.20362979635817883229e-9,
            0.10384187833037282363e-11,
            0.49300625262222222221e-14,
        ),
        (
            0.36295603928292425716e-1,
            0.11089526167995268200e-2,
            0.67654845095518363577e-5,
            0.39184292949913591646e-7,
            0.21431552202133775150e-9,
            0.10994259106646731797e-11,
            0.52409949102222222221e-14,
        ),
        (
            0.38540888038840509795e-1,
            0.11364917134175420009e-2,
            0.70058230641246312003e-5,
            0.40943644083718586939e-7,
            0.22563034723692881631e-9,
            0.11642841011361992885e-11,
            0.55721092871111111110e-14,
        ),
        (
            0.40842225954785960651e-1,
            0.11650136437945673891e-2,
            0.72569945502343006619e-5,
            0.42796161861855042273e-7,
            0.23761401711005024162e-9,
            0.12332431172381557035e-11,
            0.59246802364444444445e-14,
        ),
        (
            0.43201627431540222422e-1,
            0.11945628793917272199e-2,
            0.75195743532849206263e-5,
            0.44747364553960993492e-7,
            0.25030885216472953674e-9,
            0.13065684400300476484e-11,
            0.63000532853333333334e-14,
        ),
        (
            0.45621193513810471438e-1,
            0.12251862608067529503e-2,
            0.77941720055551920319e-5,
            0.46803119830954460212e-7,
            0.26375990983978426273e-9,
            0.13845421370977119765e-11,
            0.66996477404444444445e-14,
        ),
        (
            0.48103121413299865517e-1,
            0.12569331386432195113e-2,
            0.80814333496367673980e-5,
            0.48969667335682018324e-7,
            0.27801515481905748484e-9,
            0.14674637611609884208e-11,
            0.71249589351111111110e-14,
        ),
        (
            0.50649709676983338501e-1,
            0.12898555233099055810e-2,
            0.83820428414568799654e-5,
            0.51253642652551838659e-7,
            0.29312563849675507232e-9,
            0.15556512782814827846e-11,
            0.75775607822222222221e-14,
        ),
        (
            0.53263363664388864181e-1,
            0.13240082443256975769e-2,
            0.86967260015007658418e-5,
            0.53662102750396795566e-7,
            0.30914568786634796807e-9,
            0.16494420240828493176e-11,
            0.80591079644444444445e-14,
        ),
        (
            0.55946601353500013794e-1,
            0.13594491197408190706e-2,
            0.90262520233016380987e-5,
            0.56202552975056695376e-7,
            0.32613310410503135996e-9,
            0.17491936862246367398e-11,
            0.85713381688888888890e-14,
        ),
        (
            0.58702059496154081813e-1,
            0.13962391363223647892e-2,
            0.93714365487312784270e-5,
            0.58882975670265286526e-7,
            0.34414937110591753387e-9,
            0.18552853109751857859e-11,
            0.91160736711111111110e-14,
        ),
        (
            0.61532500145144778048e-1,
            0.14344426411912015247e-2,
            0.97331446201016809696e-5,
            0.61711860507347175097e-7,
            0.36325987418295300221e-9,
            0.19681183310134518232e-11,
            0.96952238400000000000e-14,
        ),
        (
            0.64440817576653297993e-1,
            0.14741275456383131151e-2,
            0.10112293819576437838e-4,
            0.64698236605933246196e-7,
            0.38353412915303665586e-9,
            0.20881176114385120186e-11,
            0.10310784480000000000e-13,
        ),
        (
            0.67430045633130393282e-1,
            0.15153655418916540370e-2,
            0.10509857606888328667e-4,
            0.67851706529363332855e-7,
            0.40504602194811140006e-9,
            0.22157325110542534469e-11,
            0.10964842115555555556e-13,
        ),
        (
            0.70503365513338850709e-1,
            0.15582323336495709827e-2,
            0.10926868866865231089e-4,
            0.71182482239613507542e-7,
            0.42787405890153386710e-9,
            0.23514379522274416437e-11,
            0.11659571751111111111e-13,
        ),
        (
            0.73664114037944596353e-1,
            0.16028078812438820413e-2,
            0.11364423678778207991e-4,
            0.74701423097423182009e-7,
            0.45210162777476488324e-9,
            0.24957355004088569134e-11,
            0.12397238257777777778e-13,
        ),
        (
            0.76915792420819562379e-1,
            0.16491766623447889354e-2,
            0.11823685320041302169e-4,
            0.78420075993781544386e-7,
            0.47781726956916478925e-9,
            0.26491544403815724749e-11,
            0.13180196462222222222e-13,
        ),
        (
            0.80262075578094612819e-1,
            0.16974279491709504117e-2,
            0.12305888517309891674e-4,
            0.82350717698979042290e-7,
            0.50511496109857113929e-9,
            0.28122528497626897696e-11,
            0.14010889635555555556e-13,
        ),
        (
            0.83706822008980357446e-1,
            0.17476561032212656962e-2,
            0.12812343958540763368e-4,
            0.86506399515036435592e-7,
            0.53409440823869467453e-9,
            0.29856186620887555043e-11,
            0.14891851591111111111e-13,
        ),
        (
            0.87254084284461718231e-1,
            0.17999608886001962327e-2,
            0.13344443080089492218e-4,
            0.90900994316429008631e-7,
            0.56486134972616465316e-9,
            0.31698707080033956934e-11,
            0.15825697795555555556e-13,
        ),
        (
            0.90908120182172748487e-1,
            0.18544478050657699758e-2,
            0.13903663143426120077e-4,
            0.95549246062549906177e-7,
            0.59752787125242054315e-9,
            0.33656597366099099413e-11,
            0.16815130613333333333e-13,
        ),
        (
            0.94673404508075481121e-1,
            0.19112284419887303347e-2,
            0.14491572616545004930e-4,
            0.10046682186333613697e-6,
            0.63221272959791000515e-9,
            0.35736693975589130818e-11,
            0.17862931591111111111e-13,
        ),
        (
            0.98554641648004456555e-1,
            0.19704208544725622126e-2,
            0.15109836875625443935e-4,
            0.10567036667675984067e-6,
            0.66904168640019354565e-9,
            0.37946171850824333014e-11,
            0.18971959040000000000e-13,
        ),
        (
            0.10255677889470089531e0,
            0.20321499629472857418e-2,
            0.15760224242962179564e-4,
            0.11117756071353507391e-6,
            0.70814785110097658502e-9,
            0.40292553276632563925e-11,
            0.20145143075555555556e-13,
        ),
        (
            0.10668502059865093318e0,
            0.20965479776148731610e-2,
            0.16444612377624983565e-4,
            0.11700717962026152749e-6,
            0.74967203250938418991e-9,
            0.42783716186085922176e-11,
            0.21385479360000000000e-13,
        ),
        (
            0.11094484319386444474e0,
            0.21637548491908170841e-2,
            0.17164995035719657111e-4,
            0.12317915750735938089e-6,
            0.79376309831499633734e-9,
            0.45427901763106353914e-11,
            0.22696025653333333333e-13,
        ),
        (
            0.11534201115268804714e0,
            0.22339187474546420375e-2,
            0.17923489217504226813e-4,
            0.12971465288245997681e-6,
            0.84057834180389073587e-9,
            0.48233721206418027227e-11,
            0.24079890062222222222e-13,
        ),
        (
            0.11988259392684094740e0,
            0.23071965691918689601e-2,
            0.18722342718958935446e-4,
            0.13663611754337957520e-6,
            0.89028385488493287005e-9,
            0.51210161569225846701e-11,
            0.25540227111111111111e-13,
        ),
        (
            0.12457298393509812907e0,
            0.23837544771809575380e-2,
            0.19563942105711612475e-4,
            0.14396736847739470782e-6,
            0.94305490646459247016e-9,
            0.54366590583134218096e-11,
            0.27080225920000000000e-13,
        ),
        (
            0.12941991566142438816e0,
            0.24637684719508859484e-2,
            0.20450821127475879816e-4,
            0.15173366280523906622e-6,
            0.99907632506389027739e-9,
            0.57712760311351625221e-11,
            0.28703099555555555556e-13,
        ),
        (
            0.13443048593088696613e0,
            0.25474249981080823877e-2,
            0.21385669591362915223e-4,
            0.15996177579900443030e-6,
            0.10585428844575134013e-8,
            0.61258809536787882989e-11,
            0.30412080142222222222e-13,
        ),
        (
            0.13961217543434561353e0,
            0.26349215871051761416e-2,
            0.22371342712572567744e-4,
            0.16868008199296822247e-6,
            0.11216596910444996246e-8,
            0.65015264753090890662e-11,
            0.32210394506666666666e-13,
        ),
        (
            0.14497287157673800690e0,
            0.27264675383982439814e-2,
            0.23410870961050950197e-4,
            0.17791863939526376477e-6,
            0.11886425714330958106e-8,
            0.68993039665054288034e-11,
            0.34101266222222222221e-13,
        ),
        (
            0.15052089272774618151e0,
            0.28222846410136238008e-2,
            0.24507470422713397006e-4,
            0.18770927679626136909e-6,
            0.12597184587583370712e-8,
            0.73203433049229821618e-11,
            0.36087889048888888890e-13,
        ),
        (
            0.15626501395774612325e0,
            0.29226079376196624949e-2,
            0.25664553693768450545e-4,
            0.19808568415654461964e-6,
            0.13351257759815557897e-8,
            0.77658124891046760667e-11,
            0.38173420035555555555e-13,
        ),
        (
            0.16221449434620737567e0,
            0.30276865332726475672e-2,
            0.26885741326534564336e-4,
            0.20908350604346384143e-6,
            0.14151148144240728728e-8,
            0.82369170665974313027e-11,
            0.40360957457777777779e-13,
        ),
        (
            0.16837910595412130659e0,
            0.31377844510793082301e-2,
            0.28174873844911175026e-4,
            0.22074043807045782387e-6,
            0.14999481055996090039e-8,
            0.87348993661930809254e-11,
            0.42653528977777777779e-13,
        ),
        (
            0.17476916455659369953e0,
            0.32531815370903068316e-2,
            0.29536024347344364074e-4,
            0.23309632627767074202e-6,
            0.15899007843582444846e-8,
            0.92610375235427359475e-11,
            0.45054073102222222221e-13,
        ),
        (
            0.18139556223643701364e0,
            0.33741744168096996041e-2,
            0.30973511714709500836e-4,
            0.24619326937592290996e-6,
            0.16852609412267750744e-8,
            0.98166442942854895573e-11,
            0.47565418097777777779e-13,
        ),
        (
            0.18826980194443664549e0,
            0.35010775057740317997e-2,
            0.32491914440014267480e-4,
            0.26007572375886319028e-6,
            0.17863299617388376116e-8,
            0.10403065638343878679e-10,
            0.50190265831111111110e-13,
        ),
        (
            0.19540403413693967350e0,
            0.36342240767211326315e-2,
            0.34096085096200907289e-4,
            0.27479061117017637474e-6,
            0.18934228504790032826e-8,
            0.11021679075323598664e-10,
            0.52931171733333333334e-13,
        ),
        (
            0.20281109560651886959e0,
            0.37739673859323597060e-2,
            0.35791165457592409054e-4,
            0.29038742889416172404e-6,
            0.20068685374849001770e-8,
            0.11673891799578381999e-10,
            0.55790523093333333334e-13,
        ),
        (
            0.21050455062669334978e0,
            0.39206818613925652425e-2,
            0.37582602289680101704e-4,
            0.30691836231886877385e-6,
            0.21270101645763677824e-8,
            0.12361138551062899455e-10,
            0.58770520160000000000e-13,
        ),
        (
            0.21849873453703332479e0,
            0.40747643554689586041e-2,
            0.39476163820986711501e-4,
            0.32443839970139918836e-6,
            0.22542053491518680200e-8,
            0.13084879235290858490e-10,
            0.61873153262222222221e-13,
        ),
        (
            0.22680879990043229327e0,
            0.42366354648628516935e-2,
            0.41477956909656896779e-4,
            0.34300544894502810002e-6,
            0.23888264229264067658e-8,
            0.13846596292818514601e-10,
            0.65100183751111111110e-13,
        ),
        (
            0.23545076536988703937e0,
            0.44067409206365170888e-2,
            0.43594444916224700881e-4,
            0.36268045617760415178e-6,
            0.25312606430853202748e-8,
            0.14647791812837903061e-10,
            0.68453122631111111110e-13,
        ),
        (
            0.24444156740777432838e0,
            0.45855530511605787178e-2,
            0.45832466292683085475e-4,
            0.38352752590033030472e-6,
            0.26819103733055603460e-8,
            0.15489984390884756993e-10,
            0.71933206364444444445e-13,
        ),
        (
            0.25379911500634264643e0,
            0.47735723208650032167e-2,
            0.48199253896534185372e-4,
            0.40561404245564732314e-6,
            0.28411932320871165585e-8,
            0.16374705736458320149e-10,
            0.75541379822222222221e-13,
        ),
        (
            0.26354234756393613032e0,
            0.49713289477083781266e-2,
            0.50702455036930367504e-4,
            0.42901079254268185722e-6,
            0.30095422058900481753e-8,
            0.17303497025347342498e-10,
            0.79278273368888888890e-13,
        ),
        (
            0.27369129607732343398e0,
            0.51793846023052643767e-2,
            0.53350152258326602629e-4,
            0.45379208848865015485e-6,
            0.31874057245814381257e-8,
            0.18277905010245111046e-10,
            0.83144182364444444445e-13,
        ),
        (
            0.28426714781640316172e0,
            0.53983341916695141966e-2,
            0.56150884865255810638e-4,
            0.48003589196494734238e-6,
            0.33752476967570796349e-8,
            0.19299477888083469086e-10,
            0.87139049137777777779e-13,
        ),
        (
            0.29529231465348519920e0,
            0.56288077305420795663e-2,
            0.59113671189913307427e-4,
            0.50782393781744840482e-6,
            0.35735475025851713168e-8,
            0.20369760937017070382e-10,
            0.91262442613333333334e-13,
        ),
        (
            0.30679050522528838613e0,
            0.58714723032745403331e-2,
            0.62248031602197686791e-4,
            0.53724185766200945789e-6,
            0.37827999418960232678e-8,
            0.21490291930444538307e-10,
            0.95513539182222222221e-13,
        ),
        (
            0.31878680111173319425e0,
            0.61270341192339103514e-2,
            0.65564012259707640976e-4,
            0.56837930287837738996e-6,
            0.40035151353392378882e-8,
            0.22662596341239294792e-10,
            0.99891109760000000000e-13,
        ),
        (
            0.33130773722152622027e0,
            0.63962406646798080903e-2,
            0.69072209592942396666e-4,
            0.60133006661885941812e-6,
            0.42362183765883466691e-8,
            0.23888182347073698382e-10,
            0.10439349811555555556e-12,
        ),
        (
            0.34438138658041336523e0,
            0.66798829540414007258e-2,
            0.72783795518603561144e-4,
            0.63619220443228800680e-6,
            0.44814499336514453364e-8,
            0.25168535651285475274e-10,
            0.10901861383111111111e-12,
        ),
        (
            0.35803744972380175583e0,
            0.69787978834882685031e-2,
            0.76710543371454822497e-4,
            0.67306815308917386747e-6,
            0.47397647975845228205e-8,
            0.26505114141143050509e-10,
            0.11376390933333333333e-12,
        ),
        (
            0.37230734890119724188e0,
            0.72938706896461381003e-2,
            0.80864854542670714092e-4,
            0.71206484718062688779e-6,
            0.50117323769745883805e-8,
            0.27899342394100074165e-10,
            0.11862637614222222222e-12,
        ),
        (
            0.38722432730555448223e0,
            0.76260375162549802745e-2,
            0.85259785810004603848e-4,
            0.75329383305171327677e-6,
            0.52979361368388119355e-8,
            0.29352606054164086709e-10,
            0.12360253370666666667e-12,
        ),
        (
            0.40282355354616940667e0,
            0.79762880915029728079e-2,
            0.89909077342438246452e-4,
            0.79687137961956194579e-6,
            0.55989731807360403195e-8,
            0.30866246101464869050e-10,
            0.12868841946666666667e-12,
        ),
        (
            0.41914223158913787649e0,
            0.83456685186950463538e-2,
            0.94827181359250161335e-4,
            0.84291858561783141014e-6,
            0.59154537751083485684e-8,
            0.32441553034347469291e-10,
            0.13387957943111111111e-12,
        ),
        (
            0.43621971639463786896e0,
            0.87352841828289495773e-2,
            0.10002929142066799966e-3,
            0.89156148280219880024e-6,
            0.62480008150788597147e-8,
            0.34079760983458878910e-10,
            0.13917107176888888889e-12,
        ),
        (
            0.45409763548534330981e0,
            0.91463027755548240654e-2,
            0.10553137232446167258e-3,
            0.94293113464638623798e-6,
            0.65972492312219959885e-8,
            0.35782041795476563662e-10,
            0.14455745872000000000e-12,
        ),
        (
            0.47282001668512331468e0,
            0.95799574408860463394e-2,
            0.11135019058000067469e-3,
            0.99716373005509038080e-6,
            0.69638453369956970347e-8,
            0.37549499088161345850e-10,
            0.15003280712888888889e-12,
        ),
        (
            0.49243342227179841649e0,
            0.10037550043909497071e-1,
            0.11750334542845234952e-3,
            0.10544006716188967172e-5,
            0.73484461168242224872e-8,
            0.39383162326435752965e-10,
            0.15559069118222222222e-12,
        ),
        (
            0.51298708979209258326e0,
            0.10520454564612427224e-1,
            0.12400930037494996655e-3,
            0.11147886579371265246e-5,
            0.77517184550568711454e-8,
            0.41283980931872622611e-10,
            0.16122419680000000000e-12,
        ),
        (
            0.53453307979101369843e0,
            0.11030120618800726938e-1,
            0.13088741519572269581e-3,
            0.11784797595374515432e-5,
            0.81743383063044825400e-8,
            0.43252818449517081051e-10,
            0.16692592640000000000e-12,
        ),
        (
            0.55712643071169299478e0,
            0.11568077107929735233e-1,
            0.13815797838036651289e-3,
            0.12456314879260904558e-5,
            0.86169898078969313597e-8,
            0.45290446811539652525e-10,
            0.17268801084444444444e-12,
        ),
        (
            0.58082532122519320968e0,
            0.12135935999503877077e-1,
            0.14584223996665838559e-3,
            0.13164068573095710742e-5,
            0.90803643355106020163e-8,
            0.47397540713124619155e-10,
            0.17850211608888888889e-12,
        ),
        (
            0.60569124025293375554e0,
            0.12735396239525550361e-1,
            0.15396244472258863344e-3,
            0.13909744385382818253e-5,
            0.95651595032306228245e-8,
            0.49574672127669041550e-10,
            0.18435945564444444444e-12,
        ),
        (
            0.63178916494715716894e0,
            0.13368247798287030927e-1,
            0.16254186562762076141e-3,
            0.14695084048334056083e-5,
            0.10072078109604152350e-7,
            0.51822304995680707483e-10,
            0.19025081422222222222e-12,
        ),
        (
            0.65918774689725319200e0,
            0.14036375850601992063e-1,
            0.17160483760259706354e-3,
            0.15521885688723188371e-5,
            0.10601827031535280590e-7,
            0.54140790105837520499e-10,
            0.19616655146666666667e-12,
        ),
        (
            0.68795950683174433822e0,
            0.14741765091365869084e-1,
            0.18117679143520433835e-3,
            0.16392004108230585213e-5,
            0.11155116068018043001e-7,
            0.56530360194925690374e-10,
            0.20209663662222222222e-12,
        ),
        (
            0.71818103808729967036e0,
            0.15486504187117112279e-1,
            0.19128428784550923217e-3,
            0.17307350969359975848e-5,
            0.11732656736113607751e-7,
            0.58991125287563833603e-10,
            0.20803065333333333333e-12,
        ),
        (
            0.74993321911726254661e0,
            0.16272790364044783382e-1,
            0.20195505163377912645e-3,
            0.18269894883203346953e-5,
            0.12335161021630225535e-7,
            0.61523068312169087227e-10,
            0.21395783431111111111e-12,
        ),
        (
            0.78330143531283492729e0,
            0.17102934132652429240e-1,
            0.21321800585063327041e-3,
            0.19281661395543913713e-5,
            0.12963340087354341574e-7,
            0.64126040998066348872e-10,
            0.21986708942222222222e-12,
        ),
        (
            0.81837581041023811832e0,
            0.17979364149044223802e-1,
            0.22510330592753129006e-3,
            0.20344732868018175389e-5,
            0.13617902941839949718e-7,
            0.66799760083972474642e-10,
            0.22574701262222222222e-12,
        ),
        (
            0.85525144775685126237e0,
            0.18904632212547561026e-1,
            0.23764237370371255638e-3,
            0.21461248251306387979e-5,
            0.14299555071870523786e-7,
            0.69543803864694171934e-10,
            0.23158593688888888889e-12,
        ),
        (
            0.89402868170849933734e0,
            0.19881418399127202569e-1,
            0.25086793128395995798e-3,
            0.22633402747585233180e-5,
            0.15008997042116532283e-7,
            0.72357609075043941261e-10,
            0.23737194737777777778e-12,
        ),
        (
            0.93481333942870796363e0,
            0.20912536329780368893e-1,
            0.26481403465998477969e-3,
            0.23863447359754921676e-5,
            0.15746923065472184451e-7,
            0.75240468141720143653e-10,
            0.24309291271111111111e-12,
        ),
        (
            0.97771701335885035464e0,
            0.22000938572830479551e-1,
            0.27951610702682383001e-3,
            0.25153688325245314530e-5,
            0.16514019547822821453e-7,
            0.78191526829368231251e-10,
            0.24873652355555555556e-12,
        ),
    )
    segment = y100.to(tl.int32)
    c0 = tl.full(y100.shape, coefficients[0][0], tl.float64)
    c1 = tl.full(y100.shape, coefficients[0][1], tl.float64)
    c2 = tl.full(y100.shape, coefficients[0][2], tl.float64)
    c3 = tl.full(y100.shape, coefficients[0][3], tl.float64)
    c4 = tl.full(y100.shape, coefficients[0][4], tl.float64)
    c5 = tl.full(y100.shape, coefficients[0][5], tl.float64)
    c6 = tl.full(y100.shape, coefficients[0][6], tl.float64)
    for i in tl.static_range(1, 100):
        match = segment == i
        c0 = tl.where(match, tl.full(y100.shape, coefficients[i][0], tl.float64), c0)
        c1 = tl.where(match, tl.full(y100.shape, coefficients[i][1], tl.float64), c1)
        c2 = tl.where(match, tl.full(y100.shape, coefficients[i][2], tl.float64), c2)
        c3 = tl.where(match, tl.full(y100.shape, coefficients[i][3], tl.float64), c3)
        c4 = tl.where(match, tl.full(y100.shape, coefficients[i][4], tl.float64), c4)
        c5 = tl.where(match, tl.full(y100.shape, coefficients[i][5], tl.float64), c5)
        c6 = tl.where(match, tl.full(y100.shape, coefficients[i][6], tl.float64), c6)

    offset = (2 * segment + 1).to(y100.dtype)
    t = tl.fma(y100, 2.0, -offset).to(tl.float64)
    result = tl.fma(c6, t, c5)
    result = tl.fma(result, t, c4)
    result = tl.fma(result, t, c3)
    result = tl.fma(result, t, c2)
    result = tl.fma(result, t, c1)
    result = tl.fma(result, t, c0)
    return tl.where(y100 == 100.0, 1.0, result.to(y100.dtype))


@triton.jit
def aten_erfcx(x):
    zero = tl.full(x.shape, 0.0, x.dtype)
    four = tl.full(x.shape, 4.0, x.dtype)
    four_hundred = tl.full(x.shape, 400.0, x.dtype)
    negative_cutoff = tl.full(x.shape, -6.1, x.dtype)
    overflow_cutoff = tl.full(x.shape, -26.7, x.dtype)
    positive_polynomial = (x >= 0.0) & (x <= 50.0)
    negative_polynomial = (x < 0.0) & (x >= negative_cutoff)
    positive_x = tl.where(positive_polynomial, x, zero)
    negative_x = tl.where(negative_polynomial, x, zero)
    positive_y100 = libdevice.div_rn(four_hundred, four + positive_x)
    negative_y100 = libdevice.div_rn(four_hundred, four - negative_x)
    # Only one side is ever kept, and the two differ solely in their argument, so
    # select the argument rather than the result: one pass over the 100-segment
    # table instead of two.
    poly = _aten_erfcx_y100(tl.where(x >= 0.0, positive_y100, negative_y100))

    x2 = x * x
    numerator_inner = tl.fma(x2, x2 + 4.5, 2.0)
    denominator_inner = tl.fma(x2, x2 + 5.0, 3.75)
    ispi = tl.full(x.shape, 0.56418958354775628694807945156, x.dtype)
    continued_fraction = libdevice.div_rn(ispi * numerator_inner, x * denominator_inner)
    positive_large = tl.where(x > 5e7, libdevice.div_rn(ispi, x), continued_fraction)
    positive = tl.where(x > 50.0, positive_large, poly)

    exp_x2 = libdevice.exp(x2)
    doubled_exp = exp_x2 + exp_x2
    negative = tl.where(
        x < overflow_cutoff,
        float("inf"),
        tl.where(x < negative_cutoff, doubled_exp, doubled_exp - poly),
    )
    result = tl.where(x >= 0.0, positive, negative)
    return tl.where(x != x, x, result)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def trunc_to_float(x, float_dtype: tl.constexpr):
    # libdevice.trunc only accepts floating-point inputs.
    if is_floating(x):
        # Adding zero preserves large results while canonicalizing
        # libdevice.trunc(-0.0) to Python's +0.0 materialization behavior.
        return libdevice.trunc(x) + tl.zeros_like(x)
    else:
        return x.to(float_dtype)


# libdevice.ceil, libdevice.floor, and libdevice.trunc accept only floating-point
# inputs. Conservatively canonicalized symbolic expressions can still be integer
# at Triton compile time, so bypass libdevice for those inputs.
@triton.jit
def rounding_to_int(x, index_dtype: tl.constexpr, rounding: tl.constexpr):
    if is_floating(x):
        if rounding == "ceil":
            return libdevice.ceil(x).to(index_dtype)
        elif rounding == "floor":
            return libdevice.floor(x).to(index_dtype)
        elif rounding == "trunc":
            return libdevice.trunc(x).to(index_dtype)
        else:
            tl.static_assert(False, "unsupported rounding mode")
    else:
        return x.to(index_dtype)


@triton.jit
def _prod_accumulate(a, b):
    return a * b


@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def prod_inner_tree(input, axis, reduction_ordering: tl.constexpr):
    # Strict-numerics only. Emitted solely on the strict path, which is gated
    # behind has_triton_reduction_ordering(), so Triton builds lacking the
    # keyword never compile this helper -- keeping default `prod` portable.
    return tl.reduce(
        input, axis, _prod_accumulate, reduction_ordering=reduction_ordering
    )


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _minimum_reduce(a, b):
    value = minimum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def _maximum_reduce(a, b):
    value = maximum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def fmaximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def nextafter(x, y):
    bitwidth: tl.constexpr = x.dtype.primitive_bitwidth
    if bitwidth == 64:
        result = libdevice.nextafter(x, y)
    else:
        # libdevice.nextafterf honors CUDA FTZ and skips fp32 subnormals. For
        # fp16/bf16, stepping must happen before values are promoted to fp32.
        idtype: tl.constexpr = tl.core.get_int_dtype(bitwidth, signed=False)
        ix = x.to(idtype, bitcast=True)
        iy = y.to(idtype, bitcast=True)
        sign_mask: tl.constexpr = 1 << (bitwidth - 1)

        x_is_zero = (ix & (sign_mask - 1)) == 0
        y_is_zero = (iy & (sign_mask - 1)) == 0

        # Compare bit patterns so denormal handling cannot affect IEEE ordering.
        same_sign = (ix & sign_mask) == (iy & sign_mask)
        step_up = same_sign & (iy > ix)
        stepped = ix + tl.where(step_up, 1, -1).to(idtype)
        zero_step = (iy & sign_mask) | 1

        result = (
            tl.where(x_is_zero, zero_step, stepped).to(idtype).to(x.dtype, bitcast=True)
        )
        result = tl.where((ix == iy) | (x_is_zero & y_is_zero), y, result)

    # ROCm's fp64 libdevice nextafter can preserve signaling NaNs.
    return tl.where((x != x) | (y != y), x + y, result)


@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)


@triton.jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)


@triton.jit
def min2_strict(a, dim):
    return tl.reduce(a, dim, _minimum_reduce)


@triton.jit
def max2_strict(a, dim):
    return tl.reduce(a, dim, _maximum_reduce)


@triton.jit
def fmax2(a, dim):
    return tl.reduce(a, dim, fmaximum)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _first_index_of(value, result, index, dim):
    # The smallest index whose lane attains `result` from max2/min2 (the NaN
    # lanes when it is NaN): the index a NaN-aware tuple reduce would pick, at
    # the cost of two native reductions instead of a combine of about ten
    # instructions per element. The paired value is the extremum, not the
    # winning lane's, which differs on a tie between -0.0 and 0.0.
    hit = (value == tl.expand_dims(result, dim)) | (value != value)
    sentinel = tl.full(
        [1], (1 << (index.dtype.primitive_bitwidth - 1)) - 1, index.dtype
    )
    return tl.min(tl.where(hit, index, sentinel), dim)


@triton.jit
def min_with_first_index(value, index, dim):
    min_value = min2(value, dim)
    return min_value, _first_index_of(value, min_value, index, dim)


@triton.jit
def max_with_first_index(value, index, dim):
    max_value = max2(value, dim)
    return max_value, _first_index_of(value, max_value, index, dim)


@triton.jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


@triton.jit
def exp(x, use_fast_math: tl.constexpr):
    if use_fast_math:
        return math.exp(x)
    else:
        return libdevice.exp(x)


@triton.jit
def online_softmax_reduce(
    lhs_max,
    lhs_sum,
    dim,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = max2_strict(lhs_max, dim)
    else:
        out_max = max2(lhs_max, dim)
    out_max_keepdim = tl.expand_dims(out_max, dim)
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * exp(delta, use_fast_math), dim)
    return out_max, out_sum


@triton.jit
def online_softmax_combine(
    lhs_max,
    lhs_sum,
    rhs_max,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    """
    When we do combine, we assume lhs is the accumulator and rhs is the next
    block of data.
    Then rhs_sum is always 1. With that assumption, we can save some registers
    and computation.
    """
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    scale = exp(-tl.abs(lhs_max - rhs_max), use_fast_math)
    lhs_scale = tl.where(lhs_max == out_max, 1.0, scale)
    rhs_scale = tl.where(rhs_max == out_max, 1.0, scale)

    # Should be
    #   out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    # but since rhs_sum is all 1, we can simplify it.
    out_sum = tl.fma(lhs_sum, lhs_scale, rhs_scale)
    out_sum = tl.where(out_max == float("inf"), float("nan"), out_sum)
    return out_max, out_sum


@triton.jit
def online_softmax_reduce_scalar_combine(
    lhs_max,
    lhs_sum,
    rhs,
    rhs_mask,
    dim,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    """
    Reduce a block of values along `dim` and fold it into a per-row (max, sum)
    state, so only one max/sum per output row stays live across the loop.
    """
    rhs = tl.where(rhs_mask, rhs, float("-inf")).to(lhs_max.dtype)
    if strict_signed_zero:
        rhs_max = max2_strict(rhs, dim)
    else:
        rhs_max = max2(rhs, dim)
    rhs_max_keepdim = tl.expand_dims(rhs_max, dim)
    delta = tl.where(rhs_max_keepdim == float("-inf"), 0, rhs - rhs_max_keepdim)
    # Same sum as weighting each exp by a 0/1 mask, without the multiply.
    rhs_sum = tl.sum(tl.where(rhs_mask, exp(delta, use_fast_math), 0.0), dim)
    return online_softmax_combine_with_sum(
        lhs_max, lhs_sum, rhs_max, rhs_sum, use_fast_math, strict_signed_zero
    )


@triton.jit
def online_softmax_combine_with_sum(
    lhs_max,
    lhs_sum,
    rhs_max,
    rhs_sum,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    scale = exp(-tl.abs(lhs_max - rhs_max), use_fast_math)
    lhs_scale = tl.where(lhs_max == out_max, 1.0, scale)
    rhs_scale = tl.where(rhs_max == out_max, 1.0, scale)

    out_sum = tl.fma(lhs_sum, lhs_scale, rhs_sum * rhs_scale)
    out_sum = tl.where(out_max == float("inf"), float("nan"), out_sum)
    return out_max, out_sum


@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = tl.fma(delta, value - new_mean, m2)
    return new_mean, new_m2, new_weight


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    # Guard against inf - inf = NaN when both means are infinite and equal.
    # This occurs during FP16/BF16 LayerNorm when inputs overflow to inf.
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        tl.fma(delta, w2_over_w, mean_1),
        tl.fma(delta * delta * weight_1, w2_over_w, m2_1 + m2_2),
        new_weight,
    )


@triton.jit
def welford(mean, m2, weight, dim):
    return tl.reduce((mean, m2, weight), dim, welford_combine)


@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r


@triton.jit
def rand_eager_kernel(seed, offset_blocks, tid: tl.tensor, VEC: tl.constexpr):
    inv = 1.0 / 4294967296.0
    half = inv * 0.5

    tid_u64 = tid.to(tl.uint64)

    subseq = tid_u64 // VEC
    which4 = (tid_u64 % VEC) // 4
    lane = tid_u64 % 4

    offblk = offset_blocks.to(tl.uint64) + which4

    u0, u1, u2, u3 = tl.philox(
        seed,
        (offblk & 0xFFFFFFFF).to(tl.uint32),
        ((offblk >> 32) & 0xFFFFFFFF).to(tl.uint32),
        (subseq & 0xFFFFFFFF).to(tl.uint32),
        ((subseq >> 32) & 0xFFFFFFFF).to(tl.uint32),
    )

    v01 = tl.where(lane == 0, u0, u1)
    v23 = tl.where(lane == 2, u2, u3)
    rand_int = tl.where((lane == 0) | (lane == 1), v01, v23)

    return 1.0 - tl.fma(rand_int.to(tl.float32), inv, half)


@triton.jit
def _random_4x_to_block(r0, r1, r2, r3):
    # Pack lanes by logical offset so the random stream is independent of XBLOCK.
    size: tl.constexpr = r0.numel
    return tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), [size * 4])


@triton.jit
def rand4x(seed, offsets, BLOCK: tl.constexpr):
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.rand4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.rand(seed, offsets)


@triton.jit
def randn4x(seed, offsets, BLOCK: tl.constexpr):
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.randn4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.randn(seed, offsets)


@triton.jit
def randint64(seed, offset, low, high):
    r0, r1, _r2, _r3 = tl.randint4x(seed, offset)
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    result = r0 | (r1 << 32)
    size = high - low
    result = result % size.to(tl.uint64)
    result = result.to(tl.int64) + low
    return result


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def any(a, dim):
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def bucketize_binary_search(
    values: tl.tensor,
    boundaries_ptr: tl.tensor,
    BOUNDARIES_SIZE: int,
    BOUNDARIES_UNDERLYING_NUMEL: int,
    BOUNDARIES_STRIDE: int,
    boundary_indices: tl.tensor,
    indexing_dtype: tl.dtype,
    right: "bool",  # triton can't handle the unquoted bool annotation
    sorter_ptr: tl.tensor,
    SORTER_STRIDE: int,
    sorter_indices: tl.tensor,
):
    """
    See [Note: Inductor bucketize op]

    Inputs:
    -------
    values: the values to bucketize.
    boundaries_ptr: a pointer to the beginning of the boundaries tensor, in 1-D.
    BOUNDARIES_SIZE: the length of the last dimension of the boundaries tensor (i.e. one
    individual set of boundaries).
    BOUNDARIES_UNDERLYING_NUMEL: the length of the boundaries tensor, in 1-D, ignoring
    any striding.
    BOUNDARIES_STRIDE: the stride of the last dimension of the boundaries tensor
    boundary_indices: a tensor of the same size as "values"; each element is an index
    into a 1-D, un-strided boundaries tensor, pointing to the first element in the set
    of boundaries used for that value.
    indexing_dtype: the dtype used for indexing into the boundaries tensor, and the
    return dtype.
    right: if true, use boundary intervals closed on the left; otherwise use intervals
    closed on the right.
    sorter_ptr: an optional pointer to a sorter tensor of the same shape as boundaries,
    but potentially different striding.  If present, this allows us to treat boundaries
    as sorted even if the elements of boundaries are unsorted.
    SORTER_STRIDE: must be present if sorter_ptr is non-None; the stride of the last
    dimension of the sorter tensor.
    sorter_indices: must be present if sorter_ptr is non-None; see "boundary_indices".
    BLOCK_SHAPE: the shape of the data block being processed.
    """

    low = tl.zeros(values.shape, dtype=indexing_dtype)
    high = tl.full(values.shape, BOUNDARIES_SIZE, dtype=indexing_dtype)

    full_range = BOUNDARIES_SIZE + 1
    while full_range > 1:
        mid = (high + low) // 2
        mask = (
            (mid * BOUNDARIES_STRIDE + boundary_indices) < BOUNDARIES_UNDERLYING_NUMEL
        ).logical_and(mid < BOUNDARIES_SIZE)
        mid_indices = (
            mid
            if sorter_ptr is None or SORTER_STRIDE is None
            else tl.load(
                sorter_ptr + sorter_indices + SORTER_STRIDE * mid,
                mask=mask,
                other=0,
            )
        )

        bucket_upper_bound = tl.load(
            boundaries_ptr + boundary_indices + BOUNDARIES_STRIDE * mid_indices,
            mask=mask,
            other=0,
        )
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound

        if is_floating(values):
            is_above = is_above | (values != values)

        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)

        full_range = (full_range + 1) // 2

    return low


@triton.jit
def pack_value_flag(
    value,
    flag,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    uv = value.to(DTYPE_VALUE_AS_UINT, bitcast=True).to(DTYPE_PACK)
    return flag.to(DTYPE_PACK) | (uv << bitwidth)


@triton.jit
def unpack_value(
    pack,
    DTYPE_VALUE,
    DTYPE_VALUE_AS_UINT,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE = tl.core._unwrap_if_constexpr(DTYPE_VALUE)
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    value_uint = (pack >> bitwidth).to(DTYPE_VALUE_AS_UINT)
    return value_uint.to(DTYPE_VALUE, bitcast=True)


@triton.jit
def unpack_flag(pack, DTYPE_FLAG):
    return pack.to(DTYPE_FLAG)


@triton.jit
def exclusive_scan_decoupled_lookback(
    scratch_base,
    block_value,
    index,
    combine_fn,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    DTYPE_VALUE_AS_UINT: A tl.uint{n} type equal in size to ``block_value``
    DTYPE_PACK: Unsigned type twice the width of block_value

    NOTE: This function is limited to values which are 32-bits or less because
    we need to pack (value, flag) into a single unsigned int.
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    DTYPE_VALUE = block_value.dtype
    pack = pack_value_flag(
        block_value,
        tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        # tl.atomic_load
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack, DTYPE_VALUE_AS_UINT)

        value = unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    pack = pack_value_flag(
        inclusive_prefix,
        tl.full([], 2, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    return exclusive_prefix


@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identity of combine_fn
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    if index > 0:
        block_value_u64 = block_value.to(tl.uint64, bitcast=True)
        tl.store(scratch_base + 3 * index + 1, block_value_u64)
        tl.debug_barrier()
        flag_one = tl.full([], 1, tl.uint64)
        tl.atomic_xchg(scratch_base + 3 * index + 0, flag_one, sem="release")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], block_value.dtype)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        flag = tl.full([], 0, tl.uint64)
        while flag == 0:
            flag = tl.atomic_add(scratch_base + 3 * test_target + 0, 0, sem="acquire")

        value_u64 = tl.load(scratch_base + 3 * test_target + flag.to(tl.int32))
        value = value_u64.to(block_value.dtype, bitcast=True)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = tl.full([], -1, index.dtype)  # Match the original type
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    inclusive_prefix_u64 = inclusive_prefix.to(tl.uint64, bitcast=True)
    tl.store(scratch_base + 3 * index + 2, inclusive_prefix_u64)
    tl.debug_barrier()
    flag_two = tl.full([], 2, tl.uint64)
    tl.atomic_xchg(scratch_base + 3 * index + 0, flag_two, sem="release")

    return exclusive_prefix


@triton.jit
def cuda_mul_full_block(index, numel):
    # TensorIterator recursively halves iterations whose byte offsets exceed int32.
    index = index.to(tl.int64)
    start = index * 0
    size = start + numel
    while tl.max(tl.reshape(size, (size.numel,))) > 268435456:
        active = size > 268435456
        half = size // 2
        right = active & (index - start >= half)
        start = tl.where(right, start + half, start)
        size = tl.where(active, tl.where(right, size - half, half), size)
    # CUDA's vectorized FP64 pointwise loop uses 128 threads and 8 elements per thread.
    # An odd chunk start also misaligns the fresh native output pointer.
    return (index - start < size // 1024 * 1024) & (start % 2 == 0)


@triton.jit
def eager_neg(x):
    width: tl.constexpr = x.dtype.primitive_bitwidth
    idtype = tl.core.get_int_dtype(bitwidth=width, signed=True)
    sign = tl.full((), -1 << (width - 1), idtype)
    result = (x.to(idtype, bitcast=True) ^ sign).to(x.dtype, bitcast=True)
    return result


@triton.jit
def eager_div_rn(x, y):
    # Triton's div_rn emits div.rn.ftz.f32 even when kernel FTZ is disabled.
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def eager_fmod(a, b):
    if a.dtype == tl.float64:
        return libdevice.fmod(a, b)
    else:
        # Binary32 fmod results are exactly representable in binary64. Avoid
        # libdevice's float32 implementation, which flushes subnormals.
        result = libdevice.fmod(a.to(tl.float64), b.to(tl.float64)).to(tl.float32)
        return result


@triton.jit
def frexp(x):
    # Decompose the IEEE-754 bit pattern with integer ops rather than calling
    # libdevice.ilogb/ldexp: CUDA compiles libdevice with FTZ, which flushes
    # float32 subnormals to zero and would return a mantissa of 0 for subnormal
    # inputs, and the float path also loses the sign of -0.0.
    if x.dtype == tl.float64:
        MBITS: tl.constexpr = 52
        EMASK: tl.constexpr = 0x7FF
        BIAS: tl.constexpr = 1023
    elif x.dtype == tl.float32:
        MBITS: tl.constexpr = 23
        EMASK: tl.constexpr = 0xFF
        BIAS: tl.constexpr = 127
    elif x.dtype == tl.bfloat16:
        MBITS: tl.constexpr = 7
        EMASK: tl.constexpr = 0xFF
        BIAS: tl.constexpr = 127
    else:
        tl.static_assert(x.dtype == tl.float16)
        MBITS: tl.constexpr = 10
        EMASK: tl.constexpr = 0x1F
        BIAS: tl.constexpr = 15
    FMASK: tl.constexpr = (1 << MBITS) - 1
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    bits = x.to(idtype, bitcast=True)
    exp_field = (bits >> MBITS) & EMASK
    frac = bits & FMASK
    # Normalize subnormals by converting the fraction to float (exact, since
    # it fits in the mantissa), then reuse the normal-number path on that.
    is_sub = (exp_field == 0) & (frac != 0)
    norm_bits = frac.to(x.dtype).to(idtype, bitcast=True)
    src_bits = tl.where(is_sub, (bits & ~FMASK) | (norm_bits & FMASK), bits)
    src_exp = tl.where(is_sub, (norm_bits >> MBITS) - (BIAS - 1 + MBITS), exp_field)
    mantissa_bits = (src_bits & ~(EMASK << MBITS)) | ((BIAS - 1) << MBITS)
    # frexp(+-0) = (+-0, 0), frexp(+-inf) = (+-inf, 0), frexp(nan) = (nan, 0)
    special = (exp_field == EMASK) | ((exp_field == 0) & (frac == 0))
    mantissa = tl.where(special, x, mantissa_bits.to(x.dtype, bitcast=True))
    exponent = tl.where(special, 0, src_exp - (BIAS - 1)).to(tl.int32)
    return mantissa, exponent


@triton.jit
def _compare_and_swap_with_index(
    x,
    idxs,
    rnumel,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    y = tl.reshape(x, shape)
    iy = y.to(idtype, bitcast=True)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)
    left_mask = (1 - right_mask).to(idtype)
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1).to(idtype)[:, None, :], shape)
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1).to(idtype)[:, None, :], shape)
    ileft = tl.reshape(ileft, x.shape)
    iright = tl.reshape(iright, x.shape)
    left = ileft.to(x.dtype, bitcast=True)
    right = iright.to(x.dtype, bitcast=True)

    # idx
    y_idx = tl.reshape(idxs, shape)
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # valid
    if rnumel is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)
        right_valid_mask = tl.full(x.shape, True, tl.int1)
    else:
        left_valid_mask = left_idx < rnumel
        right_valid_mask = right_idx < rnumel

    # actual compare-and-swap
    ix = x.to(idtype, bitcast=True)

    # sort treats nan as having the higher value. comparisons with nan always return False.
    # to align with sort semantics, we need to update descending to check if right_isnan,
    # and ascending to check if left_isnan.
    left_isnan = left != left
    right_isnan = right != right

    if descending:
        cond = left < right
        if is_floating(left):
            if not stable:
                cond = cond | right_isnan
            else:
                cond = cond | (right_isnan & (~left_isnan))

    else:
        cond = left > right
        if is_floating(left):
            if not stable:
                cond = cond | left_isnan
            else:
                cond = cond | (left_isnan & (~right_isnan))

    if stable:
        # When stable sorting, tie break by index
        eq = left == right
        if is_floating(left):
            eq = eq | (left_isnan & right_isnan)
        cond = cond | (eq & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )
    cond = (cond ^ flip).to(tl.int1)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))

    return ret.to(x.dtype, bitcast=True), new_idxs


@triton.jit
def _bitonic_merge_with_index(
    x,
    idxs,
    rnumel,
    stage: tl.constexpr,
    alternating: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, idxs = _compare_and_swap_with_index(
            x, idxs, rnumel, flip, i + (n_dims - stage), n_dims, stable, descending
        )
    return x, idxs


@triton.jit
def sort_with_index(
    x,  # value
    idxs,  # index
    rnumel,  # number of elements
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    x, idxs = tl.broadcast(x, idxs)
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, idxs = _bitonic_merge_with_index(
            x,
            idxs,
            rnumel,
            i,
            alternating=i < n_dims,
            n_dims=n_dims,
            stable=stable,
            descending=descending,
        )
    return x, idxs


@triton.jit
def select_one(x, mask, dim, keep_dims=False):
    idtype = tl.core.get_int_dtype(x.dtype.primitive_bitwidth, signed=False)
    ix = x.to(idtype, bitcast=True)
    iy = tl.sum(ix * mask, dim, keep_dims=keep_dims)
    return iy.to(idtype).to(x.dtype, bitcast=True)


@triton.jit
def x_grid_barrier(sem):
    """
    Wait for all other thread blocks in grid sharing same y/z program_id
    to reach this barrier before returning.

    Args:
        sem: an uint32 semaphores, zero or 0x80000000 initialized.  Must be unique to each y/z program ID.
    """
    # ensure stores before this are visible
    tl.debug_barrier()

    one_i32 = 1
    one_u32 = one_i32.to(tl.uint32)  # type: ignore[attr-defined]
    expected = tl.num_programs(0).to(tl.uint32)
    if tl.program_id(0) == 0:
        nb = 0x80000000 - (expected - one_u32)
    else:
        nb = one_u32

    old_arrive = tl.atomic_add(sem, nb, sem="release")

    bar_flipped = False
    while not bar_flipped:
        # want a `ld.acquire.gpu.u32 $0,[$1];` but Triton doesn't have it
        current_arrive = tl.atomic_add(sem, 0, sem="acquire")
        # current_arrive = tl.load(sem, volatile=True)
        bar_flipped = ((old_arrive ^ current_arrive) & 0x80000000) != 0

    # TODO(jansel): is this needed?
    tl.debug_barrier()


def triton_builtin(f: Callable[..., _T]) -> Callable[..., _T]:
    """
    Decorator to mark a function as a Triton built-in function.  These functions
    are evaluated at compile time.

    Args:
        f (function): The function to be marked as a Triton built-in.

    Returns:
        function: The same function, marked as a Triton built-in.
    """
    if builtins_use_semantic_kwarg:
        # support Triton before and after https://github.com/triton-lang/triton/pull/7054
        # and after https://github.com/triton-lang/triton/pull/7239
        def wrapper(*args, _semantic, **kwargs):
            kwargs["_builder"] = _semantic
            return f(*args, **kwargs)
    else:
        wrapper = f  # type: ignore[assignment]

    wrapper.__triton_builtin__ = True  # type: ignore[attr-defined]
    return wrapper


@triton_builtin
def constexpr_next_power_of_2(
    n: tl.constexpr, *, _builder: object = None
) -> tl.constexpr:
    """
    A version of triton.next_power_of_two that can be used within a kernel on constants.
    """
    if not isinstance(n, tl.constexpr):
        raise AssertionError(f"Expected tl.constexpr, got {type(n)}")
    return tl.constexpr(triton.next_power_of_2(n.value))


@triton_builtin
def if_mask(mask: Any, val, *, _builder: object = None) -> tl.constexpr:
    """
    Work around triton compile error: `ValueError: `other` cannot be provided without `mask``
    A compile-time to check to return either `val` or `None` depending on the value of mask.
    """
    if isinstance(mask, tl.constexpr) and mask.value is None:
        return tl.constexpr(None)
    return val


@triton.jit
def inline_asm_pack(x, pack: tl.constexpr):
    """Ravel to 1D and pad (via join with zeros) so numel is divisible by pack."""
    result = x.ravel()
    # Only pad when the block size is smaller than pack. When block >= pack
    # the numel is already divisible by pack (both are powers of 2).
    n_pad: tl.constexpr = _log2(pack) - _log2(result.numel)
    for _ in tl.static_range(n_pad):
        result = tl.reshape(
            tl.join(result, tl.zeros_like(result)), (result.shape[0] * 2,)
        )
    return result


@triton.jit
def inline_asm_unpack(x, orig, pack: tl.constexpr):
    """Unpad and reshape back to orig's shape."""
    result = x
    n_pad: tl.constexpr = _log2(pack) - _log2(orig.numel)
    for _ in tl.static_range(n_pad):
        result, _ = tl.split(tl.reshape(result, (result.shape[0] // 2, 2)))
    return tl.reshape(result, orig.shape)


@triton.jit
def cuda_masked_sum_combine(a, b):
    return a + b


@triton.jit
def cuda_native_sum_runtime_metadata(
    SHAPE, STRIDES, ORDER: tl.constexpr, RDIMS: tl.constexpr
):
    r"""Sort reduction dimensions and select CUDA vectorization from runtime metadata."""
    RANK: tl.constexpr = len(SHAPE)
    WIDTH: tl.constexpr = triton.next_power_of_2(RANK)
    lane = tl.arange(0, WIDTH)
    shape = tl.full((WIDTH,), 1, tl.int64)
    stride = tl.full((WIDTH,), 0, tl.int64)
    permutation = tl.full((WIDTH,), 0, tl.int32)
    for d in tl.static_range(RANK):
        shape = tl.where(lane == d, SHAPE[d], shape)
        stride = tl.where(lane == d, STRIDES[d], stride)
        permutation = tl.where(lane == d, ORDER[d], permutation)
    for i in tl.static_range(1, RDIMS):
        right_position = tl.full((), i, tl.int32)
        active = tl.full((), True, tl.int1)
        for j in tl.static_range(i - 1, -1, -1):
            left = tl.gather(permutation, tl.full((1,), j, tl.int32), 0).reshape(())
            right = tl.gather(permutation, right_position[None], 0).reshape(())
            a = tl.gather(stride, left[None], 0).reshape(())
            b = tl.gather(stride, right[None], 0).reshape(())
            sa = tl.gather(shape, left[None], 0).reshape(())
            sb = tl.gather(shape, right[None], 0).reshape(())
            comparable = (a != 0) & (b != 0)
            swap = active & comparable & ((a > b) | ((a == b) & (sa > sb)))
            permutation = tl.where(
                swap & (lane == j),
                right,
                tl.where(swap & (lane == right_position), left, permutation),
            )
            right_position = tl.where(swap, j, right_position)
            active = active & ~(comparable & (a < b))
    first_red_stride = tl.full((), 0, tl.int64)
    expected = tl.full((), 0, tl.int64)
    started = tl.full((), False, tl.int1)
    dense = tl.full((), True, tl.int1)
    aligned2 = tl.full((), True, tl.int1)
    aligned4 = tl.full((), True, tl.int1)
    for d in tl.static_range(RDIMS):
        dim = tl.gather(permutation, tl.full((1,), d, tl.int32), 0).reshape(())
        size = tl.gather(shape, dim[None], 0).reshape(())
        step = tl.gather(stride, dim[None], 0).reshape(())
        significant = size > 1
        dense = dense & (~significant | ~started | (step == expected))
        first_red_stride = tl.where(significant & ~started, step, first_red_stride)
        expected = tl.where(
            significant, tl.where(started, expected, step) * size, expected
        )
        started = started | significant
        aligned2 = aligned2 & (~significant | (step % 2 == 0))
        aligned4 = aligned4 & (~significant | (step % 4 == 0))
    first_out_stride = tl.full((), 0, tl.int64)
    group_size = tl.full((), 1, tl.int64)
    expected = tl.full((), 0, tl.int64)
    started = tl.full((), False, tl.int1)
    group_open = tl.full((), True, tl.int1)
    for d in tl.static_range(RDIMS, RANK):
        dim = tl.gather(permutation, tl.full((1,), d, tl.int32), 0).reshape(())
        size = tl.gather(shape, dim[None], 0).reshape(())
        step = tl.gather(stride, dim[None], 0).reshape(())
        significant = size > 1
        first = significant & ~started
        member = first | (significant & started & group_open & (step == expected))
        first_out_stride = tl.where(first, step, first_out_stride)
        group_size = tl.where(member, group_size * size, group_size)
        expected = tl.where(member, step * size, expected)
        group_open = group_open & (~significant | member)
        started = started | significant
        aligned2 = aligned2 & (~significant | member | (step % 2 == 0))
        aligned4 = aligned4 & (~significant | member | (step % 4 == 0))
    fast = ~started | (first_red_stride < first_out_stride)
    vi = fast & dense & (first_red_stride == 1)
    ov2 = (first_out_stride == 1) & aligned2 & (group_size % 2 == 0)
    ov4 = (first_out_stride == 1) & aligned4 & (group_size % 4 == 0)
    return permutation, fast, vi, ov2, ov4


@triton.jit
def cuda_native_sum_runtime_offset(
    index, SHAPE, STRIDES, PERMUTATION, BEGIN: tl.constexpr, END: tl.constexpr
):
    RANK: tl.constexpr = len(SHAPE)
    WIDTH: tl.constexpr = triton.next_power_of_2(RANK)
    lane = tl.arange(0, WIDTH)
    shape = tl.full((WIDTH,), 1, tl.int64)
    stride = tl.full((WIDTH,), 0, tl.int64)
    for d in tl.static_range(RANK):
        shape = tl.where(lane == d, SHAPE[d], shape)
        stride = tl.where(lane == d, STRIDES[d], stride)
    result = index.to(tl.int64) * 0
    remainder = index.to(tl.int64)
    for d in tl.static_range(BEGIN, END):
        dim = tl.gather(PERMUTATION, tl.full((1,), d, tl.int32), 0).reshape(())
        size = tl.gather(shape, dim[None], 0).reshape(())
        step = tl.gather(stride, dim[None], 0).reshape(())
        result += remainder % size * step
        remainder //= size
    return result


@triton.jit
def cuda_native_sum_vector_get(values, index):
    slots = tl.arange(0, values.shape[0])
    return tl.sum(tl.where(slots == index, values, 0), 0)


@triton.jit
def cuda_native_sum_coalesced_metadata(
    SHAPE, ISTRIDES, ORDER: tl.constexpr, RDIMS: tl.constexpr
):
    permutation, _, _, _, _ = cuda_native_sum_runtime_metadata(
        SHAPE, ISTRIDES, ORDER, RDIMS
    )
    slots = tl.arange(0, permutation.shape[0])
    source_shape = tl.full((permutation.shape[0],), 1, tl.int64)
    source_input = tl.full((permutation.shape[0],), 0, tl.int64)
    source_output = tl.full((permutation.shape[0],), 0, tl.int64)
    output_stride = tl.full((), 1, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        source_shape = tl.where(slots == d, SHAPE[d], source_shape)
        source_input = tl.where(slots == d, ISTRIDES[d], source_input)
        is_reduced = False
        for r in tl.static_range(RDIMS):
            is_reduced = is_reduced or d == ORDER[r]
        if not is_reduced:
            source_output = tl.where(slots == d, output_stride, source_output)
            output_stride *= SHAPE[d]
    shape = tl.full((permutation.shape[0],), 1, tl.int64)
    inputs = tl.full((permutation.shape[0],), 0, tl.int64)
    outputs = tl.full((permutation.shape[0],), 0, tl.int64)
    ndim = tl.full((), 0, tl.int32)
    for d in tl.static_range(len(SHAPE)):
        original = cuda_native_sum_vector_get(permutation, d)
        size = cuda_native_sum_vector_get(source_shape, original)
        istride = cuda_native_sum_vector_get(source_input, original)
        ostride = cuda_native_sum_vector_get(source_output, original)
        previous = tl.maximum(ndim - 1, 0)
        last_size = cuda_native_sum_vector_get(shape, previous)
        last_input = cuda_native_sum_vector_get(inputs, previous)
        last_output = cuda_native_sum_vector_get(outputs, previous)
        combine = ndim > 0 and (
            last_size == 1
            or size == 1
            or (
                last_size * last_input == istride and last_size * last_output == ostride
            )
        )
        slot = tl.where(combine, ndim - 1, ndim)
        shape = tl.where(
            slots == slot, tl.where(combine, last_size * size, size), shape
        )
        inputs = tl.where(
            slots == slot,
            tl.where(combine and last_size != 1, last_input, istride),
            inputs,
        )
        outputs = tl.where(
            slots == slot,
            tl.where(combine and last_size != 1, last_output, ostride),
            outputs,
        )
        ndim += tl.where(combine, 0, 1)
    reduced = tl.sum(((slots < ndim) & (outputs == 0)).to(tl.int32), 0)
    return shape, inputs, outputs, ndim, reduced


@triton.jit
def cuda_native_sum_split_leaf(
    SHAPE,
    ISTRIDES,
    OSTRIDES,
    NDIM,
    INPUT_BYTES: tl.constexpr,
    OUTPUT_BYTES: tl.constexpr,
    target,
):
    slots = tl.arange(0, SHAPE.shape[0])
    # A signed 64-bit element count needs at most 126 nested odd-sized halvings.
    rows = tl.arange(0, 128)
    stack_shape = tl.full((128, SHAPE.shape[0]), 1, tl.int64)
    stack_start = tl.full((128, SHAPE.shape[0]), 0, tl.int64)
    stack_accumulate = tl.full((128,), False, tl.int1)
    size = SHAPE
    starts = tl.full((SHAPE.shape[0],), 0, tl.int64)
    accumulate = False
    depth = tl.full((), 0, tl.int32)
    leaf = tl.full((), 0, tl.int64)
    found = False
    available = True
    while available and not found:
        fits = (
            prod(size, 0) <= 2147483647
            and 1 + tl.sum((size - 1) * ISTRIDES * INPUT_BYTES, 0) <= 2147483647
            and 1 + tl.sum((size - 1) * OSTRIDES * OUTPUT_BYTES, 0) <= 2147483647
        )
        if fits:
            found = leaf == target
            if not found:
                leaf += 1
                if depth > 0:
                    depth -= 1
                    size = tl.sum(tl.where(rows[:, None] == depth, stack_shape, 0), 0)
                    starts = tl.sum(tl.where(rows[:, None] == depth, stack_start, 0), 0)
                    accumulate = (
                        tl.sum(
                            tl.where(rows == depth, stack_accumulate.to(tl.int32), 0), 0
                        )
                        != 0
                    )
                else:
                    available = False
        else:
            extents = (size - 1) * tl.maximum(
                tl.abs(ISTRIDES) * INPUT_BYTES, tl.abs(OSTRIDES) * OUTPUT_BYTES
            )
            maximum = tl.max(tl.where(slots < NDIM, extents, -1), 0)
            dim = tl.max(tl.where((slots < NDIM) & (extents == maximum), slots, -1), 0)
            count = cuda_native_sum_vector_get(size, dim)
            first = count // 2
            right_size = tl.where(slots == dim, count - first, size)
            right_start = starts + tl.where(slots == dim, first, 0)
            reduced = cuda_native_sum_vector_get(OSTRIDES, dim) == 0
            stack_shape = tl.where(
                rows[:, None] == depth, right_size[None, :], stack_shape
            )
            stack_start = tl.where(
                rows[:, None] == depth, right_start[None, :], stack_start
            )
            stack_accumulate = tl.where(
                rows == depth, accumulate or reduced, stack_accumulate
            )
            depth += 1
            size = tl.where(slots == dim, first, size)
    return size, starts, accumulate, found


@triton.jit
def cuda_native_sum_offset(
    index,
    SHAPE,
    STRIDES,
    BEGIN,
    END,
    PERMUTATION=None,
    PACKED: tl.constexpr = False,
):
    if PACKED:
        result = index.to(tl.int64) * 0
        remainder = index.to(tl.int64)
        for d in tl.static_range(SHAPE.shape[0]):
            if d >= BEGIN and d < END:
                size = cuda_native_sum_vector_get(SHAPE, d)
                stride = cuda_native_sum_vector_get(STRIDES, d)
                result += remainder % size * stride
                remainder //= size
        return result
    else:
        if PERMUTATION is not None:
            return cuda_native_sum_runtime_offset(
                index, SHAPE, STRIDES, PERMUTATION, BEGIN, END
            )
        if END - BEGIN == 1:
            return index.to(tl.int64) * STRIDES[BEGIN]
        result = index.to(tl.int64) * 0
        remainder = index.to(tl.int64)
        for d in tl.static_range(BEGIN, END):
            result += remainder % SHAPE[d] * STRIDES[d]
            remainder //= SHAPE[d]
        return result


@triton.jit
def cuda_native_sum_x_reduce(
    value, lane, BW: tl.constexpr, LANES: tl.constexpr, DESCENDING: tl.constexpr
):
    for power in tl.static_range(0, 9):
        shift = BW >> (power + 1)
        if shift > 0 and (DESCENDING or shift >= 32):
            partner = tl.gather(value, (lane + shift) % LANES, 0)
            value = tl.where(lane % BW < shift, value + partner, value)
    if not DESCENDING:
        for power in tl.static_range(0, 5):
            shift = 1 << power
            if shift < BW:
                partner = tl.gather(value, (lane + shift) % LANES, 0)
                value = tl.where(
                    lane % 32 + shift < tl.minimum(BW, 32), value + partner, value
                )
    return value


@triton.jit
def cuda_cudnn_bn_reduce(value):
    lane = tl.arange(0, 512)
    value = cuda_native_sum_x_reduce(value, lane, 32, 512, True)
    warp = tl.arange(0, 16)
    value = tl.gather(value, warp * 32, 0)
    value = cuda_native_sum_x_reduce(value, warp, 16, 16, True)
    return tl.sum(tl.where(warp == 0, value, 0), 0)


@triton.jit
def cuda_cudnn_bn_statistics(
    X,
    RM,
    RV,
    channel,
    N,
    C: tl.constexpr,
    S,
    EPS,
    MOMENTUM,
    WELFORD: tl.constexpr,
    VEC: tl.constexpr,
    BW: tl.constexpr,
):
    r"""Reproduce cuDNN NCHW means, variances, and running-stat updates."""
    DOUBLE: tl.constexpr = X.dtype.element_ty == tl.float64
    DTYPE: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    EPS = tl.full((), EPS, DTYPE)
    MOMENTUM = tl.full((), MOMENTUM, DTYPE)
    lane = tl.arange(0, 512)
    scale = libdevice.div_rn(tl.full((), 1, DTYPE), (N * S).to(DTYPE))
    if WELFORD:
        local_mean = tl.full((512,), 0, DTYPE)
        local_var = tl.full((512,), 0, DTYPE)
        count = tl.full((512,), 0, DTYPE)
        for batch_tile in range(tl.cdiv(N, 512 // BW)):
            batch = lane // BW + batch_tile * (512 // BW)
            for spatial_tile in range(tl.cdiv(S, BW)):
                spatial = lane % BW + spatial_tile * BW
                valid = (batch < N) & (spatial < S)
                value = tl.load(
                    X + batch * C * S + channel * S + spatial, valid, other=0
                ).to(DTYPE)
                new_count = count + valid.to(DTYPE)
                delta = value - local_mean
                denominator = tl.where(new_count > 0, new_count, 1)
                if DOUBLE:
                    new_mean = local_mean + libdevice.div_rn(delta, denominator)
                else:
                    reciprocal = tl.inline_asm_elementwise(
                        "rcp.approx.ftz.f32 $0,$1;",
                        constraints="=f,f",
                        args=[denominator],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                    new_mean = tl.fma(delta, reciprocal, local_mean)
                local_var = tl.where(
                    valid, tl.fma(delta, value - new_mean, local_var), local_var
                )
                local_mean = tl.where(valid, new_mean, local_mean)
                count = new_count
        mean = cuda_cudnn_bn_reduce(local_mean * count) * scale
        delta = mean - local_mean
        variance = cuda_cudnn_bn_reduce(tl.fma(delta * delta, count, local_var))
        invstd = libdevice.rsqrt(tl.fma(variance, scale, EPS))
        if DOUBLE:
            unbiased = libdevice.div_rn(variance, (N * S - 1).to(DTYPE))
        else:
            reciprocal = tl.inline_asm_elementwise(
                "rcp.approx.ftz.f32 $0,$1;",
                constraints="=f,f",
                args=[(N * S - 1).to(DTYPE)],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            unbiased = variance * reciprocal
        old_var = tl.load(RV + channel).to(DTYPE)
        if DOUBLE:
            new_var = tl.fma(unbiased, MOMENTUM, (1 - MOMENTUM) * old_var)
        else:
            new_var = tl.fma(old_var, 1 - MOMENTUM, unbiased * MOMENTUM)
    else:
        total = tl.full((512,), 0, DTYPE)
        for tile in range(tl.cdiv(N * S, 512 * VEC)):
            for component in tl.static_range(VEC):
                index = lane * VEC + tile * 512 * VEC + component
                value = tl.load(
                    X + (index // S) * C * S + channel * S + index % S,
                    index < N * S,
                    other=0,
                ).to(DTYPE)
                total = total + value
        mean = cuda_cudnn_bn_reduce(total) * scale
        local_var = tl.full((512,), 0, DTYPE)
        for tile in range(tl.cdiv(N * S, 512 * VEC)):
            for component in tl.static_range(VEC):
                index = lane * VEC + tile * 512 * VEC + component
                valid = index < N * S
                value = tl.load(
                    X + (index // S) * C * S + channel * S + index % S, valid, other=0
                ).to(DTYPE)
                delta = value - mean
                local_var = tl.where(valid, tl.fma(delta, delta, local_var), local_var)
        variance = cuda_cudnn_bn_reduce(local_var) * scale
        invstd = libdevice.rsqrt(variance + EPS)
        correction = libdevice.div_rn((N * S).to(DTYPE), (N * S - 1).to(DTYPE))
        old_var = tl.load(RV + channel).to(DTYPE)
        new_var = tl.fma(variance * MOMENTUM, correction, (1 - MOMENTUM) * old_var)
    old_mean = tl.load(RM + channel).to(DTYPE)
    new_mean = tl.fma(old_mean, 1 - MOMENTUM, mean * MOMENTUM)
    return mean, invstd, new_mean, new_var


@triton.jit
def cuda_cudnn_bn_backward_reduce(value):
    lane = tl.arange(0, 512)
    value = cuda_native_sum_x_reduce(value, lane, 32, 512, False)
    warp = tl.arange(0, 32)
    value = tl.gather(value, tl.minimum(warp, 15) * 32, 0)
    value = tl.where(warp < 16, value, 0)
    value = cuda_native_sum_x_reduce(value, warp, 32, 32, False)
    return tl.sum(tl.where(warp == 0, value, 0), 0)


@triton.jit
def cuda_cudnn_bn_backward_statistics(
    X,
    G,
    M,
    channel,
    N,
    C: tl.constexpr,
    S,
    SINGLE: tl.constexpr,
    VEC: tl.constexpr,
    BW: tl.constexpr,
):
    DTYPE: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    lane = tl.arange(0, 512)
    mean = tl.load(M + channel)
    total = tl.full((512,), 0, DTYPE)
    dot = tl.full((512,), 0, DTYPE)
    if SINGLE:
        for tile in range(tl.cdiv(N * S, 512 * VEC)):
            for component in tl.static_range(VEC):
                index = lane * VEC + tile * 512 * VEC + component
                valid = index < N * S
                offset = (index // S) * C * S + channel * S + index % S
                value = tl.load(X + offset, valid, other=0).to(DTYPE)
                grad = tl.load(G + offset, valid, other=0).to(DTYPE)
                total = total + grad
                dot = tl.where(valid, tl.fma(value - mean, grad, dot), dot)
    else:
        for batch_tile in range(tl.cdiv(N, 512 // BW)):
            batch = lane // BW + batch_tile * (512 // BW)
            for spatial_tile in range(tl.cdiv(S, BW)):
                spatial = lane % BW + spatial_tile * BW
                valid = (batch < N) & (spatial < S)
                offset = batch * C * S + channel * S + spatial
                value = tl.load(X + offset, valid, other=0).to(DTYPE)
                grad = tl.load(G + offset, valid, other=0).to(DTYPE)
                total = total + grad
                dot = tl.where(valid, tl.fma(value - mean, grad, dot), dot)
    return cuda_cudnn_bn_backward_reduce(total), cuda_cudnn_bn_backward_reduce(dot)


@triton.jit
def cuda_cudnn_bn_nhwc_reduce(value, TILE: tl.constexpr, VEC: tl.constexpr):
    WARP: tl.constexpr = 32 * VEC
    lane = tl.arange(0, 512 * VEC)
    for power in tl.static_range(5):
        shift = TILE * (1 << power)
        if shift < WARP:
            value = value + tl.gather(value, (lane + shift) % (512 * VEC), 0)
    index = tl.arange(0, WARP)
    warp, channel = index // TILE, index % TILE
    total = tl.gather(value, warp * WARP + channel, 0)
    for part in tl.static_range(1, tl.cdiv(16, WARP // TILE)):
        total = total + tl.gather(
            value, (warp + part * (WARP // TILE)) * WARP + channel, 0
        )
    for power in tl.static_range(5):
        shift = TILE * (1 << power)
        if shift < WARP:
            total = total + tl.gather(total, (index + shift) % WARP, 0)
    return tl.gather(total, lane % TILE, 0)


@triton.jit
def cuda_cudnn_bn_nhwc_partial(
    X,
    cta,
    channel_tile,
    N,
    C: tl.constexpr,
    TILE: tl.constexpr,
    VEC: tl.constexpr,
    SAMPLES: tl.constexpr,
    PARTS: tl.constexpr,
):
    LANES: tl.constexpr = 512 * VEC
    ROWS: tl.constexpr = LANES // TILE
    SPAN: tl.constexpr = SAMPLES * ROWS
    lane = tl.arange(0, LANES)
    channel = channel_tile * TILE + lane % TILE
    extra = tl.maximum(0, tl.cdiv(N, PARTS * ROWS) - SAMPLES)
    mean = tl.full((LANES,), 0, tl.float32)
    variance = tl.full((LANES,), 0, tl.float32)
    count = tl.full((LANES,), 0, tl.float32)
    # Register-cached tiles precede the separately partitioned stream of extra samples.
    for sample in range(SAMPLES + extra):
        row = tl.where(
            sample < SAMPLES,
            cta * SPAN + lane // TILE + sample * ROWS,
            PARTS * SPAN
            + cta * extra * ROWS
            + lane // TILE
            + (sample - SAMPLES) * ROWS,
        )
        valid = (row < N) & (channel < C)
        value = tl.load(X + row * C + channel, valid, other=0).to(tl.float32)
        new_count = count + valid.to(tl.float32)
        delta = value - mean
        reciprocal = tl.div_rn(1, tl.where(new_count > 0, new_count, 1))
        new_mean = tl.fma(delta, reciprocal, mean)
        variance = tl.where(
            valid,
            tl.fma(delta * (value - new_mean), valid.to(tl.float32), variance),
            variance,
        )
        mean = tl.where(valid, new_mean, mean)
        count = new_count
    cached = tl.minimum(tl.maximum(N - cta * SPAN, 0), SPAN)
    extra_start = PARTS * SPAN + cta * extra * ROWS
    streamed = tl.minimum(tl.maximum(N - extra_start, 0), extra * ROWS)
    total_count = (cached + streamed).to(tl.float32)
    total = cuda_cudnn_bn_nhwc_reduce(mean * count, TILE, VEC)
    delta = tl.fma(total, tl.div_rn(1, total_count), -mean)
    variance = tl.fma(delta * delta, count, variance)
    total_var = cuda_cudnn_bn_nhwc_reduce(variance, TILE, VEC)
    return total, total_var, total_count


@triton.jit
def cuda_cudnn_bn_nhwc_backward_partial(
    X,
    G,
    M,
    cta,
    channel_tile,
    N,
    C: tl.constexpr,
    TILE: tl.constexpr,
    VEC: tl.constexpr,
    PARTS: tl.constexpr,
    GROUPS: tl.constexpr,
    EXTRA: tl.constexpr,
    STREAMED: tl.constexpr,
):
    LANES: tl.constexpr = 512 * VEC
    ROWS: tl.constexpr = LANES // TILE
    lane = tl.arange(0, LANES)
    channel = channel_tile * TILE + lane % TILE
    shift = N - PARTS * (3 * GROUPS + EXTRA) * ROWS if STREAMED else 0
    mean = tl.load(M + channel, channel < C, other=0)
    grad_mean = tl.full((LANES,), 0, tl.float32)
    dot_mean = tl.full((LANES,), 0, tl.float32)
    count = tl.full((LANES,), 0, tl.float32)
    for sample in range(3 * GROUPS + EXTRA):
        row = (
            tl.where(
                sample < 3 * GROUPS,
                cta * 3 * ROWS
                + lane // TILE
                + (sample % 3) * ROWS
                + (sample // 3) * PARTS * 3 * ROWS,
                PARTS * 3 * GROUPS * ROWS
                + cta * EXTRA * ROWS
                + lane // TILE
                + (sample - 3 * GROUPS) * ROWS,
            )
            + shift
        )
        valid = (row >= 0) & (row < N) & (channel < C)
        value = tl.load(X + row * C + channel, valid, other=0).to(tl.float32)
        grad = tl.load(G + row * C + channel, valid, other=0).to(tl.float32)
        new_count = count + valid.to(tl.float32)
        reciprocal = tl.div_rn(1, tl.where(new_count > 0, new_count, 1))
        grad_mean = tl.where(
            valid, tl.fma(grad - grad_mean, reciprocal, grad_mean), grad_mean
        )
        delta = tl.fma(value - mean, grad, -dot_mean)
        dot_mean = tl.where(valid, tl.fma(delta, reciprocal, dot_mean), dot_mean)
        count = new_count
    total = cuda_cudnn_bn_nhwc_reduce(grad_mean * count, TILE, VEC)
    dot = cuda_cudnn_bn_nhwc_reduce(dot_mean * count, TILE, VEC)
    return total, dot


@triton.jit
def cuda_cudnn_bn_nhwc_backward_finish(
    P,
    channel_tile,
    C: tl.constexpr,
    TILE: tl.constexpr,
    VEC: tl.constexpr,
    PARTS: tl.constexpr,
):
    LANES: tl.constexpr = 512 * VEC
    ROWS: tl.constexpr = LANES // TILE
    lane = tl.arange(0, LANES)
    channel = channel_tile * TILE + lane % TILE
    total = tl.full((LANES,), 0, tl.float32)
    dot = tl.full((LANES,), 0, tl.float32)
    for tile in range(tl.cdiv(PARTS, ROWS)):
        part = lane // TILE + tile * ROWS
        valid = (part < PARTS) & (channel < C)
        offset = part * C + channel
        total = total + tl.load(P + offset, valid, other=0)
        dot = dot + tl.load(P + PARTS * C + offset, valid, other=0)
    return cuda_cudnn_bn_nhwc_reduce(total, TILE, VEC), cuda_cudnn_bn_nhwc_reduce(
        dot, TILE, VEC
    )


@triton.jit
def cuda_cudnn_bn_nhwc_finish(
    P,
    RM,
    RV,
    channel_tile,
    N,
    C: tl.constexpr,
    TILE: tl.constexpr,
    VEC: tl.constexpr,
    PARTS: tl.constexpr,
    EPS: tl.constexpr,
    MOMENTUM: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
):
    LANES: tl.constexpr = 512 * VEC
    ROWS: tl.constexpr = LANES // TILE
    lane = tl.arange(0, LANES)
    channel = (channel_tile * TILE + lane % TILE).to(INDEX_DTYPE)
    total = tl.full((LANES,), 0, tl.float32)
    for tile in range(tl.cdiv(PARTS, ROWS)):
        part = (lane // TILE + tile * ROWS).to(INDEX_DTYPE)
        value = tl.load(P + part * C + channel, (part < PARTS) & (channel < C), other=0)
        total = total + value
    scale = tl.div_rn(1, N.to(tl.float32))
    mean = cuda_cudnn_bn_nhwc_reduce(total, TILE, VEC) * scale
    total_var = tl.full((LANES,), 0, tl.float32)
    for tile in range(tl.cdiv(PARTS, ROWS)):
        part = (lane // TILE + tile * ROWS).to(INDEX_DTYPE)
        valid = (part < PARTS) & (channel < C)
        offset = part * C + channel
        value = tl.load(P + offset, valid, other=0)
        variance = tl.load(P + PARTS * C + offset, valid, other=0)
        count = tl.load(P + 2 * PARTS * C + offset, valid, other=1)
        local_mean = value * tl.div_rn(1, count)
        delta = local_mean - mean
        variance = tl.fma(delta * delta, count, variance)
        total_var = total_var + tl.where(valid, variance, 0)
    variance = cuda_cudnn_bn_nhwc_reduce(total_var, TILE, VEC)
    invstd = libdevice.rsqrt(tl.fma(variance, scale, tl.full((), EPS, tl.float32)))
    momentum = tl.full((), MOMENTUM, tl.float32)
    old_mean = tl.load(RM + channel, channel < C, other=0)
    old_var = tl.load(RV + channel, channel < C, other=0)
    if VEC == 4:
        new_mean = tl.fma(old_mean, 1 - momentum, mean * momentum)
    else:
        new_mean = tl.fma(mean, momentum, old_mean * (1 - momentum))
    unbiased_scale = tl.div_rn(1, (N - 1).to(tl.float32))
    new_var = tl.fma(old_var, 1 - momentum, (variance * momentum) * unbiased_scale)
    return mean, invstd, new_mean, new_var


@triton.jit
def cuda_native_sum_y_reduce(
    value,
    lane,
    BW: tl.constexpr,
    BH: tl.constexpr,
    FAST: tl.constexpr,
    LANES: tl.constexpr,
):
    for power in tl.static_range(0, 9):
        shift = BH >> (power + 1)
        if shift > 0:
            stride = BW if FAST else 1
            partner = tl.gather(value, (lane + shift * stride) % LANES, 0)
            value = tl.where(lane // stride < shift, value + partner, value)
    return value


@triton.jit
def cuda_native_sum(
    X,
    PARTIAL,
    output,
    cta,
    SHAPE,
    ISTRIDES,
    OSTRIDES: tl.constexpr,
    RDIMS,
    N,
    M: tl.constexpr,
    BW: tl.constexpr,
    BH: tl.constexpr,
    VI: tl.constexpr,
    IV: tl.constexpr,
    FAST: tl.constexpr,
    SY: tl.constexpr,
    CTAS,
    PARTIAL_STRIDE,
    FINAL: tl.constexpr,
    DESCENDING: tl.constexpr,
    LANES: tl.constexpr,
    ALIGNMENT: tl.constexpr = -1,
    PERMUTATION=None,
    PACKED: tl.constexpr = False,
):
    r"""Reduce a CUDA TensorIterator tile in native accumulation order."""
    lane = tl.arange(0, LANES)
    BASE_STEP: tl.constexpr = BW * (BH if SY else 1) if FAST else (BH if SY else 1)
    STEP = BASE_STEP * CTAS
    DTYPE: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    if PACKED:
        RANK: tl.constexpr = SHAPE.shape[0]
    else:
        RANK: tl.constexpr = len(SHAPE)
    row = cuda_native_sum_offset(
        output, SHAPE, ISTRIDES, RDIMS, RANK, PERMUTATION, PACKED
    )
    if FINAL:
        # CUDA's final pass accumulates one partial per thread per iteration.
        value = tl.full((LANES,), 0, DTYPE)
        for k in range(tl.cdiv(CTAS, BASE_STEP)):
            idx = lane + k * BASE_STEP
            valid = idx < CTAS
            incoming = tl.load(PARTIAL + output * PARTIAL_STRIDE + idx, valid, 0)
            value = tl.where(valid, value + incoming, value)
        if SY:
            value = cuda_native_sum_y_reduce(value, lane, BW, BH, FAST, LANES)
        if FAST:
            value = cuda_native_sum_x_reduce(value, lane, BW, LANES, DESCENDING)
    else:
        VT: tl.constexpr = IV if VI else 4
        slot = tl.arange(0, VT)[:, None]
        index = lane[None, :] + cta * BASE_STEP
        value_list = tl.full((VT, LANES), 0, DTYPE)
        if VI:
            if ALIGNMENT >= 0:
                alignment = ALIGNMENT
            else:
                alignment = (
                    (X + row).to(tl.uint64)
                    // (X.dtype.element_ty.primitive_bitwidth // 8)
                    % IV
                )
            prefix = tl.where(alignment > 0, IV - alignment, 0)
            tx = lane % BW if FAST else lane * 0
            ty = lane // BW if FAST else lane
            tail_owner = (ty == 0) & (cta == 0) if SY else cta == 0
            head = (alignment > 0) & (tx >= alignment) & (tx < IV) & tail_owner
            head_index = tx - alignment
            head_value = tl.load(
                X
                + row
                + cuda_native_sum_offset(
                    head_index, SHAPE, ISTRIDES, 0, RDIMS, PERMUTATION, PACKED
                ),
                head,
                0,
            ).to(DTYPE)
            value_list = tl.where(slot == 0, head_value[None, :], value_list)
            for k in range(tl.cdiv(N, STEP * IV)):
                vector_index = index + k * STEP
                ridx = prefix + vector_index * IV + slot
                valid = prefix + vector_index * IV + IV <= N
                incoming = tl.load(
                    X
                    + row
                    + cuda_native_sum_offset(
                        ridx, SHAPE, ISTRIDES, 0, RDIMS, PERMUTATION, PACKED
                    ),
                    valid,
                    0,
                ).to(DTYPE)
                value_list = tl.where(valid, value_list + incoming, value_list)
            tail_start = N - (N - prefix) % IV
            tail_index = tail_start + tx
            tail = (tail_index < N) & tail_owner
            tail_value = tl.load(
                X
                + row
                + cuda_native_sum_offset(
                    tail_index, SHAPE, ISTRIDES, 0, RDIMS, PERMUTATION, PACKED
                ),
                tail,
                0,
            ).to(DTYPE)
            value_list = tl.where(
                (slot == 0) & tail[None, :],
                value_list + tail_value[None, :],
                value_list,
            )
        else:
            for k in range(tl.cdiv(N, STEP * 4)):
                ridx = index + k * STEP * 4 + slot * STEP
                valid = ridx < N
                incoming = tl.load(
                    X
                    + row
                    + cuda_native_sum_offset(
                        ridx, SHAPE, ISTRIDES, 0, RDIMS, PERMUTATION, PACKED
                    ),
                    valid,
                    0,
                ).to(DTYPE)
                value_list = tl.where(valid, value_list + incoming, value_list)
        value = tl.gather(value_list, tl.full((1, LANES), 0, tl.int32), 0).reshape(
            (LANES,)
        )
        for s in tl.static_range(1, VT):
            other = tl.gather(value_list, tl.full((1, LANES), s, tl.int32), 0).reshape(
                (LANES,)
            )
            value = value + other
        if FAST:
            value = cuda_native_sum_x_reduce(value, lane, BW, LANES, DESCENDING)
        if SY:
            value = cuda_native_sum_y_reduce(value, lane, BW, BH, FAST, LANES)
    value = tl.gather(value, tl.full((1,), 0, tl.int32), 0).reshape(())
    return value


@triton.jit
def cuda_bn_divide(a, b):
    if a.dtype == tl.float64:
        return a / b
    else:
        return tl.div_rn(a, b)


@triton.jit
def cuda_bn_merge(avg, var, n, lane, LANES: tl.constexpr):
    for power in tl.static_range(0, 5):
        other = lane ^ (1 << power)
        oa = tl.gather(avg, other, 0)
        ov = tl.gather(var, other, 0)
        on = tl.gather(n, other, 0)
        factor = cuda_bn_divide(
            tl.full((LANES,), 1, tl.float64), tl.maximum(1, n + on).to(tl.float64)
        ).to(avg.dtype)
        delta = avg - oa
        product = delta * delta * n.to(avg.dtype) * on.to(avg.dtype)
        var = var + tl.fma(product, factor, ov)
        avg = tl.fma(n.to(avg.dtype), avg, on.to(avg.dtype) * oa) * factor
        n = n + on
    return avg, var, n


@triton.jit
def cuda_bn_statistics(
    X,
    channel,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    BW: tl.constexpr,
    BH: tl.constexpr,
):
    lane = tl.arange(0, 512)
    tx, ty = lane % BW, lane // BW
    DTYPE: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    avg = tl.full((512,), 0, DTYPE)
    var = tl.full((512,), 0, DTYPE)
    count = tl.full((512,), 0, tl.int32)
    for bi in range(triton.cdiv(N, BH)):
        batch = ty + bi * BH
        for si in range(triton.cdiv(S, BW)):
            sample = tx + si * BW
            valid = (batch < N) & (sample < S)
            value = tl.load(X + (batch * C + channel) * S + sample, valid, 0).to(DTYPE)
            delta = value - avg
            next_count = count + 1
            next_avg = avg + cuda_bn_divide(delta, next_count.to(DTYPE))
            next_var = tl.fma(delta, value - next_avg, var)
            avg = tl.where(valid, next_avg, avg)
            var = tl.where(valid, next_var, var)
            count = tl.where(valid, next_count, count)
    avg, var, count = cuda_bn_merge(avg, var, count, lane, 512)
    leaders = tl.arange(0, 32)
    avg = tl.gather(avg, (leaders * 32) % 512, 0)
    var = tl.gather(var, (leaders * 32) % 512, 0)
    count = tl.gather(count, (leaders * 32) % 512, 0)
    valid = leaders < 16
    avg = tl.where(valid, avg, 0)
    var = tl.where(valid, var, 0)
    count = tl.where(valid, count, 0)
    avg, var, count = cuda_bn_merge(avg, var, count, leaders, 32)
    avg = tl.gather(avg, tl.full((1,), 0, tl.int32), 0).reshape(())
    var = tl.gather(var, tl.full((1,), 0, tl.int32), 0).reshape(())
    variance = cuda_bn_divide(var, tl.full((), N * S, DTYPE))
    return avg, variance


@triton.jit
def cuda_welford_update(mean, m2, count, value, valid):
    next_count = count + 1
    delta = value - mean
    next_mean = mean + cuda_bn_divide(delta, next_count.to(mean.dtype))
    next_m2 = tl.fma(delta, value - next_mean, m2)
    return (
        tl.where(valid, next_mean, mean),
        tl.where(valid, next_m2, m2),
        tl.where(valid, next_count, count),
    )


@triton.jit
def cuda_welford_combine(am, av, ac, bm, bv, bc):
    delta = bm - am
    count = ac + bc
    ratio = cuda_bn_divide(bc, count)
    mean = tl.fma(delta, ratio, am)
    m2 = tl.fma(delta * delta * ac, ratio, av + bv)
    mean = tl.where(ac == 0, bm, tl.where(bc == 0, am, mean))
    m2 = tl.where(ac == 0, bv, tl.where(bc == 0, av, m2))
    return mean, m2, count


@triton.jit
def cuda_welford_tree(
    mean,
    m2,
    count,
    lane,
    BW: tl.constexpr,
    BH: tl.constexpr,
    FAST: tl.constexpr,
    LANES: tl.constexpr,
    Y: tl.constexpr,
):
    WIDTH: tl.constexpr = BH if Y else BW
    STRIDE: tl.constexpr = (BW if FAST else 1) if Y else 1
    for power in tl.static_range(0, 9):
        shift = WIDTH >> (power + 1)
        if shift > 0:
            other = (lane + shift * STRIDE) % LANES
            bm = tl.gather(mean, other, 0)
            bv = tl.gather(m2, other, 0)
            bc = tl.gather(count, other, 0)
            nm, nv, nc = cuda_welford_combine(mean, m2, count, bm, bv, bc)
            active = lane // STRIDE < shift if Y else lane % BW < shift
            mean = tl.where(active, nm, mean)
            m2 = tl.where(active, nv, m2)
            count = tl.where(active, nc, count)
    return mean, m2, count


@triton.jit
def cuda_native_welford(
    X,
    P,
    row,
    cta,
    SHAPE: tl.constexpr,
    INS: tl.constexpr,
    OUTS: tl.constexpr,
    RDIMS: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BW: tl.constexpr,
    BH: tl.constexpr,
    FAST: tl.constexpr,
    SY: tl.constexpr,
    CTAS: tl.constexpr,
    LANES: tl.constexpr,
    FINAL: tl.constexpr,
    VI: tl.constexpr,
    PARTIAL_STRIDE,
):
    r"""Combine CUDA variance and mean accumulators in native Welford order."""
    lane = tl.arange(0, LANES)
    BASE: tl.constexpr = BW * (BH if SY else 1) if FAST else (BH if SY else 1)
    STEP: tl.constexpr = BASE * CTAS
    DTYPE: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    if FINAL:
        mean = tl.full((LANES,), 0, DTYPE)
        m2 = tl.full((LANES,), 0, DTYPE)
        count = tl.full((LANES,), 0, DTYPE)
        for k in range(triton.cdiv(CTAS, BASE)):
            idx = lane + k * BASE
            valid = idx < CTAS
            bm = tl.load(P + row * PARTIAL_STRIDE + idx, valid, 0)
            bv = tl.load(P + (M + row) * PARTIAL_STRIDE + idx, valid, 0)
            bc = tl.load(P + (2 * M + row) * PARTIAL_STRIDE + idx, valid, 0)
            mean, m2, count = cuda_welford_combine(mean, m2, count, bm, bv, bc)
        if SY:
            mean, m2, count = cuda_welford_tree(
                mean, m2, count, lane, BW, BH, FAST, LANES, True
            )
        if FAST:
            mean, m2, count = cuda_welford_tree(
                mean, m2, count, lane, BW, BH, FAST, LANES, False
            )
    else:
        slots = tl.arange(0, 2)[:, None]
        mean_list = tl.full((2, LANES), 0, DTYPE)
        m2_list = tl.full((2, LANES), 0, DTYPE)
        count_list = tl.full((2, LANES), 0, tl.int32)
        base = cuda_native_sum_offset(row, SHAPE, INS, RDIMS, len(SHAPE))
        if VI:
            alignment = (
                (X + base).to(tl.uint64)
                // (X.dtype.element_ty.primitive_bitwidth // 8)
                % 2
            )
            prefix = tl.where(alignment > 0, 2 - alignment, 0)
            tx = lane % BW
            ty = lane // BW
            owner = ((ty == 0) & (cta == 0)) if SY else cta == 0
            head = (alignment > 0) & (tx >= alignment) & (tx < 2) & owner
            head_idx = tx - alignment
            head_value = tl.load(
                X + base + cuda_native_sum_offset(head_idx, SHAPE, INS, 0, RDIMS),
                head,
                0,
            ).to(DTYPE)
            mean_list, m2_list, count_list = cuda_welford_update(
                mean_list,
                m2_list,
                count_list,
                head_value[None, :],
                (slots == 0) & head[None, :],
            )
            for k in range(triton.cdiv(N, STEP * 2)):
                vector = lane[None, :] + cta * BASE + k * STEP
                idx = prefix + vector * 2 + slots
                valid = prefix + vector * 2 + 2 <= N
                value = tl.load(
                    X + base + cuda_native_sum_offset(idx, SHAPE, INS, 0, RDIMS),
                    valid,
                    0,
                ).to(DTYPE)
                mean_list, m2_list, count_list = cuda_welford_update(
                    mean_list, m2_list, count_list, value, valid
                )
            tail_idx = N - (N - prefix) % 2 + tx
            tail = (tail_idx < N) & owner
            tail_value = tl.load(
                X + base + cuda_native_sum_offset(tail_idx, SHAPE, INS, 0, RDIMS),
                tail,
                0,
            ).to(DTYPE)
            mean_list, m2_list, count_list = cuda_welford_update(
                mean_list,
                m2_list,
                count_list,
                tail_value[None, :],
                (slots == 0) & tail[None, :],
            )
        else:
            for k in range(triton.cdiv(N, STEP * 2)):
                idx = lane[None, :] + cta * BASE + k * STEP * 2 + slots * STEP
                valid = idx < N
                value = tl.load(
                    X + base + cuda_native_sum_offset(idx, SHAPE, INS, 0, RDIMS),
                    valid,
                    0,
                ).to(DTYPE)
                mean_list, m2_list, count_list = cuda_welford_update(
                    mean_list, m2_list, count_list, value, valid
                )
        zero = tl.full((1, LANES), 0, tl.int32)
        one = tl.full((1, LANES), 1, tl.int32)
        mean = tl.gather(mean_list, zero, 0).reshape((LANES,))
        m2 = tl.gather(m2_list, zero, 0).reshape((LANES,))
        count = tl.gather(count_list, zero, 0).reshape((LANES,)).to(DTYPE)
        bm = tl.gather(mean_list, one, 0).reshape((LANES,))
        bv = tl.gather(m2_list, one, 0).reshape((LANES,))
        bc = tl.gather(count_list, one, 0).reshape((LANES,)).to(DTYPE)
        mean, m2, count = cuda_welford_combine(mean, m2, count, bm, bv, bc)
        if FAST:
            mean, m2, count = cuda_welford_tree(
                mean, m2, count, lane, BW, BH, FAST, LANES, False
            )
        if SY:
            mean, m2, count = cuda_welford_tree(
                mean, m2, count, lane, BW, BH, FAST, LANES, True
            )
    zero = tl.full((1,), 0, tl.int32)
    mean = tl.gather(mean, zero, 0).reshape(())
    m2 = tl.gather(m2, zero, 0).reshape(())
    count = tl.gather(count, zero, 0).reshape(())
    return mean, m2, count


@triton.jit
def cuda_bn_channels_last_merge(am, av, ac, bm, bv, bc, LINEAR: tl.constexpr):
    dtype: tl.constexpr = am.dtype
    factor = cuda_bn_divide(tl.full((), 1, dtype), tl.maximum(1, ac + bc).to(dtype))
    delta = am - bm
    if LINEAR:
        mean = tl.fma(bm, bc.to(dtype), am * ac.to(dtype)) * factor
    else:
        mean = tl.fma(am, ac.to(dtype), bm * bc.to(dtype)) * factor
    m2 = av + tl.fma(delta * delta * bc.to(dtype) * ac.to(dtype), factor, bv)
    return mean, m2, ac + bc


@triton.jit
def cuda_bn_channels_last_vertical(mean, m2, count, lane, BH: tl.constexpr):
    for power in tl.static_range(0, 9):
        shift = BH >> (power + 1)
        if shift > 0:
            other = (lane + shift) % BH
            bm = tl.gather(mean, other, 0)
            bv = tl.gather(m2, other, 0)
            bc = tl.gather(count, other, 0)
            nm, nv, nc = cuda_bn_channels_last_merge(mean, m2, count, bm, bv, bc, False)
            mean = tl.where(lane < shift, nm, mean)
            m2 = tl.where(lane < shift, nv, m2)
            count = tl.where(lane < shift, nc, count)
    return mean, m2, count


@triton.jit
def cuda_bn_channels_last_statistics(
    X,
    P,
    row,
    cta,
    N: tl.constexpr,
    C: tl.constexpr,
    BH: tl.constexpr,
    CTAS: tl.constexpr,
    FINAL: tl.constexpr,
):
    lane = tl.arange(0, BH)
    dtype: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    if FINAL:
        mean = tl.full((BH,), 0, dtype)
        m2 = tl.full((BH,), 0, dtype)
        count = tl.full((BH,), 0, tl.int32)
        for k in range(triton.cdiv(CTAS, BH)):
            idx = lane + k * BH
            valid = idx < CTAS
            bm = tl.load(P + row * CTAS + idx, valid, 0)
            bv = tl.load(P + (C + row) * CTAS + idx, valid, 0)
            bc = tl.load(P + (2 * C + row) * CTAS + idx, valid, 0).to(tl.int32)
            nm, nv, nc = cuda_bn_channels_last_merge(mean, m2, count, bm, bv, bc, False)
            mean = tl.where(valid, nm, mean)
            m2 = tl.where(valid, nv, m2)
            count = tl.where(valid, nc, count)
    else:
        slots = tl.arange(0, 4)[:, None]
        mean_list = tl.full((4, BH), 0, dtype)
        m2_list = tl.full((4, BH), 0, dtype)
        count_list = tl.full((4, BH), 0, tl.int32)
        for k in range(triton.cdiv(N, BH * CTAS * 4)):
            idx = cta * BH + lane[None, :] + (k * 4 + slots) * BH * CTAS
            valid = idx < N
            value = tl.load(X + idx * C + row, valid, 0).to(dtype)
            count_list += valid.to(tl.int32)
            inv = tl.where(
                valid, cuda_bn_divide(tl.full((), 1, dtype), count_list.to(dtype)), 0
            )
            delta = value - mean_list
            mean_list = tl.fma(delta, inv, mean_list)
            m2_list = tl.fma(delta * (value - mean_list), valid.to(dtype), m2_list)
        zero = tl.full((1, BH), 0, tl.int32)
        mean = tl.gather(mean_list, zero, 0).reshape((BH,))
        m2 = tl.gather(m2_list, zero, 0).reshape((BH,))
        count = tl.gather(count_list, zero, 0).reshape((BH,))
        for slot in tl.static_range(1, 4):
            index = tl.full((1, BH), slot, tl.int32)
            bm = tl.gather(mean_list, index, 0).reshape((BH,))
            bv = tl.gather(m2_list, index, 0).reshape((BH,))
            bc = tl.gather(count_list, index, 0).reshape((BH,))
            mean, m2, count = cuda_bn_channels_last_merge(
                mean, m2, count, bm, bv, bc, True
            )
    mean, m2, count = cuda_bn_channels_last_vertical(mean, m2, count, lane, BH)
    zero = tl.full((1,), 0, tl.int32)
    mean = tl.gather(mean, zero, 0).reshape(())
    m2 = tl.gather(m2, zero, 0).reshape(())
    count = tl.gather(count, zero, 0).reshape(())
    return mean, m2, count


@triton.jit
def cuda_bn_backward_warp(value, lane, LANES: tl.constexpr, INPUT: tl.constexpr):
    for power in tl.static_range(0, 5):
        shift = 16 >> power
        other = (lane + shift) % LANES
        partner = tl.gather(value, other, 0)
        value = tl.where(lane % 32 + shift < 32, value + partner, value)
    return value


@triton.jit
def cuda_bn_backward_fused_statistics(
    X,
    G,
    MEAN,
    channel,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    BW: tl.constexpr,
    BH: tl.constexpr,
    FUSED: tl.constexpr,
):
    THREADS: tl.constexpr = BW * BH
    lane = tl.arange(0, THREADS)
    tx = lane % BW
    ty = lane // BW
    dtype: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    input: tl.constexpr = X.dtype.element_ty
    mean = tl.load(MEAN + channel)
    dy = tl.full((THREADS,), 0, dtype)
    dot = tl.full((THREADS,), 0, dtype)
    for bi in range(triton.cdiv(N, BH)):
        batch = (ty + bi * BH).to(tl.int64)
        for si in range(triton.cdiv(S, BW)):
            sample = tx + si * BW
            valid = (batch < N) & (sample < S)
            index = (batch * C + channel) * S + sample
            x = tl.load(X + index, valid, 0).to(dtype)
            g = tl.load(G + index, valid, 0).to(dtype)
            dy = tl.where(valid, dy + g, dy)
            delta = x - mean
            if FUSED:
                dot = tl.where(valid, tl.fma(g, delta, dot), dot)
            else:
                product = g * delta
                dot = tl.where(valid, dot + product, dot)
    dy = cuda_bn_backward_warp(dy, lane, THREADS, input)
    dot = cuda_bn_backward_warp(dot, lane, THREADS, input)
    leaders = tl.arange(0, 32)
    dy = tl.gather(dy, (leaders * 32) % THREADS, 0)
    dot = tl.gather(dot, (leaders * 32) % THREADS, 0)
    dy = tl.where(leaders < THREADS // 32, dy, 0)
    dot = tl.where(leaders < THREADS // 32, dot, 0)
    dy = cuda_bn_backward_warp(dy, leaders, 32, input)
    dot = cuda_bn_backward_warp(dot, leaders, 32, input)
    zero = tl.full((1,), 0, tl.int32)
    dy = tl.gather(dy, zero, 0).reshape(())
    dot = tl.gather(dot, zero, 0).reshape(())
    return dy, dot


@triton.jit
def cuda_bn_backward_channels_last_vertical(dy, dot, lane, BH: tl.constexpr):
    for power in tl.static_range(0, 9):
        shift = BH >> (power + 1)
        if shift > 0:
            other = (lane + shift) % BH
            bd = tl.gather(dy, other, 0)
            bp = tl.gather(dot, other, 0)
            dy = tl.where(lane < shift, dy + bd, dy)
            dot = tl.where(lane < shift, dot + bp, dot)
    return dy, dot


@triton.jit
def cuda_bn_backward_channels_last_statistics(
    X,
    G,
    MEAN,
    P,
    row,
    cta,
    N: tl.constexpr,
    C: tl.constexpr,
    BH: tl.constexpr,
    CTAS: tl.constexpr,
    FINAL: tl.constexpr,
):
    lane = tl.arange(0, BH)
    dtype: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    if FINAL:
        dy = tl.full((BH,), 0, dtype)
        dot = tl.full((BH,), 0, dtype)
        for k in range(triton.cdiv(CTAS, BH)):
            idx = lane + k * BH
            valid = idx < CTAS
            bd = tl.load(P + row * CTAS + idx, valid, 0)
            bp = tl.load(P + (C + row) * CTAS + idx, valid, 0)
            dy += bd
            dot += bp
    else:
        mean = tl.load(MEAN + row)
        slots = tl.arange(0, 4)[:, None]
        dy_list = tl.full((4, BH), 0, dtype)
        dot_list = tl.full((4, BH), 0, dtype)
        for k in range(triton.cdiv(N, BH * CTAS * 4)):
            idx = cta * BH + lane[None, :] + (k * 4 + slots) * BH * CTAS
            valid = idx < N
            value = tl.load(X + idx * C + row, valid, 0).to(dtype)
            grad = tl.load(G + idx * C + row, valid, 0).to(dtype)
            dy_list += grad
            dot_list = tl.fma(grad, value - mean, dot_list)
        zero = tl.full((1, BH), 0, tl.int32)
        dy = tl.gather(dy_list, zero, 0).reshape((BH,))
        dot = tl.gather(dot_list, zero, 0).reshape((BH,))
        for slot in tl.static_range(1, 4):
            index = tl.full((1, BH), slot, tl.int32)
            dy += tl.gather(dy_list, index, 0).reshape((BH,))
            dot += tl.gather(dot_list, index, 0).reshape((BH,))
    dy, dot = cuda_bn_backward_channels_last_vertical(dy, dot, lane, BH)
    zero = tl.full((1,), 0, tl.int32)
    dy = tl.gather(dy, zero, 0).reshape(())
    dot = tl.gather(dot, zero, 0).reshape(())
    return dy, dot


@triton.jit
def cuda_native_mv_value(
    X,
    VECTOR,
    PARTIAL,
    row,
    cta,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    VS: tl.constexpr,
    CTAS: tl.constexpr,
    THREADS: tl.constexpr,
    ACTIVE_THREADS: tl.constexpr,
    INNER_CHUNK: tl.constexpr,
    INNER_STRIDE: tl.constexpr,
    KIND: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    FINAL: tl.constexpr,
    PARTIAL_CTAS: tl.constexpr,
):
    r"""Compute a real GEMV partial or final value in native reduction order."""
    if FINAL and KIND != 2 and KIND != 3:
        value = tl.load(PARTIAL + row * PARTIAL_CTAS)
    else:
        if FINAL and KIND == 3:
            value = tl.full((1,), 0, ACC_TYPE)
            for index in range(CTAS):
                value += tl.load(PARTIAL + row * PARTIAL_CTAS + index)
        elif FINAL and KIND == 2:
            lane = tl.arange(0, THREADS)
            total = tl.full((THREADS,), 0, ACC_TYPE)
            for chunk in range(triton.cdiv(CTAS, THREADS)):
                index = lane + chunk * THREADS
                total += tl.load(PARTIAL + row * PARTIAL_CTAS + index, index < CTAS, 0)
            for shift in tl.static_range(THREADS // 2, 16, -32):
                partner = tl.gather(total, (lane + shift) % THREADS, 0)
                total = tl.where(lane < shift, total + partner, total)
            group = tl.arange(0, 4)
            sums = tl.full((4,), 0, ACC_TYPE)
            for index in tl.static_range(8):
                sums += tl.gather(total, group + index * 4, 0)
            value = tl.full((1,), 0, ACC_TYPE)
            for index in tl.static_range(4):
                value += tl.gather(sums, tl.full((1,), index, tl.int32), 0)
        else:
            lane = tl.arange(0, THREADS)
            total = tl.full((THREADS,), 0, ACC_TYPE)
            if KIND == 2:
                pair: tl.constexpr = S1 == 1 and VS == 1 and ACC_TYPE != tl.float64
                second = tl.full((THREADS,), 0, ACC_TYPE)
                for chunk in range(triton.cdiv(K, THREADS * CTAS * (2 if pair else 1))):
                    index = (lane + THREADS * (cta + CTAS * chunk)) * (2 if pair else 1)
                    valid = index < K
                    x = tl.load(X + row * S0 + index * S1, valid, 0).to(ACC_TYPE)
                    vector = tl.load(VECTOR + index * VS, valid, 0).to(ACC_TYPE)
                    total = tl.where(valid, tl.fma(x, vector, total), total)
                    if pair:
                        valid = index + 1 < K
                        x = tl.load(X + row * S0 + (index + 1) * S1, valid, 0).to(
                            ACC_TYPE
                        )
                        vector = tl.load(VECTOR + (index + 1) * VS, valid, 0).to(
                            ACC_TYPE
                        )
                        second = tl.where(valid, tl.fma(x, vector, second), second)
                if pair:
                    total = (tl.full((THREADS,), 0, ACC_TYPE) + total) + second
            elif KIND == 3:
                CHUNK: tl.constexpr = triton.cdiv(K, CTAS)
                for step in range(triton.cdiv(CHUNK, THREADS)):
                    index = cta * CHUNK + lane + step * THREADS
                    valid = (index < K) & (lane + step * THREADS < CHUNK)
                    x = tl.load(X + row * S0 + index * S1, valid, 0).to(ACC_TYPE)
                    vector = tl.load(VECTOR + index * VS, valid, 0).to(ACC_TYPE)
                    total = tl.where(valid, tl.fma(x, vector, total), total)
            else:
                STEP: tl.constexpr = INNER_CHUNK * (
                    ACTIVE_THREADS if INNER_STRIDE == 1 else INNER_STRIDE
                )
                for chunk in range(triton.cdiv(K, STEP)):
                    for step in range(INNER_CHUNK):
                        index = (
                            lane * (INNER_CHUNK if INNER_STRIDE == 1 else 1)
                            + chunk * STEP
                            + step * INNER_STRIDE
                        )
                        valid = (index < K) & (lane < ACTIVE_THREADS)
                        x = tl.load(X + row * S0 + index * S1, valid, 0).to(ACC_TYPE)
                        vector = tl.load(VECTOR + index * VS, valid, 0).to(ACC_TYPE)
                        total = tl.where(valid, tl.fma(x, vector, total), total)
            total = cuda_native_sum_x_reduce(total, lane, THREADS, THREADS, True)
            value = tl.gather(total, tl.full((1,), 0, tl.int32), 0)
    if FINAL:
        value = tl.fma(value, 1.0, 0.0)
    return value


@triton.jit
def round_to_odd_float32(x):
    truncated = libdevice.double2float_rz(x)
    inexact = (truncated.to(tl.float64) != x) & (x == x)
    bits = truncated.to(tl.uint32, bitcast=True) | inexact.to(tl.uint32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def cuda_native_complex_mv_value(
    X,
    VECTOR,
    PARTIAL,
    row,
    cta,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    VS: tl.constexpr,
    CTAS: tl.constexpr,
    THREADS: tl.constexpr,
    ACTIVE_THREADS: tl.constexpr,
    INNER_CHUNK: tl.constexpr,
    INNER_STRIDE: tl.constexpr,
    KIND: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    FINAL: tl.constexpr,
    PARTIAL_CTAS: tl.constexpr,
    SWAP: tl.constexpr,
    PRODUCT: tl.constexpr,
    PRE_SCALE: tl.constexpr,
):
    r"""Compute a complex GEMV partial or final value in native reduction order."""
    if FINAL and KIND != 2 and KIND != 3:
        real = tl.load(PARTIAL + 2 * row * PARTIAL_CTAS)
        imag = tl.load(PARTIAL + 2 * row * PARTIAL_CTAS + 1)
    else:
        if FINAL and KIND == 3:
            real = tl.full((1,), 0, ACC_TYPE)
            imag = tl.full((1,), 0, ACC_TYPE)
            for index in range(CTAS):
                real += tl.load(PARTIAL + 2 * (row * PARTIAL_CTAS + index))
                imag += tl.load(PARTIAL + 2 * (row * PARTIAL_CTAS + index) + 1)
        elif FINAL and KIND == 2:
            lane = tl.arange(0, THREADS)
            total_real = tl.full((THREADS,), 0, ACC_TYPE)
            for chunk_real in range(triton.cdiv(CTAS, THREADS)):
                index_real = lane + chunk_real * THREADS
                total_real += tl.load(
                    PARTIAL + 2 * (row * PARTIAL_CTAS + index_real) + 0,
                    index_real < CTAS,
                    0,
                )
            for shift_real in tl.static_range(THREADS // 2, 16, -32):
                partner_real = tl.gather(total_real, (lane + shift_real) % THREADS, 0)
                total_real = tl.where(
                    lane < shift_real, total_real + partner_real, total_real
                )
            group = tl.arange(0, 4)
            sums_real = tl.full((4,), 0, ACC_TYPE)
            for index_real in tl.static_range(8):
                sums_real += tl.gather(total_real, group + index_real * 4, 0)
            real = tl.full((1,), 0, ACC_TYPE)
            for index_real in tl.static_range(4):
                real += tl.gather(sums_real, tl.full((1,), index_real, tl.int32), 0)
            total_imag = tl.full((THREADS,), 0, ACC_TYPE)
            for chunk_imag in range(triton.cdiv(CTAS, THREADS)):
                index_imag = lane + chunk_imag * THREADS
                total_imag += tl.load(
                    PARTIAL + 2 * (row * PARTIAL_CTAS + index_imag) + 1,
                    index_imag < CTAS,
                    0,
                )
            for shift_imag in tl.static_range(THREADS // 2, 16, -32):
                partner_imag = tl.gather(total_imag, (lane + shift_imag) % THREADS, 0)
                total_imag = tl.where(
                    lane < shift_imag, total_imag + partner_imag, total_imag
                )
            group = tl.arange(0, 4)
            sums_imag = tl.full((4,), 0, ACC_TYPE)
            for index_imag in tl.static_range(8):
                sums_imag += tl.gather(total_imag, group + index_imag * 4, 0)
            imag = tl.full((1,), 0, ACC_TYPE)
            for index_imag in tl.static_range(4):
                imag += tl.gather(sums_imag, tl.full((1,), index_imag, tl.int32), 0)
        else:
            lane = tl.arange(0, THREADS)
            total_real = tl.full((THREADS,), 0, ACC_TYPE)
            total_imag = tl.full((THREADS,), 0, ACC_TYPE)
            if KIND == 2:
                CHUNK: tl.constexpr = K
                STEP: tl.constexpr = THREADS * CTAS
                INNER: tl.constexpr = 1
            elif KIND == 3:
                CHUNK: tl.constexpr = triton.cdiv(K, CTAS)
                STEP: tl.constexpr = INNER_CHUNK * (
                    ACTIVE_THREADS if INNER_STRIDE == 1 else INNER_STRIDE
                )
                INNER: tl.constexpr = INNER_CHUNK
            else:
                CHUNK: tl.constexpr = K
                STEP: tl.constexpr = INNER_CHUNK * (
                    THREADS if INNER_STRIDE == 1 else INNER_STRIDE
                )
                INNER: tl.constexpr = INNER_CHUNK
            for chunk in range(triton.cdiv(CHUNK, STEP)):
                for step in range(INNER):
                    if KIND == 2:
                        index = lane + THREADS * cta + chunk * STEP
                    elif KIND == 3:
                        index = (
                            cta * CHUNK
                            + lane * (INNER_CHUNK if INNER_STRIDE == 1 else 1)
                            + chunk * STEP
                            + step * INNER_STRIDE
                        )
                    else:
                        index = (
                            lane * (INNER_CHUNK if INNER_STRIDE == 1 else 1)
                            + chunk * STEP
                            + step * INNER_STRIDE
                        )
                    valid = index < K
                    if KIND == 3:
                        valid = valid & (index - cta * CHUNK < CHUNK)
                    xr = tl.load(X + 2 * (row * S0 + index * S1), valid, 0)
                    xi = tl.load(X + 2 * (row * S0 + index * S1) + 1, valid, 0)
                    vr = tl.load(VECTOR + 2 * index * VS, valid, 0)
                    vi = tl.load(VECTOR + 2 * index * VS + 1, valid, 0)
                    if PRE_SCALE:
                        vr, vi = tl.fma(vr, 1.0, -vi * 0.0), tl.fma(vi, 1.0, vr * 0.0)
                    if PRODUCT:
                        next_real = total_real + tl.fma(xr, vr, -xi * vi)
                        next_imag = total_imag + tl.fma(xr, vi, xi * vr)
                    else:
                        next_real = tl.fma(-xi, vi, tl.fma(xr, vr, total_real))
                        if SWAP:
                            next_imag = tl.fma(xr, vi, tl.fma(xi, vr, total_imag))
                        else:
                            next_imag = tl.fma(xi, vr, tl.fma(xr, vi, total_imag))
                    total_real = tl.where(valid, next_real, total_real)
                    total_imag = tl.where(valid, next_imag, total_imag)
            total_real = cuda_native_sum_x_reduce(
                total_real, lane, THREADS, THREADS, True
            )
            total_imag = cuda_native_sum_x_reduce(
                total_imag, lane, THREADS, THREADS, True
            )
            real = tl.gather(total_real, tl.full((1,), 0, tl.int32), 0)
            imag = tl.gather(total_imag, tl.full((1,), 0, tl.int32), 0)
    if FINAL and not PRE_SCALE:
        real, imag = tl.fma(real, 1.0, -imag * 0.0), tl.fma(imag, 1.0, real * 0.0)
    return real, imag


@triton.jit
def cuda_native_mv_mma_value(
    X,
    VECTOR,
    cta,
    M: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    VS: tl.constexpr,
    K_SLICE: tl.constexpr,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    column = tl.arange(0, 16)
    inner = tl.arange(0, 64)
    total = tl.full((16, 16), 0, tl.float32)
    for chunk in range(triton.cdiv(K_SLICE, 64)):
        index = cta * K_SLICE + chunk * 64 + inner
        valid = (index < K) & (chunk * 64 + inner < K_SLICE)
        matrix = tl.load(
            X + row[:, None].to(tl.int64) * S0 + index[None, :].to(tl.int64) * S1,
            (row[:, None] < M) & valid[None, :],
            0,
        )
        vector = tl.load(
            VECTOR + index[:, None].to(tl.int64) * VS + tl.zeros((1, 16), tl.int64),
            valid[:, None],
            0,
        )
        total = tl.dot(matrix, vector, total)
    value = tl.sum(tl.where(column[None, :] == 0, total, 0), 1)
    return value
