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
def _aten_bessel_poly(t, coefficients: tl.constexpr):
    # Cephes seeds each Horner chain with 0 and folds coefficients[0] in with an
    # extra fma; that first step is exact for the finite t of the branch it belongs
    # to, so start from the constant instead.
    result = tl.full(t.shape, coefficients[0], tl.float32)
    for i in tl.static_range(1, len(coefficients)):
        result = tl.fma(result, t, coefficients[i])
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

    small = tl.div_rn(
        (xx - 5.78318596294678452118)
        * (xx - 3.04712623436620863991e01)
        * _aten_bessel_poly(xx, rp),
        _aten_bessel_poly(xx, rq),
    )
    small = tl.where(ax < 0.00001, tl.fma(xx, -0.25, 1.0), small)

    t = tl.div_rn(tl.full(x.shape, 25.0, tl.float32), xx)
    w = ax - 0.785398163397448309615660845819875721
    a = tl.div_rn(_aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)) * libdevice.cos(w)
    b = tl.div_rn(tl.full(x.shape, 5.0, tl.float32), ax) * tl.div_rn(
        _aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq)
    )
    # ptxas contracts eager's `a - b * sin(w)` into a single FFMA; strict numerics
    # compiles Triton with enable_fp_fusion=False, so spell the fma out.
    large = tl.div_rn(
        tl.fma(-b, libdevice.sin(w), a) * 0.797884560802865355879892119868763737,
        tl.sqrt_rn(ax),
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
        tl.div_rn(_aten_bessel_poly(xx, rp), _aten_bessel_poly(xx, rq))
        * ax
        * (xx - 1.46819706421238932572e01)
        * (xx - 4.92184563216946036703e01)
    )

    # j1 reduces in 5/x, j0 in 25/(x*x); the two round differently, so keep both.
    u = tl.div_rn(tl.full(x.shape, 5.0, tl.float32), ax)
    t = u * u
    w = ax - 2.356194490192344928846982537459627163
    a = tl.div_rn(_aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)) * libdevice.cos(w)
    b = u * tl.div_rn(_aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq))
    large = tl.div_rn(
        tl.fma(-b, libdevice.sin(w), a) * 0.797884560802865355879892119868763737,
        tl.sqrt_rn(ax),
    )

    result = tl.where(ax <= 5.0, small, large)
    # Eager reflects odd-symmetrically with PTX neg.f32, which flips the sign bit and
    # canonicalizes NaN. Triton's unary minus lowers to `0 - v`, leaving +0.0 positive,
    # so flip the bit directly and keep the subtraction only for the NaN case.
    negated = tl.where(
        result == result,
        (
            result.to(tl.uint32, bitcast=True)
            ^ tl.full(result.shape, 0x80000000, tl.uint32)
        ).to(tl.float32, bitcast=True),
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
        libdevice.log(x) * 0.636619772367581343075535053490057448,
        aten_bessel_j0(x),
        tl.div_rn(_aten_bessel_poly(xx, yp), _aten_bessel_poly(xx, yq)),
    )
    small = tl.where(x == 0.0, float("-inf"), small)

    t = tl.div_rn(tl.full(x.shape, 25.0, tl.float32), xx)
    w = x - 0.785398163397448309615660845819875721
    a = tl.div_rn(_aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)) * libdevice.sin(w)
    b = tl.div_rn(tl.full(x.shape, 5.0, tl.float32), x) * tl.div_rn(
        _aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq)
    )
    large = tl.div_rn(
        tl.fma(b, libdevice.cos(w), a) * 0.797884560802865355879892119868763737,
        tl.sqrt_rn(x),
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

    # Math.cuh:1871 does `return NAN;` for x < 0, unlike y0's dropped return, but
    # log(x) already yields that same canonical NaN here, so no select is needed.
    small = tl.fma(
        x,
        tl.div_rn(_aten_bessel_poly(xx, yp), _aten_bessel_poly(xx, yq)),
        tl.fma(
            libdevice.log(x),
            aten_bessel_j1(x),
            tl.div_rn(tl.full(x.shape, -1.0, tl.float32), x),
        )
        * 0.636619772367581343075535053490057448,
    )
    small = tl.where(x == 0.0, float("-inf"), small)

    u = tl.div_rn(tl.full(x.shape, 5.0, tl.float32), x)
    t = u * u
    w = x - 2.356194490192344928846982537459627163
    a = tl.div_rn(_aten_bessel_poly(t, pp), _aten_bessel_poly(t, pq)) * libdevice.sin(w)
    b = u * tl.div_rn(_aten_bessel_poly(t, qp), _aten_bessel_poly(t, qq))
    large = tl.div_rn(
        tl.fma(b, libdevice.cos(w), a) * 0.797884560802865355879892119868763737,
        tl.sqrt_rn(x),
    )
    return tl.where(x <= 5.0, small, large)


@triton.jit
def aten_log_ndtr(x):
    t = x * 0.707106781186547524400844362104849039
    log_term = libdevice.log(libdevice.erfcx(-t) * 0.5)
    left = tl.fma(-t, t, log_term)
    right = libdevice.log1p(-libdevice.erfc(t) * 0.5)
    result = tl.where(x < -1.0, left, right)
    negative = tl.full(x.shape, -1.0, x.dtype)
    return tl.where(libdevice.isnan(x), result, libdevice.copysign(result, negative))


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


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

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    # Should be
    #   out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    # but since rhs_sum is all 1, we can simplify it.
    out_sum = lhs_sum * lhs_scale + rhs_scale
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
    rhs_max, rhs_sum = online_softmax_reduce(
        rhs, tl.where(rhs_mask, 1.0, 0.0), dim, use_fast_math, strict_signed_zero
    )
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

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
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
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    # Guard against inf - inf = NaN when both means are infinite and equal.
    # This occurs during FP16/BF16 LayerNorm when inputs overflow to inf.
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
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

    return 1.0 - (rand_int.to(tl.float32) * inv + half)


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
def eager_unary_nan(x, result):
    # CUDA abs, neg and frexp canonicalize NaNs except float64, which only quiets them.
    if result.dtype == tl.float64:
        bits = x.to(tl.int64, bitcast=True)
        is_nan = (bits & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        result = tl.where(
            is_nan, bits | 0x0008000000000000, result.to(tl.int64, bitcast=True)
        ).to(tl.float64, bitcast=True)
    elif (
        result.dtype == tl.float32
        or result.dtype == tl.float16
        or result.dtype == tl.bfloat16
    ):
        width: tl.constexpr = result.dtype.primitive_bitwidth
        idtype = tl.core.get_int_dtype(bitwidth=width, signed=True)
        nan_bits = tl.full((), (1 << (width - 1)) - 1, idtype)
        result = tl.where(x != x, nan_bits, result.to(idtype, bitcast=True)).to(
            result.dtype, bitcast=True
        )
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
