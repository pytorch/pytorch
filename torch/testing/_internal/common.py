# mypy: ignore-errors

r"""Common test utilities shared across device types (CUDA, XPU, etc.).

This file provides TF32-related test helpers that work for both CUDA and XPU
devices. It is intended to be imported by device-specific common files (e.g.
common_cuda.py) so that existing import sites need not change.
"""

import contextlib
import functools
import inspect
import threading

import torch


# ---------------------------------------------------------------------------
# TF32 context managers
# ---------------------------------------------------------------------------


_tf32_off_lock = threading.Lock()
_tf32_off_depth = 0
_tf32_off_saved_precision = None
_tf32_off_cudnn_ctx = None
_tf32_off_mkldnn_ctx = None


@contextlib.contextmanager
def tf32_off():
    """Context manager that disables TF32 for both CUDA (cuBLAS/cuDNN) and
    XPU (oneDNN/mkldnn) for the duration of the ``with`` block."""
    # First-in saves state and disables TF32; last-out restores it. Nested
    # contexts can otherwise restore the process-global precision state in the
    # wrong order when tests enter this context from multiple threads.
    global _tf32_off_depth, _tf32_off_saved_precision
    global _tf32_off_cudnn_ctx, _tf32_off_mkldnn_ctx
    with _tf32_off_lock:
        if _tf32_off_depth == 0:
            # Save fp32_precision rather than allow_tf32 so that the ``None``
            # default is restored exactly instead of being changed to ``ieee``.
            _tf32_off_saved_precision = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.allow_tf32 = False
            _tf32_off_cudnn_ctx = torch.backends.cudnn.flags(
                enabled=None, benchmark=None, deterministic=None, allow_tf32=False
            )
            _tf32_off_cudnn_ctx.__enter__()
            _tf32_off_mkldnn_ctx = torch.backends.mkldnn.flags(
                enabled=None,
                deterministic=None,
                allow_tf32=False,
                fp32_precision=None,
            )
            _tf32_off_mkldnn_ctx.__enter__()
        _tf32_off_depth += 1
    try:
        yield
    finally:
        with _tf32_off_lock:
            _tf32_off_depth -= 1
            if _tf32_off_depth == 0:
                _tf32_off_mkldnn_ctx.__exit__(None, None, None)
                _tf32_off_mkldnn_ctx = None
                _tf32_off_cudnn_ctx.__exit__(None, None, None)
                _tf32_off_cudnn_ctx = None
                torch.backends.cuda.matmul.fp32_precision = _tf32_off_saved_precision
                _tf32_off_saved_precision = None


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    """Context manager that enables TF32 for both CUDA and XPU, and
    temporarily lowers the test's precision threshold to *tf32_precision*."""
    import os

    old_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    old_precision = self.precision
    # ROCm uses an environment variable to enable TF32 in hipBLASLt
    hip_allow_tf32 = None
    if torch.version.hip:
        hip_allow_tf32 = os.environ.get("HIPBLASLT_ALLOW_TF32", None)
        os.environ["HIPBLASLT_ALLOW_TF32"] = "1"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            with torch.backends.mkldnn.flags(
                enabled=None,
                deterministic=None,
                allow_tf32=True,
                fp32_precision=None,
            ):
                yield
    finally:
        if torch.version.hip:
            if hip_allow_tf32 is not None:
                os.environ["HIPBLASLT_ALLOW_TF32"] = hip_allow_tf32
            else:
                del os.environ["HIPBLASLT_ALLOW_TF32"]
        torch.backends.cuda.matmul.fp32_precision = old_fp32_precision
        self.precision = old_precision


@contextlib.contextmanager
def tf32_enabled():
    """Context manager to temporarily enable TF32 for both CUDA and XPU
    operations. The previous backend TF32 state is automatically restored when
    exiting the context."""
    old_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            with torch.backends.mkldnn.flags(
                enabled=None,
                deterministic=None,
                allow_tf32=True,
                fp32_precision=None,
            ):
                yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_fp32_precision


# ---------------------------------------------------------------------------
# TF32 test decorator
# ---------------------------------------------------------------------------


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as specified by the
# argument. For example:
#    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
#    @tf32_on_and_off(0.005)
#    def test_matmul(self, device, dtype):
#        a = ...; b = ...;
#        c = torch.matmul(a, b)
#        self.assertEqual(c, expected)
# In the above example, when testing torch.float32 and torch.complex64 on a
# CUDA Ampere (or newer) device or an XPU device that supports TF32 via DPAS,
# the matmul will be run in TF32 mode and with TF32 disabled, and on TF32 mode
# the assertEqual will use reduced precision to check values.
#
# This decorator can be used for functions with or without device/dtype:
# @tf32_on_and_off(0.005)
# def test_my_op(self)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device, dtype)
# @tf32_on_and_off(0.005)
# def test_my_op(self, dtype)
#
# - If neither device nor dtype is specified, checks whether the system has an
#   Ampere (CUDA) or TF32-capable XPU device.
# - If device is specified, checks whether device type is 'cuda' or 'xpu'.
# - If dtype is specified, checks whether dtype is float32 or complex64.
# TF32 and fp32 differ only when all three checks pass.
def tf32_on_and_off(tf32_precision=1e-5, *, only_if=True):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            kwargs.update(zip(arg_names, args, strict=False))
            # Condition: either CUDA or XPU must support TF32
            cuda_tf32 = torch.cuda.is_tf32_supported()
            xpu_tf32 = (
                torch.xpu.is_tf32_supported()
                if hasattr(torch.xpu, "is_tf32_supported")
                else False
            )
            cond = (cuda_tf32 or xpu_tf32) and only_if
            device = kwargs.get(
                "device",
                getattr(kwargs.get("self"), "device", None),
            )
            if device is not None:
                dev_type = torch.device(device).type
                cond = cond and (dev_type in {"cuda", "xpu"})
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] in {torch.float32, torch.complex64})
            if cond:
                with_tf32_disabled(kwargs["self"], lambda: f(**kwargs))
                with_tf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper


def with_tf32_off(f):
    """Decorator that runs the wrapped test with TF32 disabled for both CUDA
    and XPU.  Use this when a test exercises matmul/convolutions as a side
    effect but its correctness should not depend on TF32 precision."""

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped
