from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase
from contextlib import contextmanager
from unittest.mock import patch
import torch
import re

def test_cpu():
    try:
        CppCodeCache.load("")
        return not IS_FBCODE
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False

HAS_CPU = LazyVal(test_cpu)

HAS_CUDA = has_triton()

@register_backend
def count_bytes_inductor(gm, example_inputs):
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)

def _check_has_dynamic_shape(
    self: TestCase,
    code,
):
    for_loop_found = False
    has_dynamic = False
    lines = code.split("\n")
    for line in lines:
        if "for(" in line:
            for_loop_found = True
            if re.search(r";.*ks.*;", line) is not None:
                has_dynamic = True
                break
    self.assertTrue(
        has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
    )
    self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")

@contextmanager
def _temporary_log_handler(logger, handler, level):
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)
        logger.removeHandler(handler)

def run_and_get_code(fn, *args, **kwargs):
    """
    Run fn and return the result along with the generated code.
    """
    from torch._inductor import config

    # We use the patch context manager instead of using it as a decorator.
    # In this way, we can ensure that the attribute is patched and unpatched correctly
    # even if this run_and_get_code function is called multiple times.
    with patch.object(config, "debug", True):
        torch._dynamo.reset()
        import io
        import logging
        from torch._inductor.graph import output_code_log

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        with _temporary_log_handler(output_code_log, ch, logging.DEBUG):
            result = fn(*args, **kwargs)
            s = log_capture_string.getvalue()
    return result, s

def run_and_get_multiple_code(fn, *args, **kwargs):
    """
    Similar to run_and_get_code(), but split the logged code by output file. Useful for separating code from the forward
    and backward passes.
    """
    result, s = run_and_get_code(fn, *args, **kwargs)
    source_codes = s.split("Output code:")
    assert source_codes[0] == ""
    return result, source_codes[1:]
