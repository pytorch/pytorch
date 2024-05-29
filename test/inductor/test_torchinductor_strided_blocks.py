import importlib
import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    HAS_GPU,
    GPU_TYPE,
    requires_gpu,
    skip_windows_ci,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch._inductor.utils import run_and_get_code
from typing import Any, Callable, Tuple, Type, Union
import unittest

skip_windows_ci(__name__, __file__)

importlib.import_module("filelock")


@instantiate_parametrized_tests
class TritonBlockPointerTest(InductorTestCase):

    def run_and_compare(self, func: Callable[..., Any], *args, compile_kwargs: dict = None):
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}

        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        compiled = torch.compile(func, backend="inductor", **compile_kwargs)

        ref_tensors = flatten_tensors(func(*args))
        result, code = run_and_get_code(compiled, *args)
        actual_tensors = flatten_tensors(result)

        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual))

        return result, code


    @requires_gpu()
    def test_dynamic_shapes_multiple_max_block(self):
        def foo(x):
            view_size = (3, 2)
            full = x.tile(tile_dims)
            view = torch.as_strided(full, view_size, full.stride())
            result = view + view

            return result

        device = torch.device(GPU_TYPE)
        x_size = (1, 1)
        x = torch.randn(x_size).to(device)

        result, (code,) = self.run_and_compare(foo, x)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
