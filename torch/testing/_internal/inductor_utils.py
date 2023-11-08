import contextlib
import os
import sys
import time
from subprocess import CalledProcessError

from third_party.fbgemm.fbgemm_gpu.test.test_utils import TEST_WITH_ROCM
from torch.testing._internal.common_utils import (
    TestCase as TorchTestCase,
)
from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
    IS_MACOS,
    IS_X86,
    IS_WINDOWS,
    IS_CI,
    TEST_WITH_ASAN,
    TEST_CUDA_GRAPH,
)
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase
import torch
import re
import functools
import unittest
import dataclasses
import copy
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import Tuple
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.utils import is_big_gpu


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
WINDOWS_CI = IS_WINDOWS and IS_CI


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


HAS_MULTIGPU = HAS_CUDA and torch.cuda.device_count() >= 2
HAS_AVX2 = "fbgemm" in torch.backends.quantized.supported_engines
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")
requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, "requires multiple cuda devices"
)
skip_if_x86_mac = functools.partial(
    unittest.skipIf, IS_MACOS and IS_X86, "Does not work on x86 Mac"
)
vec_dtypes = [torch.float, torch.bfloat16, torch.float16]


@dataclasses.dataclass
class TestFailure:
    suffixes: Tuple[str]
    is_skip: bool = False
    __test__: bool = False


def copy_tests(
    my_cls, other_cls, suffix, test_failures=None, xfail_prop=None
):  # noqa: B902
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
            # You cannot copy functions in Python, so we use closures here to
            # create objects with different ids. Otherwise, unittest.skip
            # would modify all methods sharing the same object id. Also, by
            # using a default argument, we create a copy instead of a
            # reference. Otherwise, we would lose access to the value.

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            tf = test_failures and test_failures.get(name)
            if tf is not None and suffix in tf.suffixes:
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            setattr(other_cls, f"{name}_{suffix}", new_test)


def clone_preserve_strides(x, device=None):
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out


def compute_grads(args, kwrags, results, grads):
    def gather_leaf_tensors(args, kwargs):
        args = pytree.arg_tree_leaves(*args, **kwargs)
        leaf_tensors = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        return leaf_tensors

    flat_results = pytree.tree_leaves(results)
    flat_diff_results = [r for r in flat_results if r.requires_grad]
    assert len(flat_diff_results) > 0

    leaf_tensors = gather_leaf_tensors(args, kwrags)
    assert len(leaf_tensors) > 0
    return torch.autograd.grad(
        flat_diff_results,
        leaf_tensors,
        grads,
        allow_unused=True,
        retain_graph=True,
    )


def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_cuda=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
):
    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
    ref_kwargs = kwargs
    has_lowp_args = False
    original_lowp_dtype = torch.half

    if reference_in_float:
        # check_lowp is ignored here, it's kept just to be able to call `common` with extra arg
        def upcast_fn(x):
            nonlocal has_lowp_args
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                return x.float()
            else:
                return x

        def get_original_lowp_dtype(example_inputs):
            dtypes = [x.dtype for x in example_inputs if isinstance(x, torch.Tensor)]
            dtype_set = set(dtypes)
            return dtype_set.pop() if len(dtype_set) == 1 else torch.half

        ref_inputs = list(map(upcast_fn, example_inputs))
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        if has_lowp_args:
            original_lowp_dtype = get_original_lowp_dtype(example_inputs)
            if hasattr(model, "to"):
                model = model.to(torch.float)

    torch.manual_seed(0)

    correct = model(*ref_inputs, **ref_kwargs)
    # downcast the model back if needed
    if reference_in_float and has_lowp_args:
        if hasattr(model, "to"):
            model = model.to(original_lowp_dtype)

    torch._inductor.metrics.reset()

    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=nopython)(run)

    torch.manual_seed(0)
    actual = run(*example_inputs, **kwargs)
    # if not called:
    #     exp = torch._dynamo.explain(run)(*example_inputs)
    #     print("Explain:", exp[0])
    #     for graph in exp[2]:
    #         print("Graph", graph)
    if check_has_compiled:
        assert called, "Ran graph without calling compile_fx"
    assert type(actual) == type(correct)

    correct_flat, correct_spec = tree_flatten(correct)
    actual_flat = pytree.tree_leaves(actual)

    def reference_to_expect(actual_flat, correct_flat):
        return tuple(
            y.to(x.dtype)
            if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
            else y
            for x, y in zip(actual_flat, correct_flat)
        )

    if reference_in_float:
        correct_flat = reference_to_expect(actual_flat, correct_flat)
        correct = tree_unflatten(correct_flat, correct_spec)

    if assert_equal:
        self.assertEqual(
            actual,
            correct,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            exact_dtype=exact_dtype,
        )
        # In case of input mutations, check that inputs are the same
        self.assertEqual(
            ref_inputs,
            example_inputs,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            # our testing sometimes uses higher precision inputs for the reference
            exact_dtype=False,
        )
    else:
        for correct_val, actual_val in zip(correct_flat, actual_flat):
            if isinstance(correct_val, torch.Tensor):
                assert correct_val.device == actual_val.device
                assert correct_val.size() == actual_val.size()
                assert correct_val.stride() == actual_val.stride()
                assert correct_val.layout == actual_val.layout
                if exact_dtype:
                    assert correct_val.dtype == actual_val.dtype

    if check_gradient:
        actual = output_process_fn_grad(actual)
        correct = output_process_fn_grad(correct)
        actual_flat = pytree.tree_leaves(actual)
        correct_flat = pytree.tree_leaves(correct)

        # generate random unit norm gradients
        grads = [
            torch.rand(r.shape, device=r.device, dtype=r.dtype)
            for r in correct_flat
            if r.requires_grad
        ]
        for g in grads:
            g /= g.norm()

        correct_grad = compute_grads(ref_inputs, ref_kwargs, correct, grads)
        all_none_grads = all(x is None for x in correct_grad)
        if all_none_grads:
            # See Note [Detaching inputs that never need gradients]
            # There are a handful of ops that can return None gradients, into of zero gradients.
            # If all inputs to an AOTAutograd graph are supposed to get None gradients,
            # AOTAutograd will end up forcing all of the outputs of the forward to not require grad.
            # There's no easy fix to this (see the note above), although one option is to
            # force any derivative formulas in core to return tensors of zeros instead of None.
            flat_results = pytree.tree_leaves(actual)
            results_that_require_grad = [
                x
                for x in flat_results
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            self.assertEqual(len(results_that_require_grad), 0)
        else:
            actual_grad = compute_grads(example_inputs, kwargs, actual, grads)

            if reference_in_float:
                expect_grad = reference_to_expect(actual_grad, correct_grad)
            else:
                expect_grad = correct_grad

            self.assertEqual(
                actual_grad,
                expect_grad,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=exact_dtype,
            )

    torch._dynamo.reset()


@torch._inductor.config.patch("triton.cudagraphs", False)
def check_model_cuda(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_cuda=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
):
    kwargs = kwargs or {}
    if hasattr(model, "to"):
        model = model.to("cuda")

    if copy_to_cuda:
        example_inputs = tuple(
            clone_preserve_strides(x, device="cuda") for x in example_inputs
        )

    check_model(
        self,
        model,
        example_inputs,
        kwargs,
        atol=atol,
        rtol=rtol,
        exact_dtype=exact_dtype,
        nopython=nopython,
        reference_in_float=reference_in_float,
        assert_equal=assert_equal,
        check_gradient=check_gradient,
        check_has_compiled=check_has_compiled,
        output_process_fn_grad=output_process_fn_grad,
    )

    if check_lowp:

        def downcast_fn(x):
            if not isinstance(x, torch.Tensor) or not x.dtype == torch.float:
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device="cuda", dtype=torch.half
            ).copy_(x)

        example_inputs = list(map(downcast_fn, example_inputs))
        if hasattr(model, "to"):
            model = model.to(torch.half)
        if rtol is not None:
            rtol = max(2e-3, rtol)
        check_model(
            self,
            model,
            example_inputs,
            kwargs,
            atol=atol,
            rtol=rtol,
            exact_dtype=exact_dtype,
            nopython=nopython,
            reference_in_float=reference_in_float,
            assert_equal=assert_equal,
            check_gradient=check_gradient,
            check_has_compiled=check_has_compiled,
            output_process_fn_grad=output_process_fn_grad,
        )


def run_and_get_cpp_code(fn, *args, **kwargs):
    # We use the patch context manager instead of using it as a decorator.
    # In this way, we can ensure that the attribute is patched and unpatched correctly
    # even if this run_and_get_cpp_code function is called multiple times.
    with torch._inductor.config.patch(debug=True):
        torch._dynamo.reset()
        import io
        import logging

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        from torch._inductor.graph import output_code_log

        output_code_log.addHandler(ch)
        prev_level = output_code_log.level
        output_code_log.setLevel(logging.DEBUG)
        result = fn(*args, **kwargs)
        s = log_capture_string.getvalue()
        output_code_log.setLevel(prev_level)
        output_code_log.removeHandler(ch)
    return result, s


class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            torch._inductor.config.patch(
                {
                    "debug": True,
                    "debug_index_asserts": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "generate_intermediate_hooks": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        super().setUp()
        self._start = time.perf_counter()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        if os.environ.get("ERROR_ON_SLOW") == "1":
            elapsed = time.perf_counter() - self._start
            assert elapsed < 120


class ToTuple(torch.nn.Module):
    def forward(self, x):
        return (x,)


def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )


def run_inductor_tests(
    *,
    skip_rocm=False,
    skip_asan=False,
    nvcc=False,
    cudagraphs=False,
    mkl=False,
    skip_fbcode=False,
    skip_mac=False,
    triton=False,
    big_gpu=False,
):
    if IS_WINDOWS and IS_CI:
        sys.stderr.write(
            "Windows CI does not have necessary dependencies for inductor yet\n"
        )
        return
    if not (HAS_CPU or HAS_CUDA):
        sys.stderr.write("Missing both CPU compiler and Triton compiler\n")
        return
    if skip_rocm and TEST_WITH_ROCM:
        sys.stderr.write("Skipping due to rocm\n")
        return
    if skip_asan and TEST_WITH_ASAN:
        sys.stderr.write("Skipping due to asan\n")
        return
    if nvcc and not nvcc_exist():
        sys.stderr.write("Skipping due to nvcc\n")
        return
    if cudagraphs and not TEST_CUDA_GRAPH:
        sys.stderr.write("Skipping due to cudagraphs\n")
        return
    if mkl and not torch.backends.mkldnn.is_available():
        sys.stderr.write("Skipping due to mkl\n")
        return
    if skip_fbcode and IS_FBCODE:
        sys.stderr.write("Skipping due to fbcode\n")
        return
    if skip_mac and IS_MACOS:
        sys.stderr.write("Skipping due to mac\n")
        return
    if triton and not HAS_CUDA:
        sys.stderr.write("Skipping due to triton\n")
        return
    if big_gpu and not is_big_gpu(0):
        sys.stderr.write("Skipping due to is_big_gpu\n")
        return

    from torch._dynamo.test_case import run_tests

    return run_tests(("filelock", "sympy"))
