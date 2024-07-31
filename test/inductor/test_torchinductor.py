# Owner(s): ["module: inductor"]
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import itertools
import math
import operator
import os
import random
import re
import subprocess
import sys
import threading
import time
import typing
import unittest
import unittest.mock
import weakref
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.aoti_eager
import torch.nn as nn
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.testing import (
    CompileCounterWithBackend,
    expectedFailureCodegenDynamic,
    rand_strided,
    same,
    skipIfPy312,
)
from torch._dynamo.utils import ifdynstaticdefault
from torch._inductor.aoti_eager import (
    aoti_compile_with_persistent_cache,
    aoti_eager_cache_dir,
    load_aoti_eager_cache,
)
from torch._inductor.codegen.common import DataTypePropagation, OptimizationContext
from torch._inductor.fx_passes import pad_mm
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    add_scheduler_init_hook,
    run_and_get_code,
    run_and_get_cpp_code,
    run_and_get_triton_code,
    run_fw_bw_and_get_code,
)
from torch._inductor.virtualized import V
from torch._prims_common import is_integer_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import _scoped_library
from torch.nn import functional as F
from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    TEST_CUDNN,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    _has_sufficient_memory,
    expectedFailureXPU,
)
from torch.testing._internal.common_dtype import all_types, get_all_dtypes
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_X86,
    parametrize,
    serialTest,
    skipIfNNModuleInlined,
    skipIfRocm,
    skipIfWindows,
    skipIfXpu,
    subtest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.weak import WeakTensorKeyDictionary


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

importlib.import_module("functorch")
importlib.import_module("filelock")

from torch._inductor import config, test_operators
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.utils import has_torchvision_roi_align
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_MULTIGPU,
    requires_gpu,
    skipCPUIf,
    skipCUDAIf,
)


HAS_AVX2 = "fbgemm" in torch.backends.quantized.supported_engines

aten = torch.ops.aten

requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)
skip_if_x86_mac = functools.partial(
    unittest.skipIf, IS_MACOS and IS_X86, "Does not work on x86 Mac"
)
vec_dtypes = [torch.float, torch.bfloat16, torch.float16]

libtest = torch.library.Library("test", "FRAGMENT")  # noqa: TOR901
ids = set()

f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


def _large_cumprod_input(shape, dim, dtype, device):
    # Construct a cumprod input which guaruntees not to overflow or underflow
    if is_integer_dtype(dtype):
        # Large products don't fit in integers, the best we can do
        # is random +/-1 values to test the sign of the result
        x = torch.randint(0, 1, shape, dtype=dtype, device=device)
        return x * 2 - 1

    comp_dtype = torch._prims_common.get_computation_dtype(dtype)
    batch_size = 256
    if comp_dtype != dtype:
        batch_size = math.floor(math.log2(torch.finfo(dtype).max) / 3)

    # Create random values with a uniform magnitude and uniform exponent
    num_batches = (shape[dim] + 2 * batch_size - 1) // (2 * batch_size)
    batch_shape = (
        shape[:dim]
        + (
            num_batches,
            batch_size,
        )
        + shape[dim + 1 :]
    )
    magnitude = 1 + torch.rand(batch_shape, dtype=comp_dtype, device=device)
    exponent = torch.randint(-1, 1, batch_shape, device=device).to(comp_dtype)
    batch = magnitude * exponent.exp2()

    # Alternate each batch of values with their reciprocals so the product
    # never gets too far away from 1
    t = torch.cat((batch, batch.reciprocal()), dim=dim + 1)
    t = t.flatten(dim, dim + 1)
    t = aten.slice(t, dim=dim, start=0, end=shape[dim])

    # Randomize sign
    sign = torch.randint(0, 1, shape, device=device) * 2 - 1
    return (t * sign).to(dtype)


def define_custom_op_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor self) -> Tensor", tags=tags)
        libtest.impl(id_, fn_cpu, "CPU")
        libtest.impl(id_, fn_cuda, "CUDA")
        libtest.impl(id_, fn_xpu, "XPU")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def define_custom_op_2_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(
            f"{id_}(Tensor self, float scale) -> (Tensor, Tensor)", tags=tags
        )
        libtest.impl(id_, fn_cpu, "CPU")
        libtest.impl(id_, fn_cuda, "CUDA")
        libtest.impl(id_, fn_xpu, "XPU")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def define_custom_op_3_for_test(id_, fn_cpu, fn_cuda, fn_xpu, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor[] x) -> Tensor", tags=tags)
        libtest.impl(id_, fn_cpu, "CPU")
        libtest.impl(id_, fn_cuda, "CUDA")
        libtest.impl(id_, fn_xpu, "XPU")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


f32 = torch.float32


def register_ops_with_aoti_compile(ns, op_set, dispatch_key, torch_compile_op_lib_impl):
    for _op_name in op_set:
        qualified_op_name = f"{ns}::{_op_name}"
        _, overload_names = torch._C._jit_get_operation(qualified_op_name)
        for overload_name in overload_names:
            try:
                reg_op_name = qualified_op_name
                schema = torch._C._get_schema(qualified_op_name, overload_name)
                if schema.overload_name:
                    reg_op_name = f"{qualified_op_name}.{schema.overload_name}"
                torch_compile_op_lib_impl._impl_with_aoti_compile(  # noqa: F821
                    reg_op_name, dispatch_key
                )
            except Exception as e:
                continue


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
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


@dataclasses.dataclass
class InputGen:
    n: int
    device: str

    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    def transposed(self):
        return self.dense().transpose(0, 1)

    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)

    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def compute_grads(args, kwrags, results, grads):
    def gather_leaf_tensors(args, kwargs):
        args = pytree.arg_tree_leaves(*args, **kwargs)
        leaf_tensors = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        return leaf_tensors

    flat_results = pytree.tree_leaves(results)
    flat_diff_results = [
        r for r in flat_results if isinstance(r, torch.Tensor) and r.requires_grad
    ]
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


def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
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

    if reference_in_float and exact_dtype:
        # Store expected dtypes so we can check actual result gives the correct types
        torch.manual_seed(0)
        try:
            eager_result = model(*ref_inputs, **ref_kwargs)
        except RuntimeError:
            # Eager model may fail if the dtype is not supported
            eager_result = None

        ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
        expect_dtypes = [
            x.dtype if isinstance(x, torch.Tensor) else None
            for x in pytree.tree_leaves(eager_result)
        ]
        del eager_result

    ref_model = model
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

        ref_inputs = list(map(upcast_fn, example_inputs))
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        if has_lowp_args and hasattr(model, "to"):
            ref_model = copy.deepcopy(model).to(torch.float)

    torch.manual_seed(0)

    correct = ref_model(*ref_inputs, **ref_kwargs)

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
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(correct)
        assert all(
            type(actual_item) == type(correct_item)
            for actual_item, correct_item in zip(actual, correct)
        )

    correct_flat, correct_spec = tree_flatten(correct)
    actual_flat = pytree.tree_leaves(actual)

    def reference_to_expect(actual_flat, correct_flat):
        return tuple(
            (
                y.to(x.dtype)
                if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
                else y
            )
            for x, y in zip(actual_flat, correct_flat)
        )

    if reference_in_float and exact_dtype:
        for expect_dtype, actual_result in zip(expect_dtypes, actual_flat):
            if expect_dtype is not None:
                assert (
                    actual_result.dtype == expect_dtype
                ), f"dtype mismatch, expected {expect_dtype} but got {actual_result.dtype}"

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
                strides_equal, _ = torch._prims_common.check_significant_strides(
                    correct_val, actual_val
                )
                assert strides_equal
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
            if isinstance(r, torch.Tensor) and r.requires_grad
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
                atol=grad_atol or atol,
                rtol=grad_rtol or rtol,
                equal_nan=True,
                exact_dtype=exact_dtype,
            )

    torch._dynamo.reset()


@torch._inductor.config.patch("triton.cudagraphs", False)
def check_model_gpu(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
):
    kwargs = kwargs or {}
    if hasattr(model, "to"):
        model = model.to(device=GPU_TYPE)

    if copy_to_gpu:
        example_inputs = tuple(
            clone_preserve_strides(x, device=GPU_TYPE) for x in example_inputs
        )

    check_model(
        self,
        model,
        example_inputs,
        kwargs,
        atol=atol,
        rtol=rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
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
                x.size(), x.stride(), device=GPU_TYPE, dtype=torch.half
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
            grad_atol=grad_atol,
            grad_rtol=grad_rtol,
            exact_dtype=exact_dtype,
            nopython=nopython,
            reference_in_float=reference_in_float,
            assert_equal=assert_equal,
            check_gradient=check_gradient,
            check_has_compiled=check_has_compiled,
            output_process_fn_grad=output_process_fn_grad,
        )


check_model_cuda = check_model_gpu


def _run_and_assert_no_indirect_indexing(
    test_case, func, *args, has_wrapping=None, has_assert=False, **kwargs
):
    result, source_codes = run_and_get_code(func, *args, **kwargs)

    for code in source_codes:
        for line in code.split("\n"):
            stmt = None
            # Find indexing expressions
            if ".load(" in line:
                stmt = line.split(".load")[-1]
            elif "tl.store" in line:
                stmt = line.split(".store")[-1]
                stmt = ",".join(stmt.split(",")[:-2])  # Remove store value and mask
            elif ".store" in line:
                stmt = line.split(".store")[-1]
            elif "[" in line:
                stmt = line.split("[")[-1].split("]")[0]
            if "tl.make_block_ptr(" in line:
                continue

            if stmt is None:
                continue

            # indirect indexing involves a `tmp` variable
            test_case.assertTrue(
                "tmp" not in stmt,
                msg=f"Found indirect indexing in statement '{stmt}' from code:\n{code}",
            )
        if has_wrapping is not None:
            test_case.assertTrue(
                ("where" in code or "?" in code) is has_wrapping,
                msg=f"Wanted {has_wrapping=} but got\n{code}",
            )
    test_case.assertTrue(
        any(
            ("device_assert" in code or "TORCH_CHECK" in code) is has_assert
            for code in source_codes
        )
    )
    return result


def assertGeneratedKernelCountEqual(self: TestCase, expected: int):
    if config.triton.multi_kernel:
        # when multi_kernel is enabled, we generated both persistent reduction
        # and non-persistent reduction kernels for the same node schedule.
        # That will mess up with the kernel count. Just don't check it.
        return
    if config.cpp_wrapper:
        expected *= 2
    self.assertEqual(torch._inductor.metrics.generated_kernel_count, expected)


class SweepInputs2:
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    input_gen_types2 = input_gen_types1
    gen = None

    @staticmethod
    def kernel(a, b):
        return (a + b,)

    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),
                    getattr(cls.gen, name2)(),
                ),
            )

        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


def is_cpp_backend(device):
    return getattr(device, "type", device) == "cpu" and config.cpu_backend == "cpp"


def is_halide_backend(device):
    if getattr(device, "type", device) == "cpu":
        return config.cpu_backend == "halide"
    return config.cuda_backend == "halide"


def skip_if_halide(fn):
    @functools.wraps(fn)
    def wrapper(self):
        if is_halide_backend(self.device):
            raise unittest.SkipTest("halide not supported")
        return fn(self)

    return wrapper


def skip_if_gpu_halide(fn):
    @functools.wraps(fn)
    def wrapper(self):
        if (
            is_halide_backend(self.device)
            and getattr(self.device, "type", self.device) == "cuda"
        ):
            raise unittest.SkipTest("halide not supported")
        return fn(self)

    return wrapper


@instantiate_parametrized_tests
class CommonTemplate:
    def test_bool(self):
        def fn(a, b):
            return (
                a + b,
                a * b,
                a & b,
                a | b,
                a ^ b,
                torch.logical_and(a, b),
                torch.logical_or(a, b),
                torch.logical_not(a),
                torch.sign(b),
            )

        self.common(
            fn,
            (
                torch.tensor([True, False, True, False]),
                torch.tensor([False, False, True, True]),
            ),
        )

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_dtype_device_layout(self):
        ns = "aten"
        op_name = "tril_indices"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            row = 128
            col = 256
            offset = 1
            dtype = torch.int32
            layout = torch.strided
            pin_memory = False
            ref = torch.tril_indices(
                row=row,
                col=col,
                offset=offset,
                dtype=dtype,
                layout=layout,
                pin_memory=pin_memory,
                device=device,
            )
            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )
            res = torch.tril_indices(
                row=row,
                col=col,
                offset=offset,
                dtype=dtype,
                layout=layout,
                pin_memory=pin_memory,
                device=device,
            )
            self.assertEqual(ref, res)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_support_out(self):
        ns = "aten"
        op_name = "clamp"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        inp_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(1.0)
        min_tensor = inp_tensor - 0.05
        max_tensor = inp_tensor + 0.05
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            ref_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            ref_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=ref_out_tensor
            )

            ref_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            ref_tensor1 = torch.clamp(
                max=max_tensor, out=ref_out_tensor1, min=min_tensor, input=inp_tensor
            )

            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            res_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            res_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=res_out_tensor
            )

            self.assertEqual(ref_tensor, res_tensor)
            self.assertEqual(ref_out_tensor, res_out_tensor)

            res_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            res_tensor1 = torch.clamp(
                max=max_tensor, out=res_out_tensor1, min=min_tensor, input=inp_tensor
            )

            self.assertEqual(ref_tensor1, res_tensor1)
            self.assertEqual(ref_out_tensor1, res_out_tensor1)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_support_str(self):
        ns = "aten"
        op_name = "div"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        a = torch.randn(128, dtype=torch.float, device=device)
        b = torch.randn(128, dtype=torch.float, device=device)
        rounding_mode_list = ["trunc", "floor"]
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # Get ref result from eager
            ref_value_list = []
            for rounding_mode in rounding_mode_list:
                ref_value = getattr(torch.ops.aten, op_name)(
                    a, b, rounding_mode=rounding_mode
                )
                ref_value_list.append(ref_value)

            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            # Invoke the pre-compiled kernel and get result.
            res_value_list = []
            for rounding_mode in rounding_mode_list:
                res_value = getattr(torch.ops.aten, op_name)(
                    a, b, rounding_mode=rounding_mode
                )
                res_value_list.append(res_value)

            for ref_value, res_value in zip(ref_value_list, res_value_list):
                self.assertEqual(ref_value, res_value)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_cache_hit(self):
        ns = "aten"
        op_name = "abs"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            device,
            False,
            getattr(torch.ops.aten, op_name),
            (input_tensor,),
            {},
        )
        self.assertTrue(Path(kernel_lib_path).exists())

        from unittest import mock

        # Patch the aoti_compile_with_persistent_cache as None to ensure no new kernel is generated
        with mock.patch(
            "torch._inductor.aoti_eager.aoti_compile_with_persistent_cache", None
        ):
            with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
                # Get ref result from eager
                ref_value = getattr(torch.ops.aten, op_name)(input_tensor)

                register_ops_with_aoti_compile(
                    ns, [op_name], dispatch_key, torch_compile_op_lib_impl
                )

                # Invoke the pre-compiled kernel and get result.
                res_value = getattr(torch.ops.aten, op_name)(input_tensor)

                self.assertEqual(ref_value, res_value)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_with_persistent_cache(self):
        def fn(a):
            return torch.abs(a)

        ns = "aten"
        op_name = "abs"

        device = "cpu"
        if self.device.lower() == "cuda":
            device = "cuda"

        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            input_tensor.device.type,
            False,
            fn,
            args=(input_tensor,),
            kwargs={},
        )
        self.assertTrue(len(kernel_lib_path) > 0)

        device_kernel_cache = aoti_eager_cache_dir(ns, device)
        kernel_conf = device_kernel_cache / f"{op_name}.json"
        self.assertTrue(kernel_conf.exists())

        json_data = load_aoti_eager_cache("aten", "abs", input_tensor.device.type)
        self.assertTrue(json_data is not None)
        self.assertTrue(isinstance(json_data, list))
        self.assertTrue(len(json_data) > 0)

        op_info = json_data[0]
        self.assertTrue(isinstance(op_info, dict))
        self.assertTrue("meta_info" in op_info)
        self.assertTrue("kernel_path" in op_info)
        kernel_libs_abs_path = []
        for item in json_data:
            kernel_path = device_kernel_cache / item["kernel_path"]
            kernel_libs_abs_path.append(kernel_path.as_posix())

        self.assertTrue(kernel_lib_path in kernel_libs_abs_path)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_with_scalar(self):
        namespace_name = "aten"
        op_name = "add"
        op_overload_name = "Tensor"
        op_name_with_overload = f"{op_name}.{op_overload_name}"

        dispatch_key = "CPU"
        device = torch.device("cpu")
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        # Test the difference between scalar tensor and scalar
        a = torch.scalar_tensor(1.0, device=device)
        b = torch.scalar_tensor(2.0, device=device)

        kernel_lib_path = aoti_compile_with_persistent_cache(
            namespace_name,
            op_name_with_overload,
            a.device.type,
            False,
            torch.ops.aten.add,
            args=(a, b),
            kwargs={"alpha": 3.0},
        )
        self.assertTrue(Path(kernel_lib_path).exists())
        device_kernel_cache = aoti_eager_cache_dir(namespace_name, device.type)
        kernel_conf = device_kernel_cache / f"{op_name_with_overload}.json"
        self.assertTrue(kernel_conf.exists())
        json_data = load_aoti_eager_cache(
            namespace_name, op_name_with_overload, a.device.type
        )
        op_info = json_data[0]
        self.assertTrue(isinstance(op_info, dict))
        self.assertTrue("meta_info" in op_info)
        self.assertTrue(len(op_info["meta_info"]) == 3)
        # Scalar Tensor
        self.assertTrue("scalar_value" not in op_info["meta_info"][0])
        self.assertTrue(op_info["meta_info"][0]["sizes"] == [])
        self.assertTrue(op_info["meta_info"][0]["strides"] == [])
        # Scalar Tensor
        self.assertTrue("scalar_value" not in op_info["meta_info"][1])
        self.assertTrue(op_info["meta_info"][1]["sizes"] == [])
        self.assertTrue(op_info["meta_info"][1]["strides"] == [])
        # Scalar
        self.assertTrue("scalar_value" in op_info["meta_info"][2])
        self.assertTrue("sizes" not in op_info["meta_info"][2])
        self.assertTrue("strides" not in op_info["meta_info"][2])

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            a = torch.randn(128, device=device)
            b = torch.randn(128, device=device)

            scalar_values = [1.0, 2.0, 3.0]
            ref_values = []
            for scalar_value in scalar_values:
                ref_values.append(torch.add(a, b, alpha=scalar_value))

            register_ops_with_aoti_compile(
                namespace_name, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            res_values = []
            for scalar_value in scalar_values:
                res_values.append(torch.add(a, b, alpha=scalar_value))

            self.assertEqual(len(ref_values), len(res_values))
            self.assertEqual(ref_values, res_values)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skipIfWindows
    def test_aoti_eager_override_registration(self):
        namespace_name = "aten"
        dispatch_key = "CPU"
        device = torch.device("cpu")
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        unary_op_set = ["abs", "acos"]

        def fn(x, op_name=""):
            return getattr(torch, op_name)(x)

        # Invoke torch.compile directly to get referent results
        x = torch.randn(3, 4, device=device)

        ref_array = []
        for unary_op_name in unary_op_set:
            opt_fn = torch.compile(functools.partial(fn, op_name=unary_op_name))
            ref = opt_fn(x)
            ref_array.append(ref)

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            register_ops_with_aoti_compile(
                namespace_name, unary_op_set, dispatch_key, torch_compile_op_lib_impl
            )

            res_array = []
            for unary_op_name in unary_op_set:
                res_array.append(getattr(torch, unary_op_name)(x))

            for ref, res in zip(ref_array, res_array):
                self.assertEqual(ref, res)

        a = torch.randn(128, device=device)
        min_tensor = torch.randn(128, device=device)
        max_tensor = min_tensor + 0.5

        ref_with_min = torch.ops.aten.clamp(a, min_tensor)
        ref_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            register_ops_with_aoti_compile(
                namespace_name, ["clamp"], dispatch_key, torch_compile_op_lib_impl
            )
            res_with_min = torch.ops.aten.clamp(a, min_tensor)
            res_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)
            self.assertEqual(ref_with_min, res_with_min)
            self.assertEqual(ref_with_min_max, res_with_min_max)

    def test_add_const_int(self):
        def fn(a):
            return (a + 1, torch.add(a, 1, alpha=2))

        for dtype in [torch.float32, torch.int32, torch.int64]:
            self.common(fn, (torch.arange(32, dtype=dtype),))

    def test_add_const_float(self):
        def fn(a):
            return (a + 1.5,)

        self.common(fn, (torch.randn(32),))

    def test_add_inplace_permuted(self):
        def fn(x, y):
            return x.add_(y)

        x = torch.ones([2, 12, 13, 17]).transpose(1, 2)
        y = torch.randn([2, 13, 1, 17])

        self.common(fn, (x, y))

    def test_add_complex(self):
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])
        y = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])

        self.common(fn, (x, y, 2))

    def test_add_complex3(self):
        # fix https://github.com/pytorch/pytorch/issues/115071
        @torch.compile
        def fn(*args):
            a = torch.neg(args[0])
            b = torch.add(args[0], args[0])
            return (a, b)

        x = torch.randn(41, dtype=torch.complex64)
        y = x.clone()
        # should not inplace write to the input
        fn(x)
        self.assertEqual(x, y)

    def test_add_complex4(self):
        @torch.compile
        def fn(a, b):
            c = a + b
            d = a + b
            return c + d

        for dtype in [torch.complex32, torch.complex64, torch.complex128]:
            x = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            y = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            _, code = run_and_get_code(fn, x, y)
            self.assertEqual(
                " ".join(code).count(
                    "view_dtype" if config.cpp_wrapper else "aten.view"
                ),
                3,
            )

    def test_add_complex5(self):
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])

        self.common(fn, (x, y, 2))

    def test_add_complex6(self):
        # Fix https://github.com/pytorch/pytorch/issues/125745.
        # Add complex tensors with broadcasting.
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([[1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j]])

        self.common(fn, (x, y, 2))

    def test_concat_add_inplace(self):
        def fn(x, y, z):
            return torch.cat([x, y], dim=1).add_(z)

        x = torch.randn([2, 12, 14, 14])
        y = torch.randn([2, 12, 14, 14])
        z = torch.randn([2, 24, 14, 14])

        self.common(fn, (x, y, z))

    def test_abs(self):
        def fn(a):
            return (a / (torch.abs(a) + 1),)

        self.common(fn, (torch.randn(17),))

    def test_angle(self):
        def fn(a, b, c):
            return torch.angle(a), torch.angle(b), torch.angle(c)

        complex_input = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1, float("nan")]
        )
        real_input = torch.tensor([-1.0, 0.0, 1.0, float("nan")])
        interger_real_input = torch.tensor([-1, 0, 1])
        self.common(fn, (complex_input, real_input, interger_real_input))

    def test_sgn(self):
        def fn(a):
            return torch.sgn(a), torch.sgn(a + 1) - 1

        self.common(fn, [torch.linspace(-10, 10, 41)])

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    def test_scatter_bf16(self):
        def fn(inp, src, index):
            return inp.scatter_add(0, index, src)

        for dtype in [torch.int64, torch.bool, torch.bfloat16]:
            self.common(
                fn,
                [
                    torch.zeros(3, 5, dtype=dtype),
                    torch.ones((2, 5), dtype=dtype),
                    torch.tensor([[0, 1, 2, 0, 0]]),
                ],
            )

    def test_randn_generator(self):
        def fn(a, generator):
            return torch.randn([20, 20], generator=generator, device=a.device)

        self.common(fn, (torch.linspace(-10, 10, 41), None), assert_equal=False)

        # generator not yet supported in dynamo
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "Generator"):
            self.common(fn, (torch.linspace(-10, 10, 41), torch.Generator(self.device)))

    def test_sgn_extremal(self):
        def fn(a):
            return (torch.sgn(a),)

        self.common(fn, [torch.tensor([np.nan, np.inf, -np.inf, 0])])

    def test_max_min(self):
        def fn(a, b):
            return (torch.maximum(a, b), torch.minimum(a, b))

        self.common(fn, (torch.randn(8), torch.randn(8)))
        t1 = torch.randn(8)
        t1[0] = float("nan")
        t2 = torch.randn(8)
        t2[1] = float("nan")
        self.common(fn, (t1, t2))

    def test_neg_max_uint8(self):
        # https://github.com/pytorch/pytorch/issues/93380
        def fn(a, b):
            c = torch.neg(a)
            return torch.maximum(b, c)

        a = torch.randint(256, (1,), dtype=torch.uint8)
        b = torch.randint(256, (8390,), dtype=torch.uint8)
        self.common(fn, (a, b))

    def test_compar(self):
        def fn(x):
            return x.gt(3.5), x.ge(3.5), x.eq(3.5), x.le(2.5), x.lt(3.5), x.ne(3.5)

        a = torch.tensor([3])
        self.common(fn, (a,))

    def test_horizonal_fusion1(self):
        def fn(a, b, c):
            return (a + b, a - c, b * c)

        self.common(
            fn, (torch.randn(8, 16, 16), torch.randn(8, 16, 16), torch.randn(1, 16, 1))
        )

    def test_horizonal_fusion2(self):
        def fn(a, b, c):
            return a + 1, b + 2, c + 3

        self.common(fn, (torch.randn(8, 16, 8), torch.randn(8, 16), torch.randn(16, 8)))

    def test_vertical_fusion1(self):
        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = v19 * ct
            t19 = v17 + ct * (v18 + t15) + v20 * sa
            t20 = 1.0 / t19
            t128 = t19 * p
            return t20 + t128

        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        assertGeneratedKernelCountEqual(self, 1)

    @config.patch({"fx_graph_cache": False})
    @skipIfWindows
    def test_forced_buffer_realize(self):
        # Test torch._test_inductor_realize forces a buffer to be realized
        def fn(a):
            b = test_operators.realize(a * 2)
            return (b * 2,)

        self.common(fn, (torch.randn(10),))
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 2)

    @config.patch({"fx_graph_cache": False})
    @skipIfWindows
    def test_scheduler_vertical_fusion1(self):
        realize = test_operators.realize

        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = realize(v19 * ct)
            t19 = realize(v17 + ct * (v18 + t15) + v20 * sa)
            t20 = realize(1.0 / t19)
            t128 = realize(t19 * p)
            return t20 + t128

        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 5)
        assertGeneratedKernelCountEqual(
            self, 1 if not is_cpp_backend(self.device) else 2
        )

    @skipIfWindows
    def test_index_propagation(self):
        def copy(x):
            i = torch.arange(x.size(0), device=x.device)
            return x[i]

        x = torch.randn(8, device=self.device)
        copy_opt = torch._dynamo.optimize("inductor")(copy)

        expect = copy(x)
        actual = _run_and_assert_no_indirect_indexing(self, copy_opt, x)
        self.assertEqual(expect, actual)

    @dynamo_config.patch("capture_dynamic_output_shape_ops", True)
    # https://github.com/halide/Halide/issues/8308
    @config.patch("halide.scheduler_cpu", "Mullapudi2016")
    @config.patch("halide.scheduler_cuda", "Li2018")
    @config.patch(implicit_fallbacks=True)
    @skipIfWindows
    def test_index_propagation_nested_indirect_indexing(self):
        def nested(x, repeats):
            rank = torch.arange(repeats.numel(), device=x.device)
            index = rank.repeat_interleave(repeats, dim=0)
            return torch.index_select(x, index=index, dim=0)

        example_inputs = (
            torch.randn((32, 64), device=self.device),
            repeats := torch.tensor([5, 10, 15], device=self.device),
        )
        torch._dynamo.mark_dynamic(repeats, 0)  # create backed symint

        nested_opt = torch._dynamo.optimize("inductor")(nested)

        expect = nested(*example_inputs)
        actual = nested_opt(*example_inputs)
        self.assertEqual(expect, actual)

    @skipIfWindows
    def test_index_propagation_flip(self):
        def flip(x):
            i = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[i]

        x = torch.randn(8, device=self.device)
        flip_opt = torch._dynamo.optimize("inductor")(flip)

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)

    @skipIfWindows
    def test_index_propagation_floordiv(self):
        def repeat_interleave(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 1, 2, 2, 3, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i // n]

        x = torch.randn(8, 16, device=self.device)
        repeat_interleave_opt = torch._dynamo.optimize("inductor")(repeat_interleave)
        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_interleave_opt, x, 3, has_assert=has_assert
        )
        expect = torch.repeat_interleave(x, 3, dim=0)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat_interleave(x, 3))

    @skipIfWindows
    def test_index_propagation_remainder(self):
        def repeat(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 2, 3, 1, 2, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i % x.shape[0]]

        x = torch.randn(8, 16, device=self.device)
        repeat_opt = torch._dynamo.optimize("inductor")(repeat)

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_opt, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = x.repeat(3, 1)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat(x, 3))

    @skipIfWindows
    def test_index_propagation_abs(self):
        def reflection_pad_left(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [3, 2, 1, 2, 3]
            i = torch.arange(x.shape[0] + n, device=x.device)
            return x[(i - n).abs()]

        x = torch.randn(8, device=self.device)
        opt_fn = torch._dynamo.optimize("inductor")(reflection_pad_left)

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, opt_fn, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = reflection_pad_left(x, 3)
        self.assertEqual(expect, actual)

    @skipIfWindows
    def test_index_propagation_device_assert_masked(self):
        def fn(a):
            idx = torch.arange(a.size(0), device=a.device)
            padded_idx = torch.constant_pad_nd(idx, (1050, 0))
            padded_idx = torch.where(padded_idx >= 0, padded_idx, padded_idx)
            return a[padded_idx]

        self.common(fn, (torch.randn(1024),))

    @config.patch(debug_index_asserts=False)
    @skipIfWindows
    def test_neg_index(self):
        def test(
            fn, inps, has_assert: bool, has_wrapping: bool, vectorize: bool = True
        ):
            fn_opt = torch.compile(fn)
            if is_halide_backend(self.device):
                pass  # no device asserts in halide
            elif self.device == "cpu":
                _, code = run_and_get_cpp_code(fn_opt, *inps)
                self.assertTrue(("?" in code or "blendv" in code) is has_wrapping)
                self.assertTrue(("TORCH_CHECK" in code) is has_assert)
                # Assert that we always vectorize the kernel regardless of wrapping / checks
                self.assertTrue(("loadu" in code) is vectorize)
            else:
                code = run_and_get_triton_code(fn_opt, *inps)
                self.assertTrue(("tl.where" in code) is has_wrapping)
                self.assertTrue(("device_assert" in code) is has_assert)

        def indirect(a, b):
            return a[b - 1]

        a = torch.rand(1024, device=self.device)
        b = torch.zeros(256, dtype=torch.long, device=self.device)
        test(indirect, (a, b), has_assert=True, has_wrapping=True)

        def direct(x):
            return x[:, -1]

        a = torch.rand(1, 64, 32, device=self.device)
        # Does not even generate a kernel as it's a view
        test(direct, (a,), has_assert=False, has_wrapping=False, vectorize=False)

        def flip(a, b):
            return a[b]

        a = torch.rand(1024, device=self.device)
        b = torch.arange(start=-1, end=-a.numel() - 1, step=-1, device=self.device)
        test(flip, (a, b), has_assert=True, has_wrapping=True)

        # Constant propagate a constant that's negative
        def flip_with_index_constant(a):
            b = torch.arange(start=-1, end=-a.numel() - 1, step=-1, device=a.device)
            return a[b]

        # Wrapping is constant-folded
        test(flip_with_index_constant, (a,), has_assert=False, has_wrapping=False)

        # Operation where we can't prove that the index is always positive or negative
        def pos_and_neg(a):
            b = torch.arange(start=1, end=-a.numel() - 1, step=-1, device=a.device)
            return a[b]

        # It has wrapping but no assert
        test(pos_and_neg, (a,), has_assert=False, has_wrapping=True)

        # We currently don't do constant propagation with float constants
        # We cannot prove this kind of asserts just with bounds. We would need
        # to lift IndexPropagation.shape_env to be accessible in all of Inductor
        def flip_with_index(a):
            b = 1.0 * torch.arange(
                start=-1, end=-a.numel() - 1, step=-1, device=a.device
            )
            b = b.int()
            return a[b]

        test(
            flip_with_index,
            (a,),
            has_assert=ifdynstaticdefault(False, True),
            has_wrapping=False,
            vectorize=False,  # Constant propagation off -> indirect indexing -> no vec
        )

        def unsafe_index(a, b):
            return aten._unsafe_index(a, (b,))

        test(unsafe_index, (a, b), has_assert=False, has_wrapping=True)

        def constant_propagation(a):
            b = torch.tensor([2], device=a.device)
            return a[b]

        test(
            constant_propagation,
            (a,),
            has_assert=ifdynstaticdefault(False, True),
            has_wrapping=False,
            vectorize=False,  # There's no loop to vectorize!
        )

        def constant_propagation_neg(a):
            b = torch.tensor([-2], device=a.device)
            return a[b]

        # In symbolic shapes, we know that we can access -2, so no assert is necessary!
        test(
            constant_propagation_neg,
            (a,),
            has_assert=False,
            has_wrapping=False,
            vectorize=False,  # There's no loop to vectorize!
        )

    @skipIfWindows
    def test_computed_buffer_inlining(self):
        def flip(x):
            idx = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[idx], idx

        flip_opt = torch._dynamo.optimize("inductor")(flip)
        x = torch.randn(8, device=self.device)

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)

    def test__unsafe_masked_index(self):
        def fn(a, mask, idx):
            return aten._unsafe_masked_index(a, mask, idx, 1)

        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, -2], device=self.device)],
            ),
        )

    def test__unsafe_masked_index_put_accumulate(self):
        def fn(a, mask, idx, values):
            return aten._unsafe_masked_index_put_accumulate(a, mask, idx, values)

        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, -2], device=self.device)],
                torch.randn(3, device=self.device),
            ),
        )

    def test_sum1(self):
        def fn(a, b):
            return ((a + b).sum(-1),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sum2(self):
        def fn(a, b):
            return ((a + b).sum([1, 2]), (a + b).sum(-1))

        self.common(fn, (torch.randn(8, 9, 3, 21), torch.randn(8, 9, 3, 21)))

    def test_sum3(self):
        def fn(a, b):
            r1 = a + b
            r2 = r1.sum(-1)
            r3 = torch.squeeze(b) + 10
            return (r1, r2, r3)

        # Mismatched elements: 2 / 10 (20.0%)
        # Greatest absolute difference: 0.0029296875 at index (8,) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0017482517482517483 at index (6,) (up to 0.001 allowed)
        self.common(fn, (torch.randn(10, 10), torch.randn(1, 10)), atol=1e-5, rtol=2e-3)

    def test_sum4(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f, e, d, c, b)

        self.common(fn, (torch.randn(1, 16, 8, 8),))

    def test_sum5(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f,)

        self.common(fn, (torch.randn(1, 17, 8, 9),))

    def test_reduction1(self):
        def fn(a):
            return (a.sum(), a.max(), a.min(), a.argmax(), a.argmin())

        self.common(fn, (torch.tensor([float("-inf"), 0.0, float("inf")]),))

    @skip_if_x86_mac()
    def test_reduction2(self):
        def fn(a):
            # FIXME: a.argmax
            return (a.sum(), a.max(), a.min(), a.argmin())

        self.common(fn, (torch.full((4,), float("inf")),))

    @skip_if_x86_mac()
    def test_reduction3(self):
        def fn(a):
            # FIXME: a.argmin
            return (a.sum(), a.max(), a.min(), a.argmax())

        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_reduction4(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        def fn(a):
            return (a.argmax(-1), a.argmin(-1))

        inputs = (torch.ones(128), torch.ones(4, 4, 1))
        for i in inputs:
            self.common(fn, (i,), check_lowp=not is_halide_backend(self.device))

    @config.patch(unroll_reductions_threshold=1)
    def test_reduction5(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        def fn(a):
            return (a.sum(), a.max(), a.min(), a.argmax())

        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_prod(self):
        def fn(a):
            return a.prod(0), a.prod(1), a.prod()

        self.common(fn, (torch.rand((10, 10)),))
        self.common(fn, (torch.rand((1, 2050)),))

    def test_unroll_small_reduction(self):
        def fn(x):
            val1, index1 = x.min(-1)
            val2, index2 = x.max(-1)
            return (
                val1,
                index1,
                val2,
                index2,
                x.sum(-1),
                (x > 1).any(-1),
                (x > 0).all(-1),
                x.argmin(-1),
                x.argmax(-1),
                x.amin(-1),
                x.amax(-1),
                x.aminmax(),
            )

        with config.patch(unroll_reductions_threshold=8):
            # small sized reductions will get unrolled
            self.common(fn, (torch.randn(8, 3),))
        torch._dynamo.reset()
        with config.patch(unroll_reductions_threshold=1):
            # make sure things also work if they aren't unrolled
            self.common(fn, (torch.randn(8, 3),))

    def test_multilayer_sum_low_prec(self):
        # fp16 nyi for cpu
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(a):
            return torch.mean(a)

        self.common(fn, ((torch.rand((10, 3, 352, 352), dtype=torch.float16),)))

    def test_multilayer_prime_size(self):
        def fn(a):
            return torch.max(a), torch.sum(a)

        # Requires masked loading for the intermediate reduction
        sample = torch.full((3999971,), 0, dtype=torch.int64)
        sample[-1] = 1
        self.common(fn, (sample,))

    @skip_if_gpu_halide
    @skipCPUIf(IS_MACOS, "fails on macos")
    def test_multilayer_var(self):
        def fn(a):
            return torch.var(a)

        self.common(
            fn,
            ((torch.rand((10, 3, 352, 352), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )
        self.common(
            fn,
            ((torch.rand((14923), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )

    @skipCPUIf(IS_MACOS, "fails on macos")
    @skip_if_halide  # accuracy 4.7% off
    @skipIfWindows
    def test_multilayer_var_lowp(self):
        def fn(a):
            return torch.var(a)

        self.common(fn, (torch.rand((16, 16, 352, 352), dtype=torch.float16),))
        self.common(fn, (torch.rand((14923), dtype=torch.float16),))

    def test_split_cumsum(self):
        def fn(a):
            return torch.cumsum(a, -1)

        for dtype in get_all_dtypes(
            include_bfloat16=False,
            include_bool=True,
            include_complex=False,
            include_half=False,
        ):
            # Use low=0 since when the mean value is 0, cumsum at all points
            # tends towards zero which makes the relative error term blow up
            inp = make_tensor(10, 3, 352, 352, low=0, dtype=dtype, device=self.device)
            self.common(fn, (inp.view(-1),), rtol=1e-4, atol=1e-5, check_lowp=False)
            self.common(fn, (inp.view(10, -1),), rtol=1e-4, atol=1e-5, check_lowp=False)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skipCUDAIf(TEST_WITH_ROCM, "Computation not done in float on ROCm")
    @skip_if_gpu_halide  # accuracy issue
    def test_split_cumsum_low_prec(self):
        if is_cpp_backend(self.device):
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        def fn(a):
            return torch.cumsum(a.view(-1), 0)

        self.common(
            fn,
            (torch.rand((10, 3, 352, 352), dtype=torch.float16),),
            reference_in_float=True,
            check_lowp=False,
        )

    def test_consecutive_split_cumsum(self):
        def fn(a, b):
            a = a.view(-1)
            b = b.view(-1)
            return torch.cumsum(a, 0) + torch.cumsum(b, 0)

        a = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float32, device=self.device)
        b = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float64, device=self.device)
        self.common(fn, (a, b), rtol=1e-4, atol=1e-5, check_lowp=False)

    def test_split_cumprod(self):
        def fn(a):
            return torch.cumprod(a, -1)

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            self.common(fn, (inp,), atol=1e-5, rtol=1e-4, check_lowp=False)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skipCUDAIf(TEST_WITH_ROCM, "Computation not done in float on ROCm")
    @skip_if_gpu_halide  # accuracy issue
    def test_split_cumprod_low_prec(self):
        if is_cpp_backend(self.device):
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        def fn(a):
            return torch.cumprod(a.view(-1), 0)

        for dtype in [torch.float16, torch.bfloat16]:
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            self.common(
                fn,
                (inp,),
                reference_in_float=True,
                check_lowp=False,
            )

    def test_consecutive_split_cumprod(self):
        def fn(a, b):
            return torch.cumprod(a, 0) + torch.cumprod(b, 0)

        a = _large_cumprod_input(
            (10000,), dim=0, dtype=torch.float32, device=self.device
        )
        b = _large_cumprod_input(
            (10000,), dim=0, dtype=torch.float64, device=self.device
        )
        self.common(fn, (a, b), atol=1e-5, rtol=1e-5, check_lowp=False)

    @skipCUDAIf(TEST_WITH_ROCM, "associative_scan is not supported on ROCm")
    @skip_if_halide  # scan ops
    def test_custom_scan_op(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def sum_combine(a, b):
            return a + b

        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(100, 100, device=self.device)
        expect = torch.cumsum(a, 0)
        actual = associative_scan(sum_combine, a, 0)
        self.assertEqual(expect, actual)

        def logcumsum_combine(a, b):
            min_v = torch.minimum(a, b)
            max_v = torch.maximum(a, b)
            mask = (min_v != max_v) | ~min_v.isinf()
            return torch.where(mask, max_v + (min_v - max_v).exp().log1p(), a)

        expect = torch.logcumsumexp(a, 0)
        actual = associative_scan(logcumsum_combine, a, 0)
        self.assertEqual(expect, actual)

    @skip_if_halide  # scan ops
    def test_custom_scan_op_compiled(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        from torch._higher_order_ops.associative_scan import associative_scan

        def sum_combine(a, b):
            return a + b

        def fn(a, b, dim):
            diff = (a - b).abs()
            sad = associative_scan(sum_combine, diff, dim)
            return sad.sum(dim)

        a = torch.randn(100, 100, device=self.device)
        b = torch.randn(100, 100, device=self.device)
        self.common(fn, (a, b, 0))
        cfn = torch.compile(fn)
        _, code = run_and_get_code(cfn, a, b, 0)

        # Check everything is fused into a single kernel
        FileCheck().check_not("run(").check_regex(
            r"triton_.*\.run\(arg[01]_1, arg[12]_1, buf1,"
        ).check_not("run(").run(code[0])

    @skipCUDAIf(TEST_WITH_ROCM, "associative_scan is not supported on ROCm")
    @skip_if_halide  # scan ops
    def test_custom_scan_op_multi_input(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def argmax_combine(a, b):
            a_value, a_index = a
            b_value, b_index = b
            mask = (a_value > b_value) | ((a_value == b_value) & (a_index > b_index))
            return (
                torch.where(mask, a_value, b_value),
                torch.where(mask, a_index, b_index),
            )

        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(100, 100, device=self.device)
        expect = torch.cummax(a, 0)

        idx = torch.arange(100, device=self.device).view(100, 1).expand(100, 100)
        actual = associative_scan(argmax_combine, (a, idx), 0)
        self.assertEqual(expect, actual)

    @skipCUDAIf(TEST_WITH_ROCM, "associative_scan is not supported on ROCm")
    @skip_if_halide  # scan ops
    def test_custom_scan_would_split(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def combine_linear_recurrence(left, right):
            xl, fl = left
            xr, fr = right
            x = xl * fr + xr
            f = fl * fr
            return x, f

        def eager_scan(x, g):
            x, g = x.to(torch.float64), g.to(torch.float64)
            x_out = torch.empty_like(x)
            g_out = torch.empty_like(g)
            x_out[:, 0] = x[:, 0]
            g_out[:, 0] = g[:, 0]
            for i in range(1, x.shape[1]):
                x_out[:, i], g_out[:, i] = combine_linear_recurrence(
                    (x_out[:, i - 1], g_out[:, i - 1]),
                    (x[:, i], g[:, i]),
                )
            return x_out.float(), g_out.float()

        @torch.compile
        def compiled_scan(x, f):
            from torch._higher_order_ops.associative_scan import associative_scan

            x, f = associative_scan(combine_linear_recurrence, (x, f), dim=1)
            return x, f

        x = torch.randn(1, 129, 2, device=self.device)
        f = torch.randn(1, 129, 2, device=self.device)
        expect = eager_scan(x, f)
        actual = compiled_scan(x, f)
        self.assertEqual(expect, actual)

    def test_embedding_bag_byte_unpack(self):
        if self.device != "cpu":
            raise unittest.SkipTest(f"No {GPU_TYPE} implementation (it returns empty)")

        def fn(a):
            return torch.ops.quantized.embedding_bag_byte_unpack(a)

        M, N = 32, 64
        scales = torch.randn(M, 1).view(torch.uint8)
        offsets = torch.randn(M, 1).view(torch.uint8)
        data = torch.randint(0, 255, (M, N), dtype=torch.uint8)
        packed = torch.cat([data, scales, offsets], dim=-1)
        self.common(fn, [packed])

    def test_expanded_reduction(self):
        def fn(x, y):
            z = x * y
            return z.sum((0, 1))

        atol = None
        rtol = None

        # By default, inductor generate non-persistent reduction kernels in this
        # case. But when multi-kernel is enabled, inductor will pick the faster
        # of persistent reduction and non-persistent-reduction kernel.
        # In this case, inductor picked the persistent-reduction kernel.
        # The persistent reduction kernel happens to need looser tolerance.
        if config.triton.multi_kernel:
            atol = 1e-5
            rtol = 1e-5
        self.common(
            fn, (torch.randn(2, 197, 256), torch.randn(2, 1, 256)), atol=atol, rtol=rtol
        )

    @skip_if_gpu_halide
    def test_min_max_reduction(self):
        def fn(a, b):
            return (
                (a + b).max(),
                (a + b).min(),
                torch.amax(a + 1, keepdim=True),
                torch.amin(b + 1, keepdim=True),
            )

        dtypes = [torch.float, torch.float16]
        if not (self.device == "cuda" and not SM80OrLater):
            dtypes += [torch.bfloat16]
        for dtype in dtypes:
            self.common(fn, (torch.randn(8, 8).to(dtype), torch.randn(8, 8).to(dtype)))

    @skip_if_halide  # bug in nan handling
    def test_min_max_reduction_nan(self):
        def fn(a):
            return (torch.max(a), torch.min(a))

        t1 = torch.randn(32)
        t1[16] = float("nan")
        self.common(fn, (t1,))

    @skip_if_halide  # bug in nan handling
    def test_fmin_fmax(self):
        def fn(a, b):
            return (
                torch.fmin(a, b),
                torch.fmax(a, b),
                torch.fmax(a + 1, torch.tensor(0.0)),
            )

        self.common(
            fn,
            (
                torch.tensor(
                    [-10.0, 10.0, float("nan"), float("nan"), float("nan"), 3, 4]
                ),
                torch.tensor(
                    [float("nan"), float("nan"), -10.0, 10.0, float("nan"), 4, 3]
                ),
            ),
        )

    def test_sum_int(self):
        def fn(x):
            return 2 * x.sum(-1) + x.sum()

        dtypes = torch.bool, torch.uint8, torch.int
        inps = [torch.randint(2, (64,), dtype=dtype) for dtype in dtypes]
        for i in inps:
            self.common(fn, (i,), check_lowp=False)

    def test_sum_dtype(self):
        def fn(x):
            return x * x.sum(-1, dtype=torch.double) + x.sum(dtype=torch.double)

        self.common(fn, (torch.ones(32, 32) * 70,))

    def test_cumsum(self):
        def fn(x):
            return x.cumsum(0), x.cumsum(1)

        # Persistent reductions
        self.common(
            fn, (torch.rand(16, 32),), check_lowp=not is_halide_backend(self.device)
        )
        self.common(
            fn, (torch.rand(20, 30),), check_lowp=not is_halide_backend(self.device)
        )

        # Non-persistent reduction
        self.common(
            fn,
            (torch.rand(100, 4000),),
            check_lowp=not is_halide_backend(self.device),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cumsum_zero_dim(self):
        def fn(x):
            return x.cumsum(0), x.cumsum(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_cumsum_no_mask(self):
        def fn(x):
            return x.cumsum(-1)

        # Persistent reduction
        a = torch.rand((1, 1024))
        self.common(
            fn, (a,), check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device))
        )

        # Non-persistent reduction
        b = torch.rand((1, 8192))
        self.common(
            fn,
            (b,),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cumprod_zero_dim(self):
        def fn(x):
            return x.cumprod(0), x.cumprod(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_logcumsumexp(self):
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(1)

        # Persistent reductions
        self.common(
            fn,
            (torch.rand(16, 32),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )
        self.common(
            fn,
            (torch.rand(20, 30),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )

        # Non-persistent reduction
        self.common(
            fn,
            (torch.rand(100, 4000),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_logcumsumexp_zero_dim(self):
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_clamp(self):
        def fn(a, b):
            return (a.clamp(-0.1, 0.1), b.clamp(0), torch.clamp(a + b, max=0))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_clamp_type_promotion(self):
        def fn(a):
            b = torch.tensor(1.0, dtype=torch.double, device=self.device)
            c = torch.full((4,), 2, device=self.device)
            return a.clamp(min=b, max=c)

        self.common(fn, (torch.randint(4, (4,)),))

    @skip_if_gpu_halide
    def test_dist(self):
        def fn(a, b):
            return (
                torch.dist(a, b),
                torch.dist(a, b, p=1.2),
            )

        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_dist_bf16(self):
        def fn(a, b):
            return torch.dist(a.to(torch.bfloat16), b.to(torch.bfloat16))

        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    def test_arange1(self):
        def fn(x):
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        self.common(fn, (torch.randn(8, 8),))

    def test_arange2(self):
        def fn(x):
            rng1 = torch.arange(8, device=x.device)
            return (x + rng1,)

        self.common(fn, (torch.randint(4, (8, 8)),), check_lowp=False)

    def test_arange3(self):
        def fn(x):
            return x + torch.ops.aten.arange.start_step(
                0, 53, 4, dtype=torch.int64, device=x.device
            )

        self.common(fn, (torch.randn(14),))

    def test_arange4(self):
        def fn(x):
            return x - torch.arange(512, -512, -1.0, device=x.device)

        self.common(fn, (torch.randn(1024),))

    @skipIfWindows  # TODO: debug it on Windows
    def test_arange5(self):
        def fn(step, device):
            return torch.arange(512, -512, step, device=device)

        compiled_fn = torch._dynamo.optimize()(fn)

        # NOTE: use assertEqual to check dtypes which self.common doesn't do
        for step in (-1, -1.0):
            expect = fn(step, self.device)
            actual = compiled_fn(step, self.device)
            self.assertEqual(expect, actual)
        self.assertEqual(expect, actual)

    def test_arange6(self):
        def fn(x):
            return torch.arange(0.1, 8.0001, 1, dtype=x.dtype, device=x.device)

        # Test that float arguments are truncated to int when dtype is set explicitly
        make_arg = functools.partial(
            make_tensor, device=self.device, requires_grad=False
        )
        self.common(fn, (make_arg(1, dtype=torch.float32),))
        self.common(fn, (make_arg(1, dtype=torch.int64),))

    def test_linspace1(self):
        def fn(x):
            return torch.linspace(0.125, 0.875, 7, device=x.device) + x

        self.common(fn, (torch.randn(1, 7),))

    def test_linspace2(self):
        def fn(x):
            return torch.linspace(0, 2, 1, device=x.device) + x

        self.common(fn, (torch.randn(1, 1),))

    def test_linspace3(self):
        def fn(x):
            return torch.linspace(0, 2, 0, device=x.device)

        self.common(fn, (torch.Tensor([]),))

    def test_tensor1(self):
        def fn(x):
            return torch.tensor([1], device=x.device) + x, torch.tensor(
                5, device=x.device
            )

        self.common(fn, (torch.randn(10),))

    def test_tensor2(self):
        def fn(x):
            return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

        self.common(fn, (torch.randn(1),))

    def test_tensor3(self):
        def fn(x):
            return (
                torch.tensor([], device=x.device),
                torch.tensor([1, 2], device=x.device) + 1,
                torch.tensor([1, 2, 3], device=x.device) + 2,
                torch.tensor([1, 2, 3, 4], device=x.device) + x,
            )

        self.common(fn, [torch.randn(4)])

    def test_views1(self):
        def fn1(x, y):
            return (x.view(size2) + y,)

        def fn2(x, y):
            return ((x + 1).view(size2) + y,)

        views = [
            ([5 * 7], [5, 7]),
            ([2 * 3 * 4 * 5 * 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([2 * 3, 4, 5, 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([10 * 5, 20], [10, 5, 20]),
            ([1, 10, 1], [10]),
            ([10, 1, 10, 1, 10], [10, 100]),
            ([2, 2, 2, 2], [4, 4]),
        ]
        for size1, size2 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

        for size2, size1 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

    def test_views2(self):
        def fn1(x):
            return (x.view(size2) + 1,)

        def fn2(x):
            return ((x * 2).view(size2) + 1,)

        for size1, size2 in [
            ([2, 2, 2, 2], [4, -1]),
            ([10, 1, 10, 1, 10], [-1, 100]),
            ([10 * 5, 20], [10, -1, 20]),
        ]:
            self.common(fn1, (torch.randn(size1),))
            self.common(fn2, (torch.randn(size1),))

    def test_views3(self):
        # example taken from hf_BigBird
        def forward(arg1, arg2):
            index = torch.ops.aten.index(arg1, [arg2])
            view_1 = torch.ops.aten.view(index, [1, 2232, 64])
            view_2 = torch.ops.aten.view(view_1, [1, 12, 62, 192])
            return view_2

        self.common(
            forward,
            (
                rand_strided((64, 64), (64, 1), torch.float32),
                rand_strided((2232,), (1,), torch.int64),
            ),
        )

    def test_views4(self):
        # example taken from hf_BigBird
        def forward(arg1, arg2):
            arg1 = arg1.index_select(0, arg2)
            arg1 = torch.ops.aten.view(arg1, [2, 3, 4, 5, 5])
            arg1 = torch.ops.aten.view(arg1, [2, 3, 2, 10, -1])
            return arg1

        self.common(
            forward,
            (
                torch.randn(12, 5, 5),
                torch.randint(0, 11, (24,)),
            ),
        )

    def test_views5(self):
        # tensor with shape 0 in any dimension
        def forward(x):
            y = x[:, 4:]
            return y.view(len(y), -1, 4)

        self.common(
            forward,
            (torch.randn(4, 4, 4, 4),),
        )

    def test_views6(self):
        def forward(x):
            x = torch.ops.aten.relu(x)
            s = torch.ops.aten.slice(x, 0, 0, 9223372036854775807)
            s = torch.ops.aten.slice(s, 1, 0, 9223372036854775807)
            s = torch.ops.aten.slice(s, 3, 0, 0)
            y = torch.ops.aten.view(s, [4, 2, -1])
            return y

        self.common(
            forward,
            (torch.randn(4, 2, 4, 4),),
        )

    def test_views7(self):
        # x.view(dtype)
        def forward(x, y):
            x = (x + 1).to(torch.float32)
            y = (y + 1).to(torch.int32)
            return x.view(torch.int32), y.view(torch.float32)

        self.common(
            forward,
            (
                torch.rand(2, 3, dtype=torch.float32),
                torch.randint(10, (2, 3), dtype=torch.int32),
            ),
        )

    def test_relu(self):
        def fn(a, b):
            return (torch.relu(a), torch.relu(a + b) / 10)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp(self):
        def fn(a, b):
            return (torch.exp(a), torch.exp(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp2(self):
        def fn(a, b):
            return (torch.exp2(a), torch.exp2(a + b), torch.pow(2, -torch.abs(a - b)))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sigmoid(self):
        def fn(a, b):
            return (torch.sigmoid(a), torch.sigmoid(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_round(self):
        def fn(a, b):
            return torch.round(a), torch.round(b + 1), torch.round(a, decimals=2)

        # without manual_seed, there is some chance this test fails due to:
        # https://github.com/openai/triton/issues/530
        torch.manual_seed(0)

        # with *100 we are always getting a number exactly at .5 which we don't do right in half
        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 10))

    def test_round_correctness(self):
        if self.device == "cuda":
            raise unittest.SkipTest("need to debug tl.libdevice on A100/V100")

        def fn(a):
            return torch.round(a)

        self.common(
            fn,
            [torch.arange(-10, 10, 0.1, dtype=torch.float64)],
            check_lowp=False,
        )

    def test_builtins_round(self):
        def fn(x, i):
            return x[: round(i / 2 + 1)] + round(i / 2)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(5, dtype=torch.int, device=self.device)
        with torch.no_grad():
            for i in range(1, 6):
                self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_float_ndigits_pos(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_float_ndigits_zero(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 0)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_float_ndigits_neg(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, -1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_int_ndigits_pos(self):
        def fn(x, i):
            return x + round(i, 1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 123

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_int_ndigits_zero(self):
        def fn(x, i):
            return x + round(i, 0)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 123

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_silu(self):
        def fn(a):
            return (torch.nn.functional.silu(a),)

        self.common(fn, (torch.randn(8, 8),))

    @skip_if_halide  # halide has buggy nan handling
    def test_nan_to_num(self):
        def fn(a):
            return (
                torch.nan_to_num(a),
                torch.nan_to_num(a, nan=3.0),
                torch.nan_to_num(a, nan=None),
                torch.nan_to_num(a, posinf=4.0),
                torch.nan_to_num(a, neginf=5.0),
                torch.nan_to_num(a, nan=3.0, posinf=4.0, neginf=5.0),
            )

        self.common(
            fn,
            (torch.tensor((float("nan"), float("inf"), float("-inf"), 1.0)),),
            check_lowp=False,  # a much more elaborate test is required to match finfo max's for float and half
        )

    def test_one_hot(self):
        def fn(a):
            return torch.nn.functional.one_hot(a, 8) + 1

        self.common(
            fn,
            (torch.arange(100).view(4, 5, 5) % 8,),
            check_lowp=False,
        )

    def test_div1(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 100))

    def test_div2(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randint(-100, 100, [8, 8]), 100 * torch.randn(8, 8)))

    def test_div3(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        a = torch.randint(1, 100, [8, 8])
        self.common(fn, (a * 2, a))

    def test_div4(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (torch.randint(-100, 0, [8, 8]), torch.randint(1, 10, [8, 8])),
        )

    def test_div5(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # divide a scalar
        self.common(fn, (torch.randint(-100, 0, [8, 8]), 16))

    def test_div6(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # treat boolean as integer
        self.common(
            fn,
            (torch.ones([8, 8], dtype=torch.bool), torch.randint(-100, -1, [8, 8])),
        )

    def test_div7(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (
                torch.randint(2**32, 2**40, [100, 100]),
                torch.randint(-10, -1, [100, 100]),
            ),
        )

    def test_div8(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a * 0.5, b, rounding_mode=None),
                aten.div(a, b * 1.0, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (1024, 100))

    def test_div9(self):
        def fn(x):
            return (torch.div(42, x), aten.true_divide(42, x), aten.div.Tensor(42, x))

        self.common(fn, (torch.randn(8),))

    def test_div_zero_dim(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        for dtype in (torch.float32, torch.int64):
            self.common(
                fn,
                (
                    make_tensor(10, device=self.device, dtype=dtype),
                    make_tensor((), device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )
            self.common(
                fn,
                (
                    make_tensor((), device=self.device, dtype=dtype),
                    make_tensor(10, device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )

    def test_div_prim(self):
        def fn(a, b):
            return (torch.ops.prims.div(a, b),)

        for dtype in (torch.float32, torch.int64):
            self.common(
                fn,
                (
                    make_tensor(100, device=self.device, dtype=dtype),
                    make_tensor(
                        100, device=self.device, dtype=dtype, exclude_zero=True
                    ),
                ),
            )

    def test_floordiv(self):
        def fn_floor_input(a, i):
            n = (i * 1.234) // 8.234
            return a + n

        self.common(
            fn_floor_input,
            (make_tensor(10, device=self.device, dtype=torch.float32), 33),
        )

        def fn_int_input(a, i):
            n = i // 8
            return a + n

        self.common(
            fn_int_input, (make_tensor(10, device=self.device, dtype=torch.float32), 33)
        )

    def test_div_precision(self):
        # Reproducer for https://github.com/pytorch/pytorch/issues/101039

        def forward(x, y):
            z = x.div(y)
            return F.softmax(z, dim=-1)

        query = torch.randn(1, 10, 40)
        key = torch.randn(1, 2, 40)
        x = torch.matmul(query, key.transpose(-2, -1))
        self.common(forward, (x, 1e-6))

        x = torch.tensor(
            [
                [
                    [
                        [-16.1649, 5.6846, -5.1022, -9.1134],
                        [-11.5552, -2.2615, -12.8913, 10.6538],
                        [-7.1666, -5.3333, 2.0776, -9.7984],
                        [7.4469, -2.3948, 2.7371, 0.9201],
                    ],
                    [
                        [-8.0361, -16.3771, 22.7741, 4.4685],
                        [20.8047, -0.7771, -2.4355, -2.2299],
                        [3.8343, -2.0914, -2.4077, 2.2740],
                        [-15.8663, -2.7015, -12.5241, -3.0040],
                    ],
                    [
                        [-2.5139, 14.4393, -3.7186, 1.2255],
                        [5.6742, 14.1842, -8.5976, 16.8366],
                        [-9.7358, -3.0279, 11.8164, -4.0787],
                        [-9.0621, 8.2580, 29.9486, -2.4107],
                    ],
                    [
                        [7.3622, 12.5640, -20.5592, 13.6237],
                        [-11.5640, 0.8832, 16.7275, -2.5009],
                        [-2.0953, -12.2276, -26.2633, 4.5268],
                        [15.3329, -11.7492, 6.5650, -9.2483],
                    ],
                ],
                [
                    [
                        [7.9980, -4.9369, 3.1508, 5.2994],
                        [3.8052, 3.9514, 8.4987, -10.5045],
                        [-2.6827, -4.0010, -4.0611, 6.4091],
                        [-19.0318, 6.4073, 2.8923, 8.0250],
                    ],
                    [
                        [7.1650, -3.4585, 5.7720, -5.0305],
                        [-0.9765, -3.0086, 11.7114, 8.0555],
                        [-3.1027, -3.5514, 9.6182, -8.8526],
                        [-9.2348, -6.0239, 6.2528, -6.7221],
                    ],
                    [
                        [11.5936, 22.4139, -0.4089, -4.9889],
                        [14.8217, -2.3426, -17.6189, 3.7427],
                        [1.9546, -13.0902, 8.6293, -7.2457],
                        [-7.6900, -4.5796, 9.6332, -10.2631],
                    ],
                    [
                        [0.8027, -1.0955, 14.8404, -0.2673],
                        [3.2143, -1.8640, -2.9678, 6.5165],
                        [-3.9865, 6.5230, 6.3019, -0.4247],
                        [8.3185, -13.5076, 27.0986, -1.6792],
                    ],
                ],
            ]
        )
        x = torch.matmul(x, x)
        y = torch.tensor([[[0.6331]], [[1.6358]], [[-0.3459]], [[1.0196]]])
        self.common(forward, (x, y))

    def test_div_by_zero(self):
        def fn(x, runtime_zero, runtime_neg_zero):
            zero = torch.zeros_like(x)
            return (
                x / 0.0,
                x / -0.0,
                zero / 0.0,
                x / zero,
                x / -zero,
                zero / zero,
                x / runtime_zero,
                # NOTE: -runtime_zero doesn't work as -(0.0) is broken in triton
                x / runtime_neg_zero,
                runtime_zero / runtime_neg_zero,
            )

        a = torch.randn(10)
        zero = torch.zeros(10)
        neg_zero = -zero
        self.common(fn, (a, zero, neg_zero))

    def test_both_scalars(self):
        def fn(a, b):
            return (
                aten.add(a, b),
                aten.add(b, a),
                aten.sub(a, b),
                aten.sub(b, a),
                aten.mul(a, b),
                aten.mul(b, a),
            )

        self.common(fn, (4, 3.3), reference_in_float=False)

    def test_sum_keepdims(self):
        def fn(a, b):
            return (torch.sum(a + b, -1, keepdim=True),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    @skip_if_halide  # only 32-bit indexing
    def test_large_tensor_reduction(self):
        if not _has_sufficient_memory(self.device, 4.5 * 1024**3):  # 4.5 GiB
            raise unittest.SkipTest("insufficient memory")

        if self.device == "cpu":
            raise unittest.SkipTest("Fails on CPU")

        # Test 64-bit indexing works correctly
        def fn(a):
            return torch.max(a)

        t = torch.ones(2**32, dtype=torch.int8, device=self.device)
        t[-1] = 2

        # self.common OOMs here because it copies inputs to check for mutations
        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(t)
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    @skip_if_gpu_halide  # only 32-bit indexing
    def test_large_broadcast_reduction(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Fails on CPU")

        # Test 64-bit indexing works correctly when inputs are less than 32-bit
        # but intermediate tensors require 64-bit indexing
        def fn(a, b):
            return torch.max(a + b)

        t1 = torch.ones(1, 2**16, dtype=torch.int8, device=self.device)
        t2 = torch.ones(2**16, 1, dtype=torch.int8, device=self.device)

        t1[-1, -1] = 2
        t2[-1, -1] = 2

        # self.common OOMs here because it copies inputs to check for mutations
        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(t1, t2)
        expect = torch.tensor(4, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    @skip_if_halide  # only 32-bit indexing
    def test_large_pointwise(self):
        if not _has_sufficient_memory(self.device, 2 * (2**31 + 1)):
            raise unittest.SkipTest("insufficient memory")

        def fn(a):
            return a + 1

        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(t)

        # Can't use assertEqual as it expands broadcasted inputs
        del t
        if torch.device(self.device).type == GPU_TYPE:
            getattr(torch, GPU_TYPE).empty_cache()

        self.assertTrue((actual == 2).all())

    @skip_if_halide  # only 32-bit indexing
    def test_large_offset_pointwise(self):
        # Test 64-bit indexing is used when input views a tensor that can be
        # indexed with 32-bit strides but the storage offset pushes it over
        # INT_MAX
        if not _has_sufficient_memory(self.device, (2**31 + 1) + (2**30 + 1)):
            raise unittest.SkipTest("insufficient memory")

        def fn(a):
            return a + 4

        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        t[2**30 :] = 0
        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(t[2**30 :])
        self.assertTrue((actual == 4).all())

    @skip_if_halide  # only 32-bit indexing
    def test_large_strided_reduction(self):
        # Test 64-bit indexing is used when input numel is less than INT_MAX
        # but stride calculations go above INT_MAX
        if not _has_sufficient_memory(self.device, 2**31 + 2):
            raise unittest.SkipTest("insufficient memory")

        def fn(a):
            return torch.max(a)

        storage = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        view = storage[::32]
        view[-1] = 2

        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(view)
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    def test_softmax(self):
        def fn(a, b):
            return (torch.softmax(a + b, -1), torch.softmax(a, 0), torch.softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_log_softmax(self):
        def fn(a, b):
            return (F.log_softmax(a + b, -1), F.log_softmax(a, 0), F.log_softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_transpose(self):
        def fn(a, b):
            return (
                torch.t(a) + b,
                torch.transpose(b * 2, 0, 1) + 10,
            )

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_permute1(self):
        def fn(a):
            return (
                torch.permute(a + 1, [2, 1, 4, 0, 3]) + 2,
                torch.permute(a, [2, 1, 4, 0, 3]) + 2,
            )

        self.common(fn, (torch.randn(2, 2, 2, 2, 2),))

    def test_permute2(self):
        def fn(a):
            a = a.unfold(0, 2, 1)
            a = torch.unsqueeze(a, 1)
            a = torch.permute(a, [0, 2, 3, -3])
            return (a,)

        self.common(fn, (torch.randn(4, 4),))

    def test_expand(self):
        def fn(a):
            return (
                (a + 1).expand(3, 4, 2, 3, 2) + 2,
                a.expand(2, 1, 2, 3, 2) + 2,
            ), a.expand(2, -1, 5, -1)

        self.common(fn, (torch.randn(2, 1, 2),))

    def test_squeeze1(self):
        def fn(a):
            return ((a + 1).squeeze() + 2, a.squeeze() + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 1, 1),))

    def test_squeeze2(self):
        def fn(a):
            return ((a + 1).squeeze(-1).squeeze(2) + 2, a.squeeze(0) + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 2, 1),))

    def test_squeeze_varargs(self):
        def fn(x):
            return x.squeeze(1, 2).clone()

        a = torch.randn(1024, 1, 1)
        self.common(fn, (a,))

    def test_simplify_loops(self):
        def fn(a, b):
            return a + b

        self.common(
            fn,
            (
                torch.randn(2, 3, 4, 5, 6),
                torch.randn(4, 2, 3, 5, 6).permute(1, 2, 0, 3, 4),
            ),
        )

    def test_unsqueeze(self):
        def fn(a):
            return (
                torch.unsqueeze(a + 1, -1) + 2,
                torch.unsqueeze(a, 2) + 2,
                torch.unsqueeze(a + 1, 0) + 2,
                torch.unsqueeze(a, -2) + 2,
            )

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_unsqueeze_inplace(self):
        def fn(a):
            tmp1 = a + 1
            aten.unsqueeze_(tmp1, 2)
            tmp2 = aten.unsqueeze_(a + 1, 0) + 2
            return (tmp1, tmp2)

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_addmm(self):
        def fn(a, b, c):
            return (torch.addmm(a + 1, b + 2, c + 3) + 4,)

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randn(8, 8),
                torch.randn(8, 8),
            ),
        )

    # https://github.com/pytorch/pytorch/issues/98979
    @skipCUDAIf(True, "cuda failed for float64 linear")
    @skipIfXpu(msg="Double and complex datatype matmul is not supported in oneDNN")
    def test_linear_float64(self):
        mod = torch.nn.Sequential(torch.nn.Linear(8, 16).to(torch.float64)).eval()
        with torch.no_grad():
            self.common(mod, (torch.randn(2, 8).to(torch.float64),))

    def test_linear1(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Sigmoid(),
            ToTuple(),
        )
        self.common(mod, (torch.randn(2, 8),))

    def test_linear2(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
        )
        self.common(
            mod,
            (torch.randn(2, 8),),
            atol=1e-3,
            rtol=0.01,
        )

    def test_bmm1(self):
        def fn(a, b):
            return (
                torch.bmm(a, b),
                torch.bmm(a + 1, b + 2) + 3,
            )

        self.common(
            fn,
            (
                torch.randn(2, 8, 8),
                torch.randn(2, 8, 8),
            ),
            check_lowp=False,
        )
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),
                torch.randn(1, 8, 10),
            ),
            check_lowp=False,
        )

    def test_bmm2(self):
        def fn(a, b):
            return torch.bmm(a.permute(0, 2, 1), b)

        self.common(
            fn,
            (
                torch.randn(1, 8, 8),
                torch.randn(1, 8, 8),
            ),
            check_lowp=False,
        )

    @skipIfPy312  # segfaults
    @config.patch(mixed_mm_choice="triton")
    def test_mixed_mm(self):
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),
            ),
            check_lowp=True,
        )

    @skipIfPy312  # segfaults
    @config.patch(mixed_mm_choice="triton")
    def test_mixed_mm2(self):
        def fn(a, b, scale, bias):
            return torch.mm(a, b.to(a.dtype)) * scale + bias

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),
                torch.randn(8),
                torch.randn(8),
            ),
            check_lowp=True,
        )

    @skipIfPy312  # segfaults
    @config.patch(mixed_mm_choice="triton")
    def test_mixed_mm3(self):
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # (256, 256) @ (256, 256) so different block sizes are tried out during autotuning
        self.common(
            fn,
            (
                torch.randn(256, 256),
                torch.randint(-128, 127, (256, 256), dtype=torch.int8),
            ),
            check_lowp=True,
            rtol=0.01,
            atol=0.1,
        )

    @with_tf32_off
    @config.patch(use_mixed_mm=True)
    def test_uint4x2_mixed_mm(self):
        def fn(a, b):
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)
                .reshape(-1, b.shape[1])
                .to(a.dtype)
                .sub(8),
            )

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(0, 255, (4, 8), dtype=torch.uint8),
            ),
            check_lowp=True,
        )

    @skipIfXpu
    def test_mm_mixed_dtype(self):
        def fn(a, b):
            return torch.mm(a, b)

        t1 = torch.arange(6, dtype=torch.float, device=self.device).view(2, 3)
        t2 = torch.arange(9, dtype=torch.int64, device=self.device).view(3, 3)

        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn)(t1, t2)
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t1, t2)

    @skipIfXpu
    def test_linear_mixed_dtype(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()  # noqa: UP008
                self.fc1 = nn.Linear(3, 3)

            def forward(self, x):
                x = self.fc1(x.permute(1, 2, 0))
                return x

        fn = Net().to(self.device)
        t = torch.arange(27, device=self.device).view(3, 3, 3)

        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t)
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.no_grad():
                torch.compile(fn)(t)
        # TODO: Autograd internal assertion
        msg = r".*isDifferentiableType\(variable.scalar_type\(\)\) INTERNAL ASSERT FAILED.*"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn)(t)

    def test_scalar_input(self):
        def fn(x, y):
            a = torch.div(x, y, rounding_mode="floor")
            return a

        self.common(fn, [torch.randint(5, (1, 8)), 5400])

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_scalar_output(self):
        def fn(arg0_1, arg2_1):
            arg1_1 = arg2_1.size(1)
            view = torch.ops.aten.view.default(arg2_1, [-1, arg1_1])
            embedding = torch.ops.aten.embedding.default(arg0_1, view)
            full = torch.ops.aten.full.default([1, arg1_1], 1, dtype=torch.float32)
            return (full, arg1_1, embedding)

        arg0_1 = rand_strided((32128, 768), (768, 1), device="cpu", dtype=torch.float32)
        arg2_1 = rand_strided((1, 22), (22, 1), device="cpu", dtype=torch.int64)
        self.common(fn, [arg0_1, arg2_1])

    def test_shape_prop_torch_ones(self):
        class Model(torch.nn.Module):
            def forward(self, attention_scores):
                extended_attention_mask = torch.ones(
                    8, 1, 1, 512, device=attention_scores.device
                )
                attention_scores = attention_scores + extended_attention_mask

                return attention_scores

        mod = Model().eval()
        with torch.no_grad():
            self.common(
                mod,
                (torch.randn(8, 12, 512, 512),),
            )

    @slowTest
    @expectedFailureCodegenDynamic
    @config.patch({"freezing": True})
    def test_conv_bn_fuse(self):
        # For gpu path, there is an accuracy issue
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv bn test")

        # fails dynamic check which bn is fused, and there will not have loops vars.
        input_shapes = {1: (112,), 2: (112, 112), 3: (55, 55, 55)}
        conv_modules = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        bn_modules = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }
        options = itertools.product(
            [1, 2, 3],
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
        )

        for (
            dim,
            bias,
            kernel_size,
            dilation,
            groups,
        ) in options:
            oC = 32 * groups
            iC = 3 * groups
            x_shape = (1, iC) + input_shapes[dim]
            mod = torch.nn.Sequential(
                conv_modules[dim](
                    iC,
                    oC,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
                bn_modules[dim](oC),
            ).eval()
            test_memory_format = [torch.contiguous_format]
            # TODO: GPU path doesn't support channels_last now.
            if not HAS_GPU and dim > 1:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                test_memory_format.append(channels_last)
            for memory_format in test_memory_format:
                v = torch.randn(x_shape, dtype=torch.float32).to(
                    memory_format=memory_format
                )
                with torch.no_grad():
                    self.common(
                        mod,
                        (v,),
                    )

    def test_conv_functional_bn_fuse(self):
        # For gpu path, there is an accuracy issue
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv bn test")

        # Define a BatchNorm using functional BN.
        class BatchNorm(torch.nn.BatchNorm2d):
            def __init__(
                self,
                num_features,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
                device=None,
                dtype=None,
            ):
                factory_kwargs = {"device": device, "dtype": dtype}
                super().__init__(
                    num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    **factory_kwargs,
                )

            def forward(self, x):
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:  # type: ignore[has-type]
                        self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(
                                self.num_batches_tracked
                            )
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum
                if self.training:
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (
                        self.running_var is None
                    )
                x = F.batch_norm(
                    x,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    (
                        self.running_mean
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    (
                        self.running_var
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
                return x

        v = torch.randn(1, 3, 556, 56, dtype=torch.float32)
        mod = torch.nn.Sequential(
            torch.nn.Conv2d(
                3,
                64,
                kernel_size=3,
                dilation=1,
                groups=1,
                bias=True,
            ),
            BatchNorm(64),
        ).eval()
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    @skipIfRocm
    def test_conv_inference_heuristics(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest(f"{GPU_TYPE} only test")

        in_channels = 6
        out_channels = 6
        kernel_size = 3
        groups = 3

        grouped_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, groups=groups
        ).to(self.device)

        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        # Perform the forward pass
        @torch.compile()
        def foo(m, inp):
            return m(inp)

        with torch.no_grad():
            _, code = run_and_get_code(foo, grouped_conv, input_tensor)
            # no to channels last permuting before kernel
            FileCheck().check_not(".run(").check(".convolution(").run(code[0])

        # in out should do channels last in inference
        in_channels = 8
        out_channels = 4
        kernel_size = 3

        # Create the convolution layer
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size).to(self.device)

        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        with torch.no_grad():
            _, code = run_and_get_code(foo, conv_layer, input_tensor)
            # should be channels last permuting before kernel
            if is_halide_backend(self.device):
                FileCheck().check("halide_kernel_0(").check(".convolution(").run(
                    code[0]
                )
            else:
                FileCheck().check(".run(").check(".convolution(").run(code[0])

    def test_upsample_cat_conv(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu upsample_cat_conv test")

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
                self.conv = torch.nn.Conv2d(
                    8,
                    5,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    dilation=1,
                    **kwargs,
                )

            def forward(self, x, y):
                x = self.upsample(x)
                z = torch.cat([x, y], dim=1)
                z = self.conv(z)
                return z

        v1 = torch.randn([8, 2, 12, 26])
        v2 = torch.randn([8, 6, 24, 52])

        with torch.no_grad():
            self.common(
                M().eval(),
                (v1, v2),
            )

    def test_aliased_buffer_reuse(self):
        def fn(x, y):
            x = 2 * x
            y = 2 * y
            c = torch.cat([x, y], dim=-1)
            d = 1 + c
            m = torch.mm(d, d)
            return m[:, :2] + x

        self.common(fn, (torch.randn(4, 2), torch.randn(4, 2)), check_lowp=False)

    def test_slice_view_with_graph_break(self):
        def fn():
            a = torch.tensor([1], device=self.device)
            a = a[0:1]
            b = a.squeeze()
            a[0] = 0
            if a[0] < 1e5:
                pass
            a[0] = 2
            return b

        expect = fn()
        opt_fn = torch.compile(fn)
        actual = opt_fn()
        self.assertEqual(expect, actual)

    def test_view_detach(self):
        def fn(a):
            return a[0].detach()

        self.common(
            fn,
            (torch.randn([4, 4], requires_grad=True),),
        )

    def test_gather1(self):
        def fn(a, b):
            return (
                torch.gather(a.expand([4, 5, 10, 6]), 3, b + 1),
                torch.gather(a.expand([4, 5, 10, 6]), -1, b + 1),
            )

        self.common(
            fn,
            (
                torch.randn([1, 1, 10, 6]),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_gather2(self):
        # 0d tensor
        def fn(a, b):
            return torch.gather(a, 0, b) + torch.gather(a, -1, b)

        x = torch.tensor(123)
        y = torch.tensor(0)
        self.assertEqual(fn(x, y), x + x)

    def test_gather3(self):
        def fn(a, b):
            return torch.gather(a, 1, b, sparse_grad=True)

        self.common(
            fn,
            (
                torch.randn([4, 5, 10, 6], requires_grad=True),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_slice1(self):
        def fn(a):
            return (
                a[:, :10, 0] + a[:, 10:, 0],
                (a + 1)[:, :10, 0] + (a + 1)[:, 10:, 0],
                a[:, -30:, 0],  # negative index out of range
                a[:, :-30, 0],  # negative index out of range
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_slice2(self):
        def fn(a):
            return (
                a[:-1, ::2, -1] + a[-1:, 1::2, -2],
                (a + 1)[:-1, ::2, -1] + (a + 2)[-1:, 1::2, -2],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    # It's a view so it doens't generate a kernel
    @expectedFailureCodegenDynamic
    def test_slice3(self):
        def fn(a, b):
            return torch.ops.aten.slice.Tensor(a, 0, 0, -b)

        x = torch.rand(48, 3, 512, 512)
        self.common(fn, (x, 2))

    @expectedFailureCodegenDynamic
    def test_slice4(self):
        # empty slices that require clamping the start or end
        def fn(a):
            return (
                aten.slice.Tensor(a, 0, 2, 0, 1),
                aten.slice.Tensor(a, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice.Tensor(a, 0, -20, 0, 1),
                aten.slice.Tensor(a, 0, -20, -16, 1),
            )

        x = torch.rand(10)
        self.common(fn, (x,))

    def test_split_with_list(self):
        def fn(a, sizes):
            return [t + 1.0 for t in torch.split(a * 2.0, sizes, -1)]

        self.common(fn, (torch.randn(2, 2, 10), [3, 3, 4]))
        self.common(fn, (torch.randn(2, 2, 10), [4, 3, 3]))
        self.common(fn, (torch.randn(2, 2, 10), [1, 2, 3, 4]))

    def test_split_with_integer(self):
        # argument `split_size_or_sections` is integer
        @torch.compile(dynamic=True)
        def f(x, sizes):
            return torch.split(x, sizes, -1)

        # split into equally sized chunks, 10 = 5 + 5
        r1, r2 = f(torch.randn(2, 10), 5)
        self.assertTrue(r1.size() == (2, 5))
        self.assertTrue(r2.size() == (2, 5))

        # split into equally sized chunks, 12 = 4 + 4 + 4
        r1, r2, r3 = f(torch.randn(2, 12), 4)
        self.assertTrue(r1.size() == (2, 4))
        self.assertTrue(r2.size() == (2, 4))
        self.assertTrue(r3.size() == (2, 4))

        # split unevenly, 10 = 3 + 3 + 3 + 1
        r1, r2, r3, r4 = f(torch.randn(2, 10), 3)
        self.assertTrue(r1.size() == (2, 3))
        self.assertTrue(r2.size() == (2, 3))
        self.assertTrue(r3.size() == (2, 3))
        self.assertTrue(r4.size() == (2, 1))

    def test_split_failed(self):
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.split(a, [2, 1, 1], dim=1)

        with self.assertRaisesRegex(RuntimeError, ""):
            fn(torch.randn(1, 5))

    @skipIfWindows
    def test_inductor_assert(self):
        @torch._dynamo.optimize("inductor", dynamic=True)
        def fn(a):
            assert a.shape[0] >= 2 and a.shape[1] >= 4
            return a.cos()

        inp = torch.randn(2, 4, 6)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)
        self.assertEqual(fn(inp), inp.cos())

    def test_split(self):
        def fn(a):
            t = torch.split(a, 3, -1)
            return (t[0], t[1], t[2], t[3])

        def fn2(a):
            return fn(a + 1)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

        self.common(
            fn2,
            (torch.randn([2, 2, 10]),),
        )

    def test_low_memory_max_pool(self):
        prims = torch.ops.prims

        def fn(x):
            kernel_size = [3, 3]
            stride = [2, 2]
            padding = [1, 1]
            dilation = [1, 1]
            ceil_mode = False

            vals, offsets = prims._low_memory_max_pool2d_with_offsets(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )
            indices = prims._low_memory_max_pool2d_offsets_to_indices(
                offsets,
                kernel_size[1],
                x.size(-1),
                stride,
                padding,
            )
            return vals, indices, offsets

        self.common(fn, (torch.randn(1, 3, 10, 10),))

    def test_to_dtype(self):
        def fn(a, b):
            return (
                aten._to_copy(a, dtype=6),
                aten._to_copy(b + 1, dtype=6),
                aten.to(b, torch.float64),
                aten.to(b, torch.bool),
            )

        self.common(
            fn,
            (
                torch.randn([2, 2, 10]),
                torch.randn([2, 2, 10], dtype=torch.float64),
            ),
        )

    @requires_gpu()
    def test_to_device(self):
        def fn(a):
            if a.device.type == "cpu":
                return aten._to_copy(
                    a, device=torch.device(GPU_TYPE), dtype=6, layout=0
                )
            else:
                return aten._to_copy(a, device=torch.device("cpu"), dtype=6, layout=0)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

    def test_to_memory_format(self):
        def fn(a, memory_format):
            return a.to(memory_format=memory_format)

        self.common(
            fn,
            (torch.randn([2, 2, 10, 10]), torch.channels_last),
        )
        self.common(
            fn,
            (
                torch.randn([2, 2, 10, 10]).to(memory_format=torch.channels_last),
                torch.contiguous_format,
            ),
        )

    @requires_gpu()
    def test_to_device_constant(self):
        def fn(a):
            d1 = a.device.type
            if d1 == "cpu":
                d2 = GPU_TYPE
            else:
                d2 = "cpu"

            const1 = torch.as_tensor(list(range(64)), device=d2)
            return (
                torch.arange(10, device=d2).to(d1) + a,
                const1.to(d1),
                (const1 + 1).to(d1),
            )

        self.common(
            fn,
            (torch.randn([10]),),
        )

    @requires_gpu()
    def test_multi_device(self):
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=GPU_TYPE)
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            x = x.to(device=GPU_TYPE)
            x = x + 7
            x = x + 8
            x = x.cpu()
            x = x + 9
            x = x + 10
            return x

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
            check_lowp=False,  # cpu doesn't understand fp16, and there are explicit .cpu() calls
        )

    @skipIfRocm
    @requires_multigpu()
    def test_multi_gpu_device(self):
        # TODO: https://github.com/pytorch/pytorch/issues/92627
        x = torch.rand([4], device=GPU_TYPE)

        def fn(x, y):
            r = torch.ops.aten.div(x, y)
            r = r.to(f"{GPU_TYPE}:1")
            return 2 * r

        self.common(fn, (torch.randn(4), torch.randn(4)), check_lowp=False)

    @requires_multigpu()
    def test_multi_gpu_recompile_on_index(self):
        torch.set_float32_matmul_precision("high")

        def gemm(x, y):
            return x @ y

        failed_guard = None

        def fail(guard):
            nonlocal failed_guard
            failed_guard = guard

        gemm_opt = torch._dynamo.optimize("inductor", guard_fail_fn=fail)(gemm)

        x0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")
        y0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")

        gemm_opt(x0, y0)

        x1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        gemm_opt(x1, y1)
        self.assertTrue(failed_guard is not None)
        self.assertTrue(
            "tensor 'L['x']' Tensor device index mismatch. Expected device index to be"
            in failed_guard.reason
        )

    def test_unbind(self):
        def fn(a):
            return torch.unbind(a), torch.unbind(a, -1)

        self.common(
            fn,
            (torch.randn([4, 4, 4]),),
        )

    def test_convolution1(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            # Mismatched elements: 10 / 2352 (0.4%)
            # Greatest absolute difference: 5.7220458984375e-05 at index (0, 3, 12, 12) (up to 1e-05 allowed)
            # Greatest relative difference: 0.06512477175897748 at index (0, 4, 11, 9) (up to 0.001 allowed)
            atol=6e-5,
            rtol=0.001,
        )

    def test_convolution2(self):
        def fn(x, w, b):
            # transposed conv
            return (aten.convolution(x, w, b, [4], [0], [1], True, [0], 1),)

        self.common(
            fn,
            (
                torch.randn([2, 32, 90]),
                torch.randn([32, 16, 8]),
                torch.randn([16]),
            ),
            check_lowp=False,
        )

    def test_convolution3(self):
        # Test stride or padding or dilation is 1 element list.
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3], stride=[1], padding=[0], dilation=[1]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            atol=6e-5,
            rtol=0.001,
        )

    @skip_if_gpu_halide
    def test_convolution4(self):
        def fn(x, w):
            x = F.conv2d(x, w, groups=w.shape[0])
            return x.sum()

        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 20]),
                torch.randn([3, 1, 5, 5]),
            ),
        )

    def test_convolution5(self):
        def fn(x, w):
            x = F.conv2d(x, w, dilation=[x.size(0)])
            return x.sum()

        x = torch.randn([2, 1, 16, 20])
        w = torch.randn([1, 1, 5, 5])

        torch._dynamo.mark_dynamic(x, 0)

        atol = None
        rtol = None
        if self.device == "xpu":
            # set to float32 default tolerance,
            # check_model_gpu with update rotl to 2e-3 for fp16.
            # fix issue #129974
            atol = 1e-05
            rtol = 1.3e-06
        self.common(fn, (x, w), atol=atol, rtol=rtol)

    def test_conv3d(self):
        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=7),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([1, 3, 8, 16, 32]),),
            atol=6e-5,
            rtol=0.001,
            # Make sure we compute also with fp16 in the reference. Otherwise,
            # the reference will compute with fp32 and cast back to fp16, which
            # causes numeric differences beyond tolerance.
            reference_in_float=False if torch.version.hip else True,
        )

    def test_conv2d_channels_last(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv2d channels_last")

        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        )
        # only weight is channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]),),
            check_lowp=False,
        )
        # only activation is channels_last
        self.common(
            m,
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )
        # activation and weight are all channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )

    def test_conv2d_backward_channels_last(self):
        def fn(grad_output, inp, weight):
            convolution_backward_8 = torch.ops.aten.convolution_backward.default(
                grad_output,
                inp,
                weight,
                [320],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            return convolution_backward_8

        # only weight is channels_last
        self.common(
            fn,
            (
                torch.randn([2, 320, 8, 8]),
                torch.randn([2, 2048, 8, 8]),
                torch.randn([320, 2048, 1, 1]).to(memory_format=torch.channels_last),
            ),
            check_lowp=False,
        )

    def test_conv3d_channels_last(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv3d channels_last")

        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, 1, 1),
            ToTuple(),
        )
        # only weight is channels_last
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]),),
        )
        # only activation is channels_last
        self.common(
            m,
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )
        # activation and weight are all channels_last
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )

    @skip_if_gpu_halide  # slow
    def test_adaptive_avg_pool2d1(self):
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
                x + 1, (2, 5)
            )

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
            check_lowp=False,
        )

        # lowering to avg_pool2d case
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),
        )

        # no-op case
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),
        )

    def test_adaptive_avg_pool2d2(self):
        # Big kernel size, use fallback
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (4, 4))

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    @skip_if_gpu_halide  # slow
    def test_adaptive_max_pool2d1(self):
        def fn(x):
            return aten.adaptive_max_pool2d(x, (6, 6))

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
            check_lowp=False,
        )

        # lowering to max_pool2d case
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),
        )

        # no-op case
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),
        )

    @skip_if_gpu_halide  # slow
    def test_adaptive_max_pool2d2(self):
        # Big kernel size, use fallback
        def fn(x):
            return aten.adaptive_max_pool2d(x, (4, 4))

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_fractional_max_pool2d1(self):
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (3, 3), (2, 2), samples)

        self.common(
            fn, (torch.randn(1, 4, 16, 16), torch.rand(1, 4, 2)), check_lowp=False
        )

    def test_fractional_max_pool2d2(self):
        # fallback for larger kernel size

        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (6, 5), (3, 3), samples)

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn(2, 4, 36, 36), torch.rand(2, 4, 2)),
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_fractional_max_pool2d3(self):
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (1, 1), (16, 16), samples)

        self.common(
            fn, (torch.randn(2, 4, 16, 16), torch.rand(2, 4, 2)), check_lowp=False
        )

    @config.patch(fallback_random=True)
    @skip_if_halide  # Can only unroll for loops over a constant extent
    def test_fractional_max_pool2d4(self):
        random.seed(1234)
        torch.manual_seed(1234)

        # check rectangular kernel/output size

        def fn(x):
            return torch.nn.functional.fractional_max_pool2d_with_indices(
                x, (4, 3), (3, 2)
            )

        self.common(fn, (torch.randn(1, 4, 16, 16),), check_lowp=False)

    def test_multi_threading(self):
        model = torch.nn.Linear(2, 3).eval()
        inp = torch.randn(4, 2)

        num_run = 3

        def run_weights_sharing_model(m, inp):
            with torch.no_grad():
                for i in range(num_run):
                    y = m(inp)

        numb_instance = 2
        threads = []
        compiled_m = torch.compile(model)
        for i in range(1, numb_instance + 1):
            thread = threading.Thread(
                target=run_weights_sharing_model, args=(compiled_m, inp)
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    @unittest.skipIf(config.is_fbcode(), "fbcode triton error, needs debugging")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_adaptive_avg_pool2d_low_prec(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.avgpool(x)
                return x

        mod = Model().to(self.device)
        for dtype in [torch.half, torch.bfloat16]:
            x = torch.randn(4, 3, 7, 7, device=self.device).to(dtype=dtype)
            opt_mod = torch.compile(mod)
            res = opt_mod(x)
            expected = mod(x)
            self.assertTrue(torch.allclose(res, expected))

    def test_buffer_copied_in_graph(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return (self.w1 * x * self.w2).sum() + self.buf.sum()

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(1, requires_grad=True, device=self.device)
        inp_test = torch.ones(1, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    @skip_if_gpu_halide
    def test_buffer_copied_in_graph_with_different_shapes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(4, 4))
                self.w = torch.nn.Parameter(
                    torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]])
                )

            def forward(self, x):
                self.buf.add_(1)
                return (self.w @ x).sum() + self.buf.sum()

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(2, 4, requires_grad=True, device=self.device)
        inp_test = torch.ones(2, 4, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    @skipIfNNModuleInlined("https://github.com/pytorch/pytorch/issues/128198")
    def test_buffer_batch_norm(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.BatchNorm1d(100)

            def forward(self, x):
                return self.m(x)

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(20, 100, requires_grad=True, device=self.device)
        inp_test = torch.ones(20, 100, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            # TODO: remove the + 1 after https://github.com/pytorch/pytorch/issues/120622 is fixed
            buffer._version + 1
            if k in ["m.running_mean", "m.running_var"]
            else buffer._version
            for k, buffer in model_for_eager.named_buffers()
        ]

        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    def test_adaptive_avg_pool_with_output_size_0(self):
        m1 = nn.AdaptiveAvgPool1d(0)
        self.common(m1, (torch.randn(1, 2),))
        m2 = nn.AdaptiveAvgPool2d(0)
        self.common(m2, (torch.randn(1, 2, 3),))

    def test_max_pool2d1(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d2(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d3(self):
        def fn(x):
            # with padding
            return (
                aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1]),
                aten.max_pool2d_with_indices(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    @skip_if_halide  # Can only unroll for loops over a constant extent
    def test_max_pool2d4(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [0, 0], [1, 1], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d5(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d6(self):
        # Too big kernel size, use fallback
        def fn(x):
            return aten.max_pool2d_with_indices(x, [13, 13], [])

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )
        assertGeneratedKernelCountEqual(self, 0)

    # From https://github.com/pytorch/pytorch/issues/94775
    def test_max_pool2d7(self):
        # ceil mode turns on
        def fn(x):
            return torch.nn.functional.max_pool2d(
                x, 1, stride=(2, 2), padding=0, ceil_mode=True
            )

        self.common(
            fn,
            (torch.randn([1, 1, 6, 7]),),
        )

    # From https://github.com/pytorch/pytorch/issues/93384
    def test_max_pool2d8(self):
        # dialtion is not 1, use fallback
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 2], [2, 1], [1, 1], [1, 2])

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn([2, 2, 3, 6]),),
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool2d1(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_avg_pool2d2(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_avg_pool2d3(self):
        def fn(x):
            return (
                aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1]),
                aten.avg_pool2d(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d4(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [0, 0], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    def test_avg_pool2d5(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], count_include_pad=False)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d6(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], divisor_override=3)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d7(self):
        # Large kernel size, use fallback
        def fn(x):
            return aten.avg_pool2d(x, [13, 13], [1, 1], [0, 0])

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (-torch.arange(1 * 24 * 24, dtype=torch.float32).view(1, 1, 24, 24),),
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool2d8(self):
        # https://github.com/pytorch/pytorch/issues/100987
        def fn(x):
            return aten.avg_pool2d(
                x, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

        self.common(
            fn,
            (torch.randn(1, 3, 6, 6),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    @skip_if_gpu_halide  # slow
    def test_alexnet_prefix(self):
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16, arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            relu = torch.ops.aten.relu(convolution)
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(
                relu, [3, 3], [2, 2]
            )
            getitem = max_pool2d_with_indices[0]
            return (getitem,)

        self.common(
            forward,
            (
                rand_strided((64,), (1,), torch.float32, "cpu"),
                rand_strided((64, 3, 11, 11), (363, 121, 11, 1), torch.float32, "cpu"),
                rand_strided(
                    (16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, "cpu"
                ),
            ),
            # Mismatched elements: 127 / 746496 (0.0%)
            # Greatest absolute difference: 0.0009765625 at index (1, 62, 7, 16) (up to 1e-05 allowed)
            # Greatest relative difference: 0.05187467899332306 at index (14, 18, 11, 0) (up to 0.001 allowed)
            atol=3e-3,
            rtol=2,
        )

    def test_elu(self):
        def fn(x):
            return aten.elu(x, 1.6732632423543772, 1.0507009873554805) + 2, aten.elu(
                x + 1, 2, 3, 4
            )

        self.common(
            fn,
            (torch.randn([16, 16]),),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_tan(self):
        def fn(x):
            return aten.tan(x) + 2, aten.tan(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_tanh(self):
        def fn(x):
            return aten.tanh(x) + 2, aten.tanh(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    @skip_if_halide  # lgamma not implemented
    def test_lgamma(self):
        def fn(x):
            return aten.lgamma(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_cos(self):
        def fn(x):
            return aten.cos(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_sin(self):
        def fn(x):
            return aten.sin(x) + 2, aten.sin(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_repeat(self):
        def fn(x):
            return (
                x.repeat(0, 1, 1, 1),
                x.repeat(2, 2, 3, 1),
                x.repeat(8, 1, 1, 1),
                x.repeat(2, 1, 1, 1, 1, 1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_repeat_as_strided(self):
        # Reproducer for #127474

        def fn(x):
            view_size = (3, 2)
            full = x.repeat((3, 2))
            view = torch.as_strided(full, view_size, full.stride())
            result = view + view

            return result

        self.common(fn, (torch.randn(1, 1),))

    def test_repeat_interleave(self):
        def fn(x):
            return (
                x.repeat_interleave(2),
                x.repeat_interleave(3, dim=0),
                x.repeat_interleave(x.size(1), dim=1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave_2(self):
        def fn(x):
            return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        self.common(
            fn,
            (torch.tensor([2, 4, 6]),),
        )

    @config.patch(fallback_random=True)
    def test_randn_with_dtype_and_device(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu randn_with_dtype_and_device test")

        def fn(vectors):
            rotations_shape = (12, vectors.shape[-1], 1, 64)
            random_rotations = torch.randn(
                rotations_shape, device=vectors.device, dtype=vectors.dtype
            )
            random_rotations += 1
            return random_rotations

        self.common(
            fn,
            (torch.randn([4, 12, 2, 64]),),
        )

    def test_embedding(self):
        m = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, padding_idx=0),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randint(10, [2, 8]),),
        )

    def test_mean(self):
        def fn(x):
            return (
                x.mean(),
                x.mean(-1),
                torch.mean(x, -2, keepdim=True),
                x.mean([0, 1]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_var_mean(self):
        def fn(x):
            return (
                *torch.var_mean(x, -1),
                *torch.var_mean(x, [1, 3]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_var_correction(self):
        def fn(x):
            dim = -1
            return (
                torch.var(x, dim=dim, correction=1.3),
                torch.var(x, dim=dim, correction=3),
                torch.var(x, dim=dim, correction=10),
            )

        self.common(fn, (torch.randn([2, 8]),))
        # Unrolled reduction
        self.common(fn, (torch.randn([2, 4]),))

    @config.patch(pick_loop_orders=True)
    @skipIfWindows
    def test_transposed_propagates(self):
        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(x, y):
            return x + y

        a = torch.randn(1, 4, 4, 4, device=self.device).permute(0, 2, 3, 1)
        b = torch.randn(4, 4, 4, device=self.device).permute(1, 2, 0)
        c = fn(a, b)
        self.assertEqual(a.stride(), c.stride())
        self.assertEqual(c.stride()[2], 1)

    @skip_if_gpu_halide
    def test_std(self):
        def fn(x):
            return (
                torch.var(x, True),
                torch.var(x, False),
                torch.var(x, -1, True),
                torch.var(x, -1, False),
                torch.std(x, False),
                torch.std(x, [0, 1], True),
                torch.std(x, [0, 1], False),
                torch.std(x, -2, True, keepdim=True),
            )

        self.common(
            fn,
            (torch.randn([2, 4, 4, 8]),),
        )

    def test_embedding_bag(self):
        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None)

        self.common(
            fn,
            (torch.randn([10, 4]), torch.randint(10, [8]), torch.tensor([0, 2, 6])),
        )

    def test_batch_norm_2d(self):
        m = torch.nn.Sequential(
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
        )
        m.eval()
        self.common(m, (torch.randn([2, 10, 8, 8]),), check_lowp=False)
        self.common(
            m,
            (torch.randn([3, 10, 16, 16]),),
            check_lowp=False,  # too painful to match types of bn model
        )

    # From yolov3
    @with_tf32_off
    def test_batch_norm_2d_2(self):
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_0 = torch.nn.Conv2d(
                    64,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                self.self_1 = torch.nn.BatchNorm2d(
                    128,
                    eps=0.0001,
                    momentum=0.03,
                    affine=True,
                    track_running_stats=True,
                )
                self.self_2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

            def forward(self, l_input_: torch.Tensor):
                self_0 = self.self_0(l_input_)
                self_1 = self.self_1(self_0)
                self_2 = self.self_2(self_1)
                return (self_2,)

        inp = torch.randn((4, 64, 192, 256), dtype=torch.float32, device=GPU_TYPE)
        mod = Repro().to(device=GPU_TYPE)
        o1 = mod(inp)
        o2 = torch.compile(mod)(inp)
        self.assertEqual(o1, o2, rtol=1e-3, atol=1e-3)

    @patch.object(config.trace, "enabled", True)
    def test_layer_norm(self):
        m = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
        )
        m.eval()
        with torch.no_grad():
            self.common(m, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    def test_transpose_add(self):
        def fn(a, b):
            return a.t() + b

        self.common(
            fn, (torch.randn([16, 32]), torch.randn([32, 16])), check_lowp=False
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @patch.object(config.triton, "persistent_reductions", True)
    def test_softmax_one_kernel_persist(self):
        def fn(x):
            dim = 1
            x_max = torch.amax(x, dim, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
            return result

        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @patch.object(config.triton, "persistent_reductions", False)
    def test_softmax_one_kernel_loop(self):
        def fn(x):
            x_max = torch.amax(x, 1, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, 1, keepdim=True)
            return result

        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    def test_complex_fallback(self):
        def fn(x):
            return x * x + 10

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]).to(dtype=torch.complex64),),
        )
        assertGeneratedKernelCountEqual(self, 0)

        class ToComplex(nn.Module):
            def forward(self, x):
                return (x + x + 12).to(torch.complex64)

        self.common(ToComplex(), (torch.rand([1, 2, 4, 8]),), check_lowp=False)

        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    def test_view_as_complex(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, view_2):
                clone = torch.ops.aten.clone.default(
                    view_2, memory_format=torch.contiguous_format
                )
                view_2 = None
                view_as_complex = torch.ops.aten.view_as_complex.default(clone)
                clone = None
                return (view_as_complex,)

        inp = torch.empty_strided((128, 64, 12, 32, 2), (1, 98304, 8192, 256, 128)).to(
            self.device
        )
        mod = Repro()

        o1 = mod(inp)
        o2 = torch.compile(mod)(inp)

        self.assertEqual(o1, o2)

    def test_view_as_real(self):
        def fn(x):
            y = torch.view_as_real(x)
            return y + 1

        x = torch.randn(4, dtype=torch.complex64)

        self.common(fn, (x,))

    def test_polar(self):
        def fn(dist, angle):
            return torch.polar(dist, angle)

        inp = (
            torch.tensor([1, 2], dtype=torch.float64),
            torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64),
        )
        self.common(fn, (*inp,))

    @skip_if_gpu_halide  # incorrect result on CUDA
    def test_cauchy(self):
        def fn(x, y):
            return torch.sum(1 / (torch.unsqueeze(x, -1) - y))

        self.common(
            fn,
            (
                torch.randn(32),
                torch.randn(32),
            ),
            # Absolute difference: 0.0003662109375 (up to 0.0001 allowed)
            # Relative difference: 1.8804297408767818e-05 (up to 1e-05 allowed)
            atol=5 * 1e-4,
            rtol=5 * 1e-5,
            check_lowp=False,
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @skip_if_gpu_halide  # misaligned address error
    def test_fusing_write_into_disjoint_read(self):
        def test_flip(a):
            return a.copy_(torch.flip(a, (0,)))

        self.common(test_flip, (torch.rand([20]),))

        assertGeneratedKernelCountEqual(self, 2)

        # issue only manifests on cuda with large tensors
        if self.device != "cpu":

            def f(a):
                a[:, 20:40] = a[:, 20:40] + 1
                a[:, 2:900025] = a[:, 1:900024] + 2

            a = torch.rand((1, 1000000), device=self.device)
            self.common(f, (a,))

    def test_gather_scatter(self):
        def fn(node_feat, edge_index):
            src_node_feat = node_feat[edge_index[0]]
            dst_node_feat = node_feat[edge_index[1]]
            edge_feat = src_node_feat - dst_node_feat + 1
            new_node_feat = torch.zeros_like(node_feat)
            new_node_feat.scatter_add_(
                0, edge_index[1].unsqueeze(-1).expand_as(edge_feat), edge_feat
            )
            return new_node_feat

        num_nodes = 16
        num_features = 32
        node_feat = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, size=(2, num_nodes * 5))
        self.common(
            fn,
            (
                node_feat,
                edge_index,
            ),
            check_lowp=False,
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 2)

    @config.patch(max_fusion_size=1)
    def test_no_mega_fusion_during_lowering(self):
        n = 50

        def fn(*args):
            x = args[0]
            for i in range(n):
                x = torch.add(x, args[i])
            return x

        self.common(
            fn,
            [torch.randn(64) for _ in range(n)],
            check_lowp=False,
        )
        print("-->", torch._inductor.metrics.generated_kernel_count)
        if self.device != "cpu":
            self.assertTrue(torch._inductor.metrics.generated_kernel_count > 1)

    def test_move_arange(self):
        def fn(x):
            return torch.arange(len(x), device="cpu").to(x.device) + x

        self.common(fn, (torch.randn([32]),), check_lowp=False)
        # if we have a copy there will be more than 1 kernel
        assertGeneratedKernelCountEqual(self, 1)

    def test_leaky_relu(self):
        def fn(x):
            return aten.leaky_relu(x, 0.2) + 2, aten.leaky_relu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_gelu(self):
        def fn(x):
            return aten.gelu(x) + 2, aten.gelu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_clone(self):
        def fn(x):
            return aten.clone(x) + 2, aten.clone(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_masked_fill(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, -10000.0) + 2, aten.masked_fill(
                value / 2.0, torch.logical_not(mask), 667
            )

        self.common(
            fn,
            (
                torch.randint(0, 1, [1, 16], dtype=torch.bool),
                torch.randn([16, 16]),
            ),
        )

    @skipIfWindows
    def test_masked_fill_promotion(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, torch.tensor(3.5))

        opt_fn = torch._dynamo.optimize("inductor")(fn)
        for inp in (
            torch.randn(
                [16, 16],
                dtype=torch.float16 if self.device == GPU_TYPE else torch.float32,
                device=self.device,
            ),
            torch.randint(16, (16, 16), device=self.device),
        ):
            inputs = (
                torch.randint(0, 1, [1, 16], dtype=torch.bool, device=self.device),
                inp,
            )
            self.assertEqual(fn(*inputs), opt_fn(*inputs))

    def test_masked_scatter(self):
        def fn(value, mask, source):
            return torch.masked_scatter(value, mask, source)

        value = make_tensor(10, 10, dtype=torch.float32, device=self.device)
        mask = make_tensor(10, 10, dtype=torch.bool, device=self.device)
        source = make_tensor(
            mask.count_nonzero(), dtype=torch.float32, device=self.device
        )

        self.common(fn, (value, mask, source))

    def test_fill1(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Scalar(tmp, 2)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_fill2(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Tensor(tmp, torch.tensor(3.0))

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow1(self):
        def fn(x):
            return [aten.pow(x, e) for e in range(-8, 9)]

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow2(self):
        def fn(x):
            return aten.pow(1000, x), aten.pow(x, 1000)

        self.common(
            fn,
            (
                torch.randn(
                    [16, 16],
                    dtype=torch.float32,
                ),
            ),
            # Mismatched elements: 9 / 256 (3.5%)
            # Greatest absolute difference: 2.491354329061828e+28 at index (6, 6) (up to 1e-05 allowed)
            # Greatest relative difference: 2.9793410720160818e-05 at index (4, 5) (up to 1.3e-06 allowed)
            atol=1e-5,
            rtol=3e-05,
        )

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8318
    @config.patch("halide.scheduler_cuda", "Li2018")
    @skipIfWindows
    def test_pow3(self):
        # power of 0.5 is special-cased, arbitrary power would still produce triton codegen error
        def fn(x):
            z = torch.tensor(0.123, device=self.device)
            w = z + x
            return torch.pow(w, 0.5)

        opt = torch._dynamo.optimize("inductor")(fn)
        input = torch.rand((), device=self.device)
        self.assertTrue(same(opt(input), fn(input)))

    def test_pow_int(self):
        def fn(x, y):
            return torch.pow(x, 0x57), torch.pow(x, y)

        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            intmax = torch.iinfo(dtype).max
            make_arg = functools.partial(
                make_tensor, dtype=dtype, device=self.device, requires_grad=False
            )
            self.common(
                fn,
                (
                    make_arg(16, 16),
                    make_arg(16, 16, high=intmax),
                ),
            )

    def test_glu(self):
        def fn(x):
            return aten.glu(x, -1), aten.glu(x, 1), aten.glu(x, 2)

        self.common(
            fn,
            (torch.randn([8, 16, 8, 8]),),
        )

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_unbacked_refinement(self):
        def fn(x):
            z = x.nonzero()
            torch._check(z.size(0) == 4)
            return z + 3

        self.common(
            fn,
            (torch.tensor([0, 1, 3, 4, 2, 0, 0]),),
        )

        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.tensor([0, 0, 0, 0]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify(self):
        def fn(x, y):
            z = y.item()
            torch._check(z // 2 == 3)
            return x + x.new_ones(z)

        self.common(
            fn,
            (
                torch.randn(6),
                torch.tensor([6]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn(7),
                torch.tensor([7]),
            ),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify_errors(self):
        def fn(x, y):
            z = y.item()
            torch._check(z // 2 == 3)
            return x + x.new_zeros(z)

        # This is a little suboptimal: we actually fail /in the compiler/ but
        # not in a way that causes Dynamo to graph break
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.randn(8), torch.tensor(8))

    def test_cat(self):
        def fn(a):
            tmp = a * 2
            return (
                torch.cat((a, a[:, :4] + 1, a + 2), -1),
                torch.cat((tmp, tmp), 0),
                torch.cat((tmp, tmp.double()), 0),
            )

        self.common(
            fn,
            (torch.randn([8, 16]),),
        )
        self.common(
            fn,
            (torch.randn([1, 3, 3, 16]).to(memory_format=torch.channels_last),),
        )

    def test_cat_uint8(self):
        def fn(x):
            batch_shape = x.shape[:1]
            out = torch.cat([x.new_zeros(1).expand(batch_shape + (1,)), x], dim=-1)
            return out

        self.common(
            fn,
            (torch.randint(0, 256, size=(3, 255), dtype=torch.uint8),),
        )

    def test_cat_empty(self):
        def fn_2(*tensors):
            return torch.cat(tensors)

        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),
                torch.ones([0]),
            ),
        )
        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),
                torch.ones([0]),
                torch.randn([1, 3, 3, 16]),
            ),
        )
        self.common(
            fn_2,
            (
                torch.ones([0]),
                torch.randn([1, 3, 3, 16]),
            ),
        )

    def test_cat_empty_index(self):
        def fn(out, x):
            return torch.cat([out[0], x], dim=0)

        self.common(fn, (torch.randn(1, 0, 64), torch.randn(128, 64)))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_legacy_empty(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z)])

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.tensor([0]),
            ),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_empty_1d(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z)])

        self.common(
            fn,
            (
                torch.randn([2]),
                torch.tensor([0]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2]),
                torch.tensor([3]),
            ),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_2d(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z, x.shape[1])])

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.tensor([0]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.tensor([4]),
            ),
        )

    def test_cat_negative_dim(self):
        def fn(*tensors):
            return torch.cat(tensors, dim=-1)

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.randn([2, 4]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.randn([0]),
                torch.randn([2, 4]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([0]),
                torch.randn([2, 3]),
                torch.randn([2, 4]),
            ),
        )

    @expectedFailureCodegenDynamic
    def test_cat_single_empty(self):
        # fails dynamic check for 'has a dynamic dimension'
        def fn_2(*tensors):
            return torch.cat(tensors)

        self.common(
            fn_2,
            (torch.ones([0]),),
        )

    def test_cat_upcasting(self):
        def fn(arg4_1, slice_7):
            cat_1 = aten.cat.default([arg4_1, slice_7], 1)
            return (cat_1,)

        self.common(
            fn,
            (
                torch.randn([8, 16], dtype=torch.float32),
                torch.randn([8, 20], dtype=torch.float16),
            ),
        )

    def test_cat_extern_kernel(self):
        def fn(x1, x2, x3, x4):
            x = torch.mm(x2, x3)
            s = torch.narrow(x, 1, 0, 100)
            x = torch.mm(s, x4)
            c = torch.cat((x, x1), 1)
            return (c,)

        if self.device == "xpu":
            atol = 3e-4
            rtol = 1e-4
        else:
            # use default
            atol = None
            rtol = None
        self.common(
            fn,
            (
                torch.randn(256, 256),
                torch.randn(256, 1024),
                torch.randn(1024, 1600),
                torch.randn(100, 256),
            ),
            atol=atol,
            rtol=rtol,
            check_lowp=False,  # accuracy issues with relatively large matmuls
        )

    @skip_if_gpu_halide
    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    # Constant folding was explicitly turned off due to issue #108388
    # Turn it back on for test
    @torch._inductor.config.patch(joint_graph_constant_folding=True)
    def test_remove_no_ops(self):
        def matmul_with_op(x, y, fn):
            return fn(x @ y)

        foo_opt = torch.compile(matmul_with_op)

        # test no-op
        fns = (
            lambda x: x
            + torch.zeros(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            - torch.zeros(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            * torch.ones(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
            lambda x: x
            / torch.ones(
                [256, 256], dtype=torch.float32, device=x.device
            ),  # noqa: E731
        )

        inps = [torch.rand([256, 256], device=self.device) for _ in range(2)]

        for fn in fns:
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))

            if self.device == "cpu":
                FileCheck().check_not("cpp_fused").run(source_codes[0])
            else:
                FileCheck().check_not("triton.jit").run(source_codes[0])

        # test dtype conversion
        inps = [
            torch.rand([256, 256], device=self.device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        for fn in fns:
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))

        # test broadcasted shape bail
        fn = lambda x: x + torch.zeros(  # noqa: E731
            [256, 256, 256], dtype=torch.bfloat16, device=self.device
        )
        out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
        self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))

    def test_remove_noop_copy(self):
        def fn(x, y):
            x = x.cos()
            a = x.copy_(y)
            return a.sin()

        self.common(fn, (torch.randn(8, 8), torch.randn(8)))

        def fn2(a, b):
            abs_max = torch.abs(a).max()
            b[0] = abs_max.to(a.dtype)
            return b

        self.common(
            fn2,
            (
                torch.randn(8, 8, dtype=torch.float16),
                torch.randn(8, dtype=torch.float32),
            ),
        )

    def test_remove_noop_clone(self):
        def fn(x):
            y = x.clone().reshape(-1, 4)
            y[:, [2, 0]] = y[:, [0, 2]]
            return y + x

        self.common(fn, (torch.randn(2, 4),))

    @skipIfWindows
    def test_cat_of_loops_and_extern_kernel(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    64,
                    5,
                    1,
                    **kwargs,
                )
                self.max_pool2d = torch.nn.MaxPool2d(2)

            def forward(self, x, y):
                x1 = self.conv(x)
                y1 = self.max_pool2d(y)
                return torch.cat([x1, y1], 1)

        mod = M()
        opt_mod = torch._dynamo.optimize("inductor")(mod)
        memory_format = torch.channels_last
        inputs = (
            torch.randn([1, 64, 16, 16]).to(memory_format=memory_format),
            torch.randn([1, 64, 32, 32]).to(memory_format=memory_format),
        )
        y = mod(*inputs)
        opt_y = opt_mod(*inputs)
        self.assertEqual(y, opt_y)
        self.assertEqual(y.stride(), opt_y.stride())

    def test_cat_inplace(self):
        def fn(x):
            rt = torch.cat([x])
            v = x.sin_()
            return rt

        # can't use self.common because input is modified inplace
        inp = torch.ones(2)
        opt_fn = torch.compile(fn)
        res = opt_fn(inp.clone())
        expected = fn(inp.clone())
        self.assertEqual(res, expected)

    def test_stack(self):
        def fn(a, b):
            return torch.stack(
                [
                    a.expand(12, 16),
                    b.expand(12, 16),
                ],
                2,
            )

        self.common(fn, (torch.randn([1, 16]), torch.randn([12, 1])))

    def test_hardtanh(self):
        def fn(x):
            return F.hardtanh(x), F.hardtanh(x + 1), F.hardtanh(x - 1)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardsigmoid(self):
        def fn(x):
            return F.hardsigmoid(x), F.hardsigmoid(x + 3), F.hardsigmoid(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardswish(self):
        def fn(x):
            return F.hardswish(x), F.hardswish(x + 3), F.hardswish(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_rsqrt(self):
        def fn(x):
            return torch.rsqrt(x), torch.rsqrt(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_expm1(self):
        def fn(x):
            return torch.expm1(x), torch.expm1(x) * 2

        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )

    def test_log1p(self):
        def fn(x):
            return torch.log1p(x), torch.log1p(x) * 2

        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )

    def test_flip(self):
        def fn(x):
            return torch.flip(x, (-1,)), torch.flip(x, (0, 2)) - 2

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_signbit(self):
        def fn(x):
            return torch.signbit(x), ~torch.signbit(-x) & 1

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_sign_dtype(self):
        def fn(x):
            y = torch.sign(x)
            return torch.tanh(y)

        self.common(fn, (torch.randn([1, 2, 6, 6]),))

    def test_fmod(self):
        def fn(a, b):
            return torch.fmod(a, b), torch.fmod(3.0 * a, b) - 2.0

        shape = [1, 2, 6, 6]
        self.common(fn, (torch.randn(shape), torch.randn(shape)))

    def test_fmod_zero_dim(self):
        def fn(a, b):
            return (torch.fmod(a, b),)

        self.common(
            fn,
            (
                make_tensor(10, device=self.device, dtype=torch.float32),
                make_tensor((), device=self.device, dtype=torch.float32),
            ),
        )
        self.common(
            fn,
            (
                make_tensor((), device=self.device, dtype=torch.float32),
                make_tensor(10, device=self.device, dtype=torch.float32),
            ),
        )

    def test_log2(self):
        def fn(x):
            return torch.log2(x), torch.log2(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]) + 10,),
        )

    def test_logsumexp(self):
        def fn(x):
            return torch.logsumexp(x, -1), torch.logsumexp(x, 0) - 2

        self.common(
            fn,
            (torch.randn([8, 8]) + 10,),
        )

    def test_log_fp64(self):
        def fn(x):
            return torch.log(x), torch.log2(x)

        self.common(
            fn,
            (torch.randn([1024], dtype=torch.float64) + 10,),
        )

    def test_bitwise(self):
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2**30, [64], dtype=torch.int32),
                torch.randint(0, 2**30, [64], dtype=torch.int32),
            ),
        )

    def test_bitwise2(self):
        # again with bool types
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
            ),
        )

    def test_bitwise3(self):
        # Repro for https://github.com/pytorch/pytorch/issues/97968
        def fn(x, y):
            return (
                torch.max(torch.bitwise_and(x, y), y),
                torch.clamp_max(torch.bitwise_or(x, y), y),
                torch.clamp_min(torch.bitwise_xor(x, y), y),
            )

        self.common(
            fn,
            (
                torch.rand([5, 10, 1]).to(torch.int8),
                torch.rand([10, 1]).to(torch.int8),
            ),
        )

    def test_inf(self):
        def fn(a):
            return a + float("inf"), a + float("-inf"), a * -float("inf")

        self.common(fn, (torch.randn(8),))

    def test_remainder(self):
        def fn(a, b):
            return (
                torch.remainder(a, b),
                torch.remainder(a + 1, b - 1),
                torch.remainder(a - 1, b + 1),
            )

        self.common(fn, (torch.randn(64), torch.randn(64)))

    def test_zeros(self):
        def fn(a):
            return (
                a + 1,
                torch.zeros(
                    (1, 8, 64, 64),
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(
                    1,
                    8,
                    64,
                    64,
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(2, 3),
                a + torch.ones(8, device=a.device),
                torch.full((2, 3), 3.1416, device=a.device),
            )

        self.common(fn, (torch.randn(8),))

    def test_new_ones(self):
        def fn(a):
            return (
                aten.new_ones(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
                aten.new_zeros(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
            )

        self.common(fn, (torch.randn(8),))

    def test_full_like(self):
        def fn(a):
            return torch.full_like(a, 7.777) - 1

        self.common(fn, (torch.randn(8),))

    def test_full_truncation(self):
        def fn(a):
            return a + torch.full_like(a, 7.777)

        for dtype in all_types():
            self.common(fn, (make_tensor(8, dtype=dtype, device=self.device),))

    def test_full_boolean(self):
        def fn(n):
            x = torch.full((1,), n >= 1024, device=self.device)
            return x, x + 1

        self.common(fn, (1024,))
        self.common(fn, (1023,))

    def test_index1(self):
        def fn(a, b, c):
            return aten.index(a, [b, c])

        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([0, 0, 2, 2], dtype=torch.int64),
                torch.tensor([3, 4, 4, 3], dtype=torch.int64),
            ),
        )
        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
                torch.tensor([[3], [4], [4], [3]], dtype=torch.int64),
            ),
        )

    def test_index2(self):
        def fn(a, b):
            return (
                aten.index(a, [b]),
                aten.index(a, [None, b]),
            )

        self.common(
            fn,
            (
                torch.randn(8, 8, 8),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
            ),
        )

    def test_index3(self):
        def fn(x, ia, ib):
            return (x[:, ia, None, ib, 0],)

        self.common(
            fn,
            (
                torch.randn(3, 4, 4, 4, 3),
                torch.tensor([0, 2, 1], dtype=torch.int64),
                torch.tensor([0, 2, 1], dtype=torch.int64),
            ),
        )

    @skipIfWindows
    def test_output_strides(self):
        def fn(x):
            y = x.permute(0, 2, 3, 1).contiguous()
            torch._dynamo.graph_break()
            return y.view(-1, 4)

        inp = torch.rand([4, 4, 4, 4], device=self.device)
        fn_opt = torch._dynamo.optimize("inductor")(fn)

        self.assertEqual(fn(inp), fn_opt(inp))
        self.assertEqual(fn(inp).stride(), fn_opt(inp).stride())

        # no redundant copy
        def foo(x):
            return x[0:2:2].T[3:].squeeze(0)

        foo_opt = torch._dynamo.optimize("inductor")(foo)
        out = foo_opt(inp)
        self.assertEqual(inp.storage(), out.storage())

    def test_index_select(self):
        def fn(a, b):
            return (
                torch.index_select(a, 0, b),
                torch.index_select(a, 1, b),
                torch.index_select(torch.index_select(a, 2, b), 1, b),
            )

        for ind_dtype in (torch.int32, torch.int64):
            self.common(
                fn,
                (
                    torch.randn(8, 8, 8),
                    torch.tensor([0, 0, 2, 1], dtype=ind_dtype),
                ),
            )

    @skipCUDAIf(not TEST_CUDNN, "CUDNN not available")
    @skipIfXpu
    @skipIfRocm
    def test_cudnn_rnn(self):
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]
            return aten._cudnn_rnn(
                a0,
                a1,
                4,
                a3,
                a4,
                a5,
                2,
                2048,
                0,
                2,
                False,
                0.0,
                False,
                True,
                [],
                None,
            )

        self.common(
            fn,
            (
                torch.randn([92, 8, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([167837696]),
                torch.randn([4, 8, 2048]),
                torch.randn([4, 8, 2048]),
            ),
            check_lowp=False,  # difference in rnn is too large between half and float inputs
        )

    def test_upsample_nearest1d(self):
        def fn(a):
            return (
                aten.upsample_nearest1d(a, [74], None),
                aten.upsample_nearest1d(a, [70], None),
                aten.upsample_nearest1d(a, [45], None),
                aten.upsample_nearest1d(a, [36], None),
                aten.upsample_nearest1d(a, None, [2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37]),))

    def test_upsample_nearest2d(self):
        def fn(a):
            return (
                aten.upsample_nearest2d(a, [74, 76]),
                aten.upsample_nearest2d(a, [70, 75]),
                aten.upsample_nearest2d(a, [45, 74]),
                aten.upsample_nearest2d(a, [36, 39]),
                aten.upsample_nearest2d(a, None, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),))

    def test_upsample_nearest3d(self):
        def fn(a):
            return (
                aten.upsample_nearest3d(a, [74, 76, 78], None),
                aten.upsample_nearest3d(a, [70, 75, 80], None),
                aten.upsample_nearest3d(a, [45, 74, 103], None),
                aten.upsample_nearest3d(a, [36, 39, 40], None),
                aten.upsample_nearest3d(a, None, [2.0, 2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38, 39]),))

    def test_upsample_nearest2d_backward(self):
        func = torch.ops.aten.upsample_nearest2d_backward

        def fn(a):
            return (
                func(a, output_size=[6, 12], input_size=[3, 3, 3, 6]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 5]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 7]),
            )

        self.common(fn, (torch.randn([3, 3, 6, 12]),))

    @skip_if_x86_mac()
    def test_upsample_bilinear2d_a(self):
        def fn(a):
            return (
                aten.upsample_bilinear2d(a, [45, 45], False, None),
                aten.upsample_bilinear2d(a, None, True, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),), atol=2.5e-5, rtol=1.3e-6)

    def test_upsample_bilinear2d_b(self):
        def fn(a):
            return aten.upsample_bilinear2d(a, None, True, [2.0, 2.0])

        self.common(
            fn,
            [
                torch.randn([1, 2, 40, 59]),
            ],
            atol=2.5e-5,
            rtol=1.3e-6,
        )

    @skip_if_gpu_halide  # accuracy issue
    def test_reflection_pad2d(self):
        def fn(a, pad):
            return (
                aten.reflection_pad2d(a, [1, 1, 1, 1]),
                aten.reflection_pad2d(a, pad),
            )

        self.common(
            fn,
            (
                torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),
                [5, 2, 3, 4],
            ),
        )

    def test_reflection_pad2d_backward(self):
        def template(size, padding):
            def fn(grad_output, x):
                return aten.reflection_pad2d_backward(grad_output, x, padding)

            x = torch.randint(0, 999, size=size, dtype=torch.float32)
            result = aten.reflection_pad2d(x, padding)
            grad_output = torch.randn_like(result)

            self.common(
                fn, (grad_output, x), check_lowp=not is_halide_backend(self.device)
            )

        template([1, 1, 8, 8], [0, 0, 0, 0])
        template([1, 1, 8, 8], [1, 1, 1, 1])
        template([1, 1, 8, 8], [1, 2, 3, 4])
        template([1, 1, 8, 8], [0, -1, 2, 2])
        template([1, 1, 8, 8], [-1, 0, 2, 2])
        template([1, 1, 8, 8], [2, 2, 0, -1])
        template([1, 1, 8, 8], [2, 2, -1, 0])

    def test_grid_sampler_2d(self):
        def fn(a, b):
            return (
                aten.grid_sampler_2d(a, b, 0, 0, True),
                aten.grid_sampler_2d(a, b, 0, 1, False),
            )

        self.common(
            fn,
            (
                torch.randn([4, 3, 352, 352], dtype=torch.float32),
                torch.rand([4, 352, 352, 2], dtype=torch.float32) * 2 - 1,
            ),
            check_lowp=False,
            # Mismatched elements: 154697 / 1486848 (10.4%)
            # Greatest absolute difference: 0.0001976490020751953 at index (0, 0, 101, 243) (up to 1e-05 allowed)
            # Greatest relative difference: 7.332530120481928 at index (1, 1, 258, 301) (up to 1.3e-06 allowed)
            atol=0.0002,
            rtol=1.3e-06,
        )

    def test_upsample_bicubic2d(self):
        def fn(a):
            return (
                aten.upsample_bicubic2d(a, (128, 128), True),
                aten.upsample_bicubic2d(a, (128, 256), False),
            )

        # Mismatched elements: 10 / 196608 (0.0%)
        # Greatest absolute difference: 1.3869255781173706e-05 at index (2, 1, 88, 65) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0033082996811011046 at index (3, 1, 88, 91) (up to 1.3e-06 allowed)
        self.common(
            fn,
            (torch.randn([4, 3, 64, 32], dtype=torch.float32),),
            atol=2e-5,
            rtol=1e-3,
        )

    def test_float_index_expression(self):
        # Test that index propagation doesn't generate bad index_expr calls like
        # ops.index_expr(0.5*x, dtype) where the expression is not integral
        def fn(x):
            return aten.upsample_bicubic2d(x, (256, 256), False)

        x = torch.randn(1, 1, 128, 128, dtype=torch.float32, device=self.device)
        _, source_codes = run_and_get_code(fn, x)

        pattern = r"0\.50*\*[ix][\d]"
        for code in source_codes:
            self.assertIsNone(
                re.search(pattern, code), msg="Found bad index_expr in code:\n" + code
            )

    def test_float_index_expression_type_promotion(self):
        # Test that float indexing expressions participate in type promotion
        def fn(x):
            return x + 1.0 / x.size(0)

        x = torch.arange(10)
        self.common(fn, (x,))

    @skipIfWindows
    def test_sort(self):
        def fn(a, descending):
            return torch.sort(a)

        inp = torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    @skipIfWindows
    def test_sort_stable(self):
        def fn(a, descending):
            return a.sort(dim=-1, stable=True, descending=descending)

        # Duplicates give deterministic indices when stable sorting
        inp = torch.rand(10, 128, dtype=torch.float32)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

        # Non-power of two
        inp = inp[:, :120]
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    @skipIfWindows
    def test_sort_bool(self):
        def fn(a, descending):
            return torch.sort(a.to(torch.int8), stable=True, descending=descending)

        inp = torch.randint(0, 2, size=[10, 128], dtype=torch.bool)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    @skipIfWindows
    def test_sort_transpose(self):
        def fn(a, descending):
            return torch.sort(a, stable=True, descending=descending)

        inp = torch.randn(128, 10).transpose(0, 1)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    def test_topk(self):
        def fn(a):
            return torch.topk(a, 2, -1)

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_long_tensor(self):
        def fn(a):
            return (
                torch.LongTensor([294]).to(a.device) - a,
                torch.as_tensor([295]).to(a.device) + a,
            )

        self.common(fn, (torch.randint(0, 999, size=[8, 8]),))

    @skip_if_gpu_halide  # correctness issue
    def test_constant_pad_1d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [0, 1], 6.0),
                aten.constant_pad_nd(a, [2, 3], 99.0),
            )

        self.common(fn, (torch.randint(0, 999, size=[2, 16, 31], dtype=torch.float32),))

    def test_constant_pad_fill_dtype(self):
        def fn(a, b):
            return (
                aten.constant_pad_nd(a, (1, 1), 1.0) & b,
                aten.constant_pad_nd(a, (1, 1), 0.0) & b,
            )

        self.common(
            fn,
            (torch.randint(2, (4,), dtype=torch.bool), torch.ones(6, dtype=torch.bool)),
        )

    @skip_if_gpu_halide  # misaligned address
    def test_constant_pad_2d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 1, 1, 1], 6.0),
                aten.constant_pad_nd(a, [1, 2, 3, 4], 99.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    @skip_if_gpu_halide  # misaligned address
    def test_constant_pad_3d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 2, 3, 4, 5, 6], 6.0),
                aten.constant_pad_nd(a, [0, 0, 3, 4, 0, 0], 6.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[2, 4, 4, 4], dtype=torch.float32),)
        )

    def test_constant_pad_float64(self):
        # Repro for https://github.com/pytorch/pytorch/issues/93351
        def fn(input):
            v1 = torch.nn.functional.pad(input, pad=(1, 0))
            return torch.gt(v1, input)

        x = torch.rand([1, 2, 2, 1], dtype=torch.float64)
        self.common(fn, (x,))

    def test_constant_pad_nd_inplace(self):
        def fn(a):
            return aten.constant_pad_nd(a, [0, 0])

        x = torch.randn([2], device=self.device)
        fn_compiled = torch.compile(fn)
        y = fn_compiled(x)
        self.assertTrue(y is not x)

    def test_l1_loss(self):
        def fn(a, b):
            return torch.nn.functional.l1_loss(a, b), torch.nn.functional.mse_loss(a, b)

        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 16]),
                torch.randn([2, 3, 16, 16]),
            ),
            check_lowp=False,
        )

    def test_triu(self):
        def fn(a):
            return aten.triu(a, 1), aten.triu(a, 0), aten.triu(a, 2)

        self.common(fn, (torch.randn([2, 10, 10]),))

    def test_no_op_reduction(self):
        def fn(a):
            return a.sum(-1), torch.amax(a + 1, 1, keepdim=True)

        self.common(fn, (torch.randn([8, 1, 1]),))

    @skipIfWindows
    def test_inplace_add(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x.add_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device),
            rand_strided((4, 4), (4, 1), device=self.device),
        )
        inp_clone = inputs[0].clone()
        out = fn(*inputs)
        self.assertTrue(same(out, inp_clone + inputs[1]))
        self.assertTrue(out is inputs[0])

    # The following 2 tests are meant to check the logic that drops
    # xmask from triton load/store if xnumel = 1
    @requires_gpu()
    def test_single_elem(self):
        def fn(a):
            b = a + 1
            return (b,)

        self.common(fn, (torch.randn(1),))

    @requires_gpu()
    def test_single_elem_indirect(self):
        def fn(a, b):
            c = a[b] + 1
            return (c,)

        a = torch.randn(1)
        b = (torch.tensor([0], dtype=torch.int64),)

        self.common(fn, (a, b))

    # This test is meant to check for issues from the logic
    # that drops xmask from trito load/store if XBLOCK divides xnumel

    @requires_gpu()
    def test_xblock_divides_xnumel(self):
        def fn(a):
            b = a + 1
            return (b,)

        # assumption is that XBLOCK is always a divisor of 1024
        # so xmask will be dropped iff xnumel is multiple of 1024
        self.common(fn, (torch.randn(1024),))
        self.common(fn, (torch.randn(1025),))

    @skipIfWindows
    def test_inplace_mixed_dtype_ops(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            z = x + y.float()
            w = z.add_(y)
            return w.mul_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.float),
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.double),
        )
        out = fn(*inputs)
        out_eager = (inputs[0] + inputs[1].float()).add_(inputs[1]).mul_(inputs[1])
        self.assertTrue(same(out, out_eager))

    @config.patch(
        {"triton.unique_kernel_names": True, "triton.descriptive_names": False}
    )
    @skipIfWindows
    def test_kernel_names(self):
        @torch._dynamo.optimize("inductor")
        def fn(x):
            return 2 * x

        inputs = (rand_strided((8,), (1,), device=self.device),)
        self.assertTrue(same(fn(*inputs), 2 * inputs[0]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    @skipIfWindows
    def test_strided_inputs(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((8, 16), (32, 2), device=self.device),
            rand_strided((8, 16), (16, 1), device=self.device),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_input_mutation1(self):
        def fn(a):
            b = a + 1
            a.copy_(b)
            c = a + 2
            return a * b / c

        arg1 = torch.randn(64, device=self.device)
        arg2 = arg1.clone()
        arg3 = torch.randn(64, device=self.device)
        arg4 = arg3.clone()
        correct1 = fn(arg1)
        correct2 = fn(arg3)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)
        actual2 = opt_fn(arg4)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(actual2, correct2))
        self.assertTrue(same(arg1, arg2))
        self.assertTrue(same(arg3, arg4))

    def test_input_mutation2(self):
        def fn(a):
            b = a + 1
            a.view(64).copy_(torch.tensor([66.0], device=a.device))
            c = a + 2
            return b, c

        # NOTE: this test fails when none of the inputs require grad.
        # That seems like an inductor bug.
        arg1 = torch.randn([1, 64], device=self.device).requires_grad_(True).add(1)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation3(self):
        def fn(a):
            a += 1
            a *= 2
            aten.sigmoid_(a)
            a = a.view(64)
            a += 3
            a *= 4
            aten.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation4(self):
        def fn(a):
            torch.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    @skipIfWindows
    def test_input_mutation5(self):
        def fn(x):
            tmp = x.ceil()
            x.add_(10)
            return tmp

        opt_fn = torch._dynamo.optimize()(fn)

        a = torch.zeros((), dtype=torch.int64, device=self.device)
        a_expect = a.clone()
        expect = fn(a_expect)

        a_actual = a.clone()
        actual = opt_fn(a_actual)

        self.assertEqual(a_expect, a_actual)
        self.assertEqual(expect, actual)

    def test_slice_mutation1(self):
        def fn(a):
            x = torch.zeros_like(a)
            b = x + 1
            x[:, 3] = 3.0
            c = torch.clone(x)
            x[4, :] = 4.0
            d = x + 1
            return x, b, c, d

        self.common(fn, (torch.randn([8, 8]),))

    @skip_if_gpu_halide  # accuracy issue
    def test_slice_mutation2(self):
        def fn(a):
            a[:, 20:40] = a[:, 20:40] + 1
            a[:, 2:11] = a[:, 1:10] + 2

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        opt_fn(arg2)
        self.assertTrue(same(arg1, arg2))

    def test_slice_mutation3(self):
        def fn(a):
            a[:2, :2].fill_(10)

        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)

        x1 = torch.randn(8, 8, device=self.device)
        x2 = x1.clone()
        fn(x1)
        opt_fn(x2)
        self.assertEqual(x1, x2)

    def test_tensor_index_slice(self):
        def fn(a):
            x = torch.tensor([1, 2], device=self.device)
            y = torch.tensor([2, 3], device=self.device)
            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)
            return [
                a[x, y],
                a[:, x, y],
                a[:, x, y, :],
                a[x, :, y],
                a[:, x, :, y, :],
                a[xx, yy],
                a[:, xx, yy],
                a[xx, :, yy],
                a[xx, yy, :],
                a[:, xx, :, yy],
            ]

        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device).view(3, 4, 5, 6, 7)
        refs = fn(a)
        tests = torch.compile(fn)(a)
        for ref, test in zip(refs, tests):
            torch.testing.assert_close(ref, test)

    @torch._dynamo.config.patch(cache_size_limit=10)
    def test_tensor_index_put_slice(self):
        def fn(a, version):
            x = torch.tensor([1, 2], device=self.device, dtype=torch.int32)
            y = torch.tensor([2, 3], device=self.device, dtype=torch.int32)

            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)

            if version == 0:
                a[x, y] = torch.zeros_like(a[x, y])
            elif version == 1:
                a[:, x, y] = torch.zeros_like(a[:, x, y])
            elif version == 2:
                a[:, x, y, :] = torch.zeros_like(a[:, x, y, :])
            elif version == 3:
                a[x, :, y] = torch.zeros_like(a[x, :, y])
            elif version == 4:
                a[:, x, :, y, :] = torch.zeros_like(a[:, x, :, y, :])
            elif version == 5:
                a[xx, yy] = torch.zeros_like(a[xx, yy])
            elif version == 6:
                a[:, xx, yy] = torch.zeros_like(a[:, xx, yy])
            elif version == 7:
                a[xx, :, yy] = torch.zeros_like(a[xx, :, yy])
            elif version == 8:
                a[xx, yy, :] = torch.zeros_like(a[xx, yy, :])
            elif version == 9:
                a[:, xx, :, yy] = torch.zeros_like(a[:, xx, :, yy])

            return a

        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device, dtype=torch.int32).view(
            3, 4, 5, 6, 7
        )
        for i in range(10):
            ref = fn(torch.clone(a), i)
            test = torch.compile(fn)(torch.clone(a), i)
            torch.testing.assert_close(ref, test)

    def test_indirect_load_broadcast(self):
        def fn(in_ptr0, in_ptr1, in_ptr2):
            return torch.gather(in_ptr1, 0, in_ptr2) + in_ptr0

        arg190 = rand_strided((32, 21), (1, 32), device=self.device, dtype=torch.int64)
        arg190.fill_(0)
        arg111 = rand_strided(
            (9521, 512), (512, 1), device=self.device, dtype=torch.float32
        )
        self.common(
            fn,
            (
                torch.randn(32, 1),
                arg111,
                arg190,
            ),
        )

    def test_roi_align(self):
        if not has_torchvision_roi_align():
            raise unittest.SkipTest("requires torchvision")

        def fn(a, b):
            return torch.ops.torchvision.roi_align(a, b, 0.25, 7, 7, 2, False)

        self.common(fn, (torch.zeros([4, 256, 296, 304]), torch.zeros([2292, 5])))

    # https://github.com/halide/Halide/issues/8256
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_nll_loss_forward(self):
        def fn(a, b):
            return aten.nll_loss_forward(a, b, None, 1, -100)

        labels = (
            torch.zeros([5], dtype=torch.int64),
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))
        for a, b in zip(inps, labels):
            self.common(
                fn,
                (a, b),
            )

    def test_nll_loss_backward(self):
        def fn(a, b, c):
            return aten.nll_loss_backward(
                a, b, c, None, 1, -100, torch.tensor(1.0, device=self.device)
            )

        labels = (
            torch.zeros([5], dtype=torch.int64),
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))
        grad_outs = (torch.randn(()), torch.randn(()))
        for a, b, c in zip(grad_outs, inps, labels):
            self.common(
                fn,
                (a, b, c),
            )

    def test_isinf(self):
        def fn(x):
            return x.isinf(), x.isnan()

        self.common(
            fn, [torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")])]
        )
        self.common(
            fn,
            [
                torch.tensor(
                    [1, float("inf"), 2, float("-inf"), float("nan")],
                    dtype=torch.float64,
                )
            ],
        )

    @skip_if_halide  # different nan behavior in ==
    def test_isinf2(self):
        def fn(x):
            y = torch.tensor(
                [1, float("inf"), 2, float("-inf"), float("nan")], device=self.device
            )
            return x == y

        self.common(
            fn, (torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]),)
        )

    def test_any(self):
        def fn(x):
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        self.common(fn, [-torch.rand(64)])
        tmp = torch.randn(16, 8)
        tmp[1, 1] = float("inf")
        self.common(fn, [tmp])

    @skip_if_gpu_halide
    def test_multilayer_any(self):
        def fn(x):
            return (x.isinf().any(), x.isfinite().all())

        sample = torch.rand(9, 3, 353, 353)
        self.common(fn, [sample])

        sample.view(-1)[-1] = float("inf")
        self.common(fn, [sample])

    def test_inplace_activations(self):
        def fn(x):
            a = aten.hardswish_(x + 1)
            b = aten.hardtanh_(x + 1)
            c = aten.leaky_relu_(x + 1)
            d = aten.silu_(x + 1)
            e = aten.log1p(x + 1)
            f = aten.masked_fill_(x + 1, torch.zeros_like(x, dtype=torch.bool), 99.0)
            h = aten.masked_fill_(x + 1, torch.ones_like(x, dtype=torch.bool), 99.0)
            return (a, b, c, d, e, f, h)

        self.common(fn, [torch.randn(64) * 10])

    def test_baddbmm(self):
        def fn(a, b, c, beta):
            return aten.baddbmm(a, b, c, beta=beta)

        b = torch.randn(6, 128, 64)
        c = torch.randn(6, 64, 100)
        options = itertools.product(
            [torch.randn(6, 1, 100), torch.randn(6, 1, 100).fill_(torch.nan)],
            [0.0, 1.0],
        )
        for a, beta in options:
            self.common(
                fn,
                [a, b, c, beta],
                # Mismatched elements: 1212 / 76800 (1.6%)
                # Greatest absolute difference: 0.001953125 at index (0, 0, 93) (up to 1e-05 allowed)
                # Greatest relative difference: 1.0 at index (3, 19, 4) (up to 0.001 allowed)
                atol=0.002,
                rtol=0.001,
            )

    @config.patch({"triton.max_tiles": 2})
    def test_fuse_tiled(self):
        def fn(a, b, c):
            return a + b, c + 1

        self.common(
            fn, [torch.randn(128, 1), torch.randn(1, 128), torch.randn(128, 128)]
        )

    def test_expand_as(self):
        def fn(a, b):
            return aten.expand_as(a, b), aten.expand_as(a + 1, b + 1) + 1

        self.common(
            fn,
            [
                torch.randn(6, 1, 100),
                torch.randn(6, 128, 100),
            ],
        )

    def test_index_put1(self):
        def fn(a, b, c):
            return (
                torch.index_put(a, [b], c),
                torch.index_put_(a + 1, [b + 1], c + 1) + 1,
            )

        self.common(
            fn,
            [
                torch.randn([800, 256, 7, 7]),
                torch.randperm(601),
                torch.randn([601, 256, 7, 7]),
            ],
        )
        self.common(
            fn, [torch.randn(1024, 4, 2), torch.arange(4), torch.randn(4, 1, 1)]
        )

    def test_index_put2(self):
        def fn(a, b, c):
            return torch.index_put(a, [b], c, True)

        self.common(
            fn,
            [
                torch.randn([100, 256, 7, 7]),
                torch.randint(0, 100, size=[600], dtype=torch.int64),
                torch.randn([600, 256, 7, 7]),
            ],
            # workaround for https://github.com/openai/triton/issues/558
            check_lowp=False,
        )

    def test_index_put3(self):
        def fn(a, b, c):
            torch.ops.aten.index_put_(a, (None, b, None), c)
            a1 = a + 1
            torch.ops.aten.index_put_(a1, (None, b + 1, None), c + 1)
            return (a, a1)

        self.common(
            fn,
            [
                torch.randn([1024, 4, 2]),
                torch.arange(3),
                torch.randn([1024, 1, 2]),
            ],
        )

    def test_index_put4(self):
        # a, b[0] are not broadcastable
        # https://github.com/pytorch/pytorch/issues/97104
        def fn(a, b, c):
            return torch.index_put(a, [b], c)

        self.common(
            fn,
            [
                torch.rand([8, 2]),
                torch.rand([8]) > 0.5,
                torch.rand([]),
            ],
        )

    def test_index_put_as_masked_fill(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                True,
            ),
        )

    def test_index_put_fallback1(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                True,
            ),
        )

    def test_index_put_fallback2(self):
        def fn(a, b, c, d, e):
            a = a.clone()
            torch.ops.aten.index_put_(a, [None, b, c], d, e)
            return a

        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                False,
            ),
        )
        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                True,
            ),
        )

    def test_index_put_deterministic_fallback(self):
        with DeterministicGuard(True):

            def fn(a, b, c):
                return torch.index_put(a, [b], c, True)

            self.common(
                fn,
                [
                    torch.randn([100, 32]),
                    torch.randint(0, 100, size=[600], dtype=torch.int64),
                    torch.randn([600, 32]),
                ],
                check_lowp=False,
            )

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8312
    def test_index_put_index(self):
        def fn(ind, x, src):
            y = torch.ops.aten.index_put.default(x, [ind], src)
            return torch.ops.aten.index.Tensor(y, [ind])

        args = [torch.tensor([1], dtype=torch.int64), torch.randn(8, 4), torch.randn(4)]
        self.common(fn, args)

    def test_index_put_reinplace(self):
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            x.index_put_((idx,), src)
            return x.expand((2, x.shape[0]))

        a = torch.randn(1024)
        idx = torch.arange(10)
        torch._inductor.metrics.generated_kernel_count = 0
        self.common(fn, (a, idx))
        assertGeneratedKernelCountEqual(self, 1)

    def test_index_put_failed_reinplace(self):
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            y = x.index_put((idx,), src)
            return x, y

        a = torch.randn(1024)
        idx = torch.arange(10)
        torch._inductor.metrics.generated_kernel_count = 0
        self.common(fn, (a, idx))
        assertGeneratedKernelCountEqual(self, 2)

    def test_adding_tensor_offsets(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x[16:32]

        with torch.no_grad():
            x = torch.randn(1024, device=self.device)
            self.assertEqual(fn(x[0:]), x[16:][:16])
            self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])

    # from GPT2ForSequenceClassification
    @skip_if_gpu_halide
    def test_index_tensor(self):
        def fn(x, y):
            ne = torch.ops.aten.ne.Scalar(x, 0)
            sum = torch.ops.aten.sum.dim_IntList(ne, [-1])
            sub = torch.ops.aten.sub.Tensor(sum, 1)
            iota = torch.ops.prims.iota.default(
                1,
                start=0,
                step=1,
                dtype=torch.int64,
                device=x.device,
                requires_grad=False,
            )
            return torch.ops.aten.index.Tensor(y, [iota, sub])

        self.common(fn, [torch.randn(1, 1024), torch.randn(1, 1024, 2)])

    @config.patch(fallback_random=True)
    def test_bernoulli1(self):
        def fn(a):
            b = torch.empty_like(a)
            return aten.bernoulli_(b), b

        self.common(
            fn,
            [
                torch.randn([100]),
            ],
        )

    def test_bernoulli2(self):
        def fn(a):
            return aten.bernoulli(a)

        self.common(
            fn,
            [torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])],
        )

    def test_narrow(self):
        def fn(x):
            return (
                aten.narrow(x, 1, 10, 16),
                aten.narrow(x + 2, 0, 10, 16) + 1,
                aten.narrow_copy(x, 1, 10, 16),
            )

        self.common(fn, [torch.randn(64, 64)])

    def test_new_cpp_build_logical(self):
        from torch._inductor.codecache import validate_new_cpp_commands

        validate_new_cpp_commands()

    def test_as_strided(self):
        def fn(x):
            return (
                aten.as_strided(x, (8, 8, 64), (8 * 64, 64, 1), 0),
                aten.as_strided(x + 1, (8, 8, 64), (8 * 64, 64, 1), 0) + 2,
            )

        def fn_channels_last(x):
            return (
                aten.as_strided(
                    x, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
                ),
                aten.as_strided(
                    x + 1, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
                )
                + 2,
            )

        self.common(fn, [torch.randn(64, 64)])
        self.common(
            fn_channels_last,
            [torch.randn(8, 384, 20, 20).to(memory_format=torch.channels_last)],
        )

    def test_like_channels_last(self):
        def foo():
            randn = torch.randn((4, 3, 8, 8), device=self.device, dtype=torch.float32)
            xc = randn.contiguous(memory_format=torch.channels_last)
            clone = torch.zeros_like(xc, memory_format=torch.preserve_format)
            rand_like = torch.rand_like(randn)
            return (xc, clone, rand_like)

        out = foo()
        out_comp = torch.compile()(foo)()

        for t, t_comp in zip(out, out_comp):
            self.assertEqual(t.stride(), t_comp.stride())

    def test_as_strided_scatter(self):
        def fn(a, b):
            return aten.as_strided_scatter(
                a * 8 + 10,
                b * 2 - 4,
                size=(a.shape[0], a.shape[1] // 2),
                stride=(a.shape[1], 2),
                storage_offset=0,
            )

        self.common(fn, [torch.randn(10, 1024), torch.randn(10, 512)])

    def test_select_scatter(self):
        def fn(x, a, b):
            return (
                aten.select_scatter(x, a, 1, 0),
                aten.select_scatter(x, b, 0, 1),
            )

        self.common(
            fn,
            [
                torch.randn(8, 197, 38),
                torch.randn(8, 38),
                torch.randn(197, 38),
            ],
        )

    @skip_if_gpu_halide  # accuracy issue
    def test_slice_scatter(self):
        def fn(x, a):
            return (
                aten.slice_scatter(x, a, 2, 10, -10),
                aten.slice_scatter(x, a[:, :, :40], 2, 10, -10, 2),
            )

        self.common(
            fn,
            [
                torch.randn(4, 8, 100),
                torch.randn(4, 8, 80),
            ],
        )

    def test_slice_scatter2(self):
        def fn(a, b):
            return aten.slice_scatter(a, b, 0, 0, 9223372036854775807)

        self.common(
            fn,
            [
                torch.randn([8, 197, 384]),
                torch.randn([8, 197, 384]),
            ],
        )

    def test_slice_scatter3(self):
        def fn(a, b):
            return aten.slice_scatter.default(a, b, 1, 1, 9223372036854775807, 2)

        self.common(
            fn,
            [
                torch.randn([1, 4]),
                torch.randn([1, 2]),
            ],
        )

    def test_slice_scatter4(self):
        def fn(a, b):
            return aten.slice_scatter.default(a, b, 1, 2, 9223372036854775807, 3)

        self.common(
            fn,
            [
                torch.randn([1, 9]),
                torch.randn([1, 3]),
            ],
        )

    def test_slice_scatter5(self):
        # empty slices that require clamping the start or end
        def fn(a, b):
            return (
                aten.slice_scatter.default(a, b, 0, 2, 0, 1),
                aten.slice_scatter.default(a, b, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice_scatter.default(a, b, 0, -20, 0, 1),
                aten.slice_scatter.default(a, b, 0, -20, -16, 1),
            )

        a = torch.arange(10, dtype=torch.float)
        b = torch.empty(0)
        self.common(fn, [a, b])

    def test_slice_scatter_reinplace(self):
        class M(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.linear1 = nn.Linear(64, 64, bias=False)
                self.cache_k = torch.zeros((56, 384, 8, 64), device=device)

            def forward(self, x, start_pos):
                bsz, seqlen, _, _ = x.shape
                xk = self.linear1(x)
                with torch.no_grad():
                    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                scores = torch.matmul(
                    xk.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3)
                )
                return scores

        kv_cache_module = M(self.device)
        inp = torch.randn(1, 32, 8, 64)

        # Test that the cache update is reinplaced such that the cache is updated inplace
        # rather than copy-scatter-copy-back.

        torch._inductor.metrics.generated_kernel_count = 0
        with torch.no_grad():
            self.common(kv_cache_module, (inp, 1), check_lowp=False)
        assertGeneratedKernelCountEqual(self, 1)

    @skip_if_gpu_halide  # compile error on gpu
    def test_scatter1(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        self.common(
            fn,
            [
                torch.zeros(2, 3),
                -1,
                torch.tensor([[0]]),
                torch.ones(2, 3),
            ],
        )

    def test_scatter2(self):
        if self.device == "cuda":
            raise unittest.SkipTest("unstable on sm86")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        def fn(a, dim, index, b):
            return aten.scatter.reduce(a, dim, index, b, reduce="add")

        self.common(
            fn,
            [
                torch.zeros(64, 512),
                0,
                torch.zeros((64, 512), dtype=torch.int64),
                torch.ones(64, 512),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter3(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                0.8,  # src can be a scalar
            ],
            # Mismatched elements: 1 / 1885 (0.1%)
            # Greatest absolute difference: 0.00018310546875 at index (0, 0, 3) (up to 1e-05 allowed)
            # Greatest relative difference: 0.0022371364653243847 at index (0, 0, 3) (up to 0.001 allowed)
            atol=2e-4,
            rtol=1e-3,
            check_lowp=check_lowp,
        )

    def test_scatter4(self):
        def fn(x, ind, src):
            return torch.scatter(x, 0, ind, src)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(196, 992),
                        torch.randint(196, (1, 992)),
                        torch.randn(1, 992),
                    ],
                    check_lowp=check_lowp,
                )

    def test_scatter5(self):
        def fn(a, dim, index, b, reduce):
            a = a.clone()
            a.scatter_(dim, index, b, reduce=reduce)
            a1 = a + 1.0
            a1.scatter_(dim, index, b, reduce=reduce)
            return (a, a1)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["add", "multiply"]:
            self.common(
                fn,
                [
                    torch.ones((4, 5)),
                    0,
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),
                    torch.randn(4, 5),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    def test_scatter6(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(5, 8, 13),
                        2,
                        torch.tensor([[[3, 5, 7, 9]]]),
                        0.8,  # src can be a scalar
                    ],
                    check_lowp=check_lowp,
                )

    @unittest.skip("Flaky test, needs debugging")
    def test_scatter_add1(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0]]),
                torch.randn(2, 3),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_add2(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0, 0, 0], [1, 1, 1]]),
                torch.randn(2, 3),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_add3(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(5, 29, 13),
                        2,
                        torch.tensor([[[3, 5, 7, 9]]]),
                        torch.randn(1, 1, 10),
                    ],
                    check_lowp=check_lowp,
                )

    def test_scatter_reduce1(self):
        def fn(a, dim, index, b):
            return aten.scatter_reduce(a, dim, index, b, "sum")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                torch.randn(1, 1, 10),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_reduce2(self):
        def fn(a, dim, index, b, reduce):
            return aten.scatter_reduce(a, dim, index, b, reduce, include_self=False)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["sum", "amax"]:
            self.common(
                fn,
                [
                    torch.randn(2, 3),
                    0,
                    torch.zeros((2, 3), dtype=torch.int64),
                    torch.randn(2, 3),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    def test_scatter_reduce3(self):
        def fn(a, dim, index, b, reduce):
            a = a.clone()
            a.scatter_reduce_(dim, index, b, reduce=reduce)
            a1 = a + 1.0
            a1.scatter_reduce_(dim, index, b, reduce=reduce)
            return (a, a1)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["sum", "prod"]:
            self.common(
                fn,
                [
                    torch.ones((4, 5)),
                    0,
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),
                    torch.randn(4, 5),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    @skip_if_gpu_halide
    def test_dense_mask_index(self):
        r"""
        There will be a little difference for reduce order between aten and inductor
        https://github.com/pytorch/pytorch/pull/122289
        Absolute difference: 0.00067138671875 (up to 1e-05 allowed)
        Relative difference: 3.1747371732500974e-06 (up to 1.3e-06 allowed)
        """
        kwargs = {}
        if self.device == "cpu":
            kwargs["atol"] = 1e-4
            kwargs["rtol"] = 1.3e-5

        def fn(x, y):
            y = torch.ops.aten.select.int(y, 0, 2)
            z = x * y
            return z.sum()

        self.common(fn, [torch.randn(102400), torch.randn(3)], **kwargs)

    def test_empty1(self):
        def fn():
            return torch.empty((1, 128, 128))

        self.common(fn, [], assert_equal=False)

    def test_empty2(self):
        def fn():
            return aten.empty((1, 128, 128))

        self.common(fn, [], assert_equal=False)

    def test_new_empty(self):
        def fn(a):
            return aten.new_empty(a, [1, 128, 128])

        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_empty_strided(self):
        def fn():
            return aten.empty_strided([1, 128, 128], [16384, 128, 1])

        self.common(fn, [], assert_equal=False)

    def test_new_empty_strided(self):
        def fn(a):
            return aten.new_empty_strided(a, [1, 128, 128], [16384, 128, 1])

        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_dropout_trivial_0(self):
        def fn1(a):
            return torch.nn.functional.dropout(a, 0.0, True) + a

        self.common(fn1, [torch.randn(55)])

    def test_dropout_trivial_1(self):
        def fn2(a):
            return torch.nn.functional.dropout(a, 1.0, True) + a

        self.common(fn2, [torch.randn(55)])

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    @skipIfWindows
    def test_dropout(self):
        random.seed(1234)
        torch.manual_seed(1234)

        @torch._dynamo.optimize("inductor")
        def fn1(a):
            return torch.nn.functional.dropout(a)

        x = torch.ones(1000, device=self.device, dtype=torch.float32)
        result1 = fn1(x)
        self.assertTrue(400 < result1.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result1.mean().item() < 1.1)

        random.seed(1234)
        torch.manual_seed(1234)

        @torch._dynamo.optimize("inductor")
        def fn2(a):
            return torch.nn.functional.dropout(a, 0.5, True)

        result2 = fn2(x)
        self.assertTrue(400 < result2.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result2.mean().item() < 1.1)

    @dynamo_config.patch(automatic_dynamic_shapes=True)
    @skipIfWindows
    def test_dropout_deterministic(self):
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.nn.functional.dropout(a, 0.55, True)

        for cg in [False, True]:
            with patch.object(config.triton, "cudagraphs", cg):
                torch._dynamo.reset()

                x = torch.ones(1024, device=self.device, dtype=torch.float32)

                torch.manual_seed(1234)
                a0 = fn(x).clone()
                a1 = fn(x).clone()
                a2 = fn(x).clone()

                torch.manual_seed(1234)
                b0 = fn(x).clone()
                b1 = fn(x).clone()
                b2 = fn(x).clone()

                # same seed, same values
                self.assertTrue(torch.allclose(a0, b0))
                self.assertTrue(torch.allclose(a1, b1))
                self.assertTrue(torch.allclose(a2, b2))

                # different calls, different values
                self.assertFalse(torch.allclose(a0, a1))
                self.assertFalse(torch.allclose(a1, a2))

    @skipIfWindows
    def test_rand_like_deterministic(self):
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.rand_like(a), torch.rand_like(a)

        x = torch.ones(1024, device=self.device, dtype=torch.float32)

        torch.manual_seed(1234)
        a0 = fn(x)[0].clone()
        a1 = fn(x)[0].clone()
        a2 = fn(x)[0].clone()

        torch.manual_seed(1234)
        b0 = fn(x)[0].clone()
        b1 = fn(x)[0].clone()
        b2 = fn(x)[0].clone()

        # same seed, same values
        self.assertTrue(torch.allclose(a0, b0))
        self.assertTrue(torch.allclose(a1, b1))
        self.assertTrue(torch.allclose(a2, b2))

        # different calls, different values
        self.assertFalse(torch.allclose(a0, a1))
        self.assertFalse(torch.allclose(a1, a2))

        c, d = fn(x)
        self.assertFalse(torch.allclose(c, d))
        self.assertTrue((c >= 0).all())
        self.assertTrue((c < 1).all())
        self.assertTrue((d >= 0).all())
        self.assertTrue((d < 1).all())

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_basic(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b, c, d, e=2):
                a.add_(b[0] * c * e),
                if d is not None:
                    d.add_(b[1])

            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> ()"
            )
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # We do some clones and copy_ to test that Inductor doesn't reorder
            # the copy_ w.r.t. inplace_.
            def f(a, b1, b2, c, d):
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                torch.ops.mylib.inplace_(a_, (b1, b2), c, d=d_)
                a.copy_(a_)
                if d is not None:
                    d.copy_(d_)
                return ()

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f(list(cloned_args))
            f(*args)
            self.assertEqual(cloned_args, args)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_with_return(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b, c, d, e=2):
                a.add_(b[0] * c * e),
                if d is not None:
                    d.add_(b[1])
                return b[0] + b[1]

            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> Tensor"
            )
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # We do some clones and copy_ to test that Inductor doesn't reorder
            # the copy_ w.r.t. inplace_.
            def f(a, b0, b1, c, d):
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                res = torch.ops.mylib.inplace_(a_, (b0, b1), c, d=d_)
                a.copy_(a_)
                if d is not None:
                    d.copy_(d_)
                return (res,)

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_out = compiled_f(list(cloned_args))
            out = f(*args)
            self.assertEqual(cloned_args, args)
            self.assertEqual(compiled_out, out)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_no_mutated_tensors(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b):
                if b is not None:
                    b.add_(1)

            m.define("inplace_(Tensor a, Tensor(b!)? b) -> ()")
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            def f(a):
                torch.ops.mylib.inplace_(a, None)
                return ()

            a = torch.tensor([0.0, 1.0, 2])
            args = (a,)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f(list(cloned_args))
            f(*args)
            self.assertEqual(cloned_args, args)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_list(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b):
                for bi in b:
                    bi.add_(a)

            m.define("inplace_(Tensor a, Tensor(a!)[] b) -> ()")
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            def f(a, b):
                torch.ops.mylib.inplace_(a, b)
                return None

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            args = (a, b)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)

            compiled_f = compile_fx_inner(mod, cloned_args)

        @torch.library.custom_op("mylib::sin_out", mutates_args={"outs"})
        def sin_out(x: torch.Tensor, outs: typing.List[torch.Tensor]) -> None:
            x_np = x.numpy()
            assert len(outs) == 2
            out_np0 = out[0].numpy()
            out_np1 = out[1].numpy()
            np.sin(x_np, out=out_np0)
            np.sin(x_np, out=out_np1)

        @torch.compile
        def g(x):
            outs = [torch.empty_like(x) for _ in range(2)]
            sin_out(x, outs)
            return outs

        x = torch.randn(3)
        out = [torch.empty_like(x) for _ in range(2)]
        y = g(x)

    @expectedFailureXPU
    def test_functionalize_rng_wrappers(self):
        # Ideally, we would like to use torch.compile for these operators. But
        # currently the plan is to introduce these operators at the partitioner
        # level, obviating the need to support them fully through the
        # torch.compile stack. To ensure that we have good enough debugging with
        # minifiers, we have ensure that they work with make_fx. This test uses
        # make_fx to do the testing. In future, we can move on torch.compile.
        def fn():
            rng_state1, a1 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            rng_state2, a2 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            b1 = torch._prims.rng_prims.run_with_rng_state(
                rng_state1,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            b2 = torch._prims.rng_prims.run_with_rng_state(
                rng_state2,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            return (a1, a2, b1, b2)

        mod = make_fx(fn)()
        compiled_f = compile_fx_inner(mod, ())
        a1, a2, b1, b2 = compiled_f(())
        self.assertEqual(a1, b1)
        self.assertEqual(a2, b2)

    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    @expectedFailureXPU
    @skip_if_gpu_halide  # rand
    def test_philox_rand(self):
        if self.device == "cpu":
            raise unittest.SkipTest(
                f"functionalization of rng ops supported only on {GPU_TYPE}"
            )

        @torch._dynamo.optimize("inductor")
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        def check(x):
            torch.manual_seed(123)
            a = fn(x)

            torch.manual_seed(1234)
            b = fn(x)

            torch.manual_seed(123)
            c = fn(x)

            # same seed, same values
            self.assertTrue(torch.allclose(a, c))

            # different calls, different values
            self.assertFalse(torch.allclose(a, b))

        check(torch.ones(1024, device=self.device, dtype=torch.float32))
        # Need comment: should we add "_get_rng_state_offset" to common device interface?
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 2048)
        # Check non-multiple of 4 numel
        check(torch.ones(3, device=self.device, dtype=torch.float32))
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 8)

    # Already on by default, just want to make sure
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    def test_reuse_buffers_with_aliasing(self):
        def f(x):
            z = x + 1
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        self.common(f, (torch.zeros((4, 2)),))

        code = run_and_get_triton_code(torch.compile(f), torch.zeros((4, 2)))
        # Make sure that we haven't added complex support and made this test
        # invalid. If we've added complex support please update the test to use
        # a different set of view ops we don't lower
        self.assertTrue("aten.view_as_real" in code)

        def f2(x):
            z = x + 1
            z = torch.view_as_complex(z)
            z = torch.view_as_real(z)
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        self.common(f, (torch.zeros((4, 2)),))

    def test_randn_like_empty(self):
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, v1: torch.Tensor):
                vx = v1.min(dim=1).values
                v2 = torch.randn_like(vx)
                return v2

        model = Model()
        x = torch.rand(10, 3, 0)

        self.common(model, (x,))

    def test_randint(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return (
                torch.randint(10, [1024], device=x.device),
                torch.randint(-4, 7, [1024], dtype=torch.int32, device=x.device),
                torch.randint_like(x, 2**50),
            )

        torch.manual_seed(12345)
        a0, b0, c0 = fn(torch.zeros([40, 40], device=self.device))
        self.assertEqual(a0.shape, [1024])
        self.assertEqual(b0.shape, [1024])
        self.assertEqual(c0.shape, [40, 40])
        torch.manual_seed(12345)
        a1, b1, c1 = fn(torch.zeros([40, 40], device=self.device))
        self.assertEqual(a0, a1)
        self.assertEqual(b0, b1)
        self.assertEqual(c0, c1)

        self.assertEqual(a0.min(), 0)
        self.assertEqual(a0.max(), 9)

        self.assertEqual(b0.min(), -4)
        self.assertEqual(b0.max(), 6)

        self.assertGreaterEqual(c0.min(), 0)
        self.assertGreater(c0.max(), 2**40)
        self.assertLess(c0.max(), 2**50)

    @config.patch(fallback_random=True)
    def test_like_rands(self):
        def fn(x):
            return torch.rand_like(x), torch.randn_like(x)

        self.common(fn, [torch.zeros([20, 20])])

    def test_like_rands2(self):
        # rand_like with kwargs `device` of str type
        d = self.device
        assert isinstance(d, str)

        @torch.compile
        def fn(x):
            return torch.rand_like(x, device=d)

        x = torch.ones(10, device=self.device, dtype=torch.float32)
        a0 = fn(x).clone()
        a1 = fn(x).clone()
        self.assertFalse(torch.allclose(a0, a1))

    @requires_gpu()
    def test_like_rands3(self):
        # rand_like with `device` which is different from `x.device`
        def test_like_rands_on_different_device(device1, device2):
            @torch.compile
            def fn(x, device):
                return torch.rand_like(x, device=device)

            x = torch.ones(10, device=device1, dtype=torch.float32)
            return fn(x, device2).clone()

        a0 = test_like_rands_on_different_device("cpu", GPU_TYPE)
        a1 = test_like_rands_on_different_device(GPU_TYPE, "cpu")
        self.assertTrue(a0.device.type == GPU_TYPE)
        self.assertTrue(a1.device.type == "cpu")

    def test_max_pool2d_with_indices_backward(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [2, 2], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([2, 4, 18, 14])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d_with_indices_backward2(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 3], [2, 2], [1, 1], [1, 1], True, c
            )

        x = torch.randn([2, 4, 40, 56])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1200
    def test_max_pool2d_with_indices_backward3(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [1, 1], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([32, 256, 37, 38])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [1, 1],
            [2, 2],
            0,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1352
    @skip_if_halide  # hangs forever
    def test_max_pool2d_with_indices_backward4(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [5, 5], [1, 1], [2, 2], [1, 1], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 64, 3, 4])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [5, 5],
            [1, 1],
            2,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    @expectedFailureXPU
    def test_max_pool2d_with_indices_backward5(self):
        # Window size is too big. Should fallback
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [13, 13], [1, 1], [2, 2], [1, 1], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 64, 20, 20])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [13, 13],
            [1, 1],
            2,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 0)

    # From https://github.com/pytorch/pytorch/issues/93384
    def test_max_pool2d_with_indices_backward6(self):
        # dilation is not 1. Should fallback
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 2], [2, 1], [1, 1], [1, 2], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 2, 3, 6])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 2],
            [2, 1],
            [1, 1],
            [1, 2],
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_issue102546(self):
        def fn(x):
            return x.mean(0)

        self.common(fn, [torch.rand(())])

    def test_avg_pool2d_backward(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [2, 2],
                [2, 2],
                [0, 0],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7]),
                torch.randn([2, 4, 14, 14]),
            ],
        )

    @skip_if_gpu_halide  # slow
    def test_avg_pool2d_backward2(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [3, 3],
                [1, 1],
                [1, 1],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 15]),
                torch.randn([1, 1, 20, 15]),
            ],
        )

    def test_avg_pool2d_backward3(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [1, 1],
                [2, 2],
                [0, 0],
                False,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11]),
                torch.randn([1, 2016, 21, 21]),
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    def test_avg_pool2d_backward4(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [13, 13],
                [1, 1],
                [0, 0],
                True,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12]),
                torch.randn([1, 16, 24, 24]),
            ],
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool3d_backward(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7, 7]),
                torch.randn([2, 4, 14, 14, 14]),
            ],
        )

    @skip_if_halide  # compiles for 5+ minutes
    def test_avg_pool3d_backward2(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 20, 15]),
                torch.randn([1, 1, 20, 20, 15]),
            ],
        )

    def test_avg_pool3d_backward3(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                False,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11, 11]),
                torch.randn([1, 2016, 21, 21, 21]),
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    def test_avg_pool3d_backward4(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [13, 13, 13],
                [1, 1, 1],
                [0, 0, 0],
                True,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12, 12]),
                torch.randn([1, 16, 24, 24, 24]),
            ],
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    @config.patch(search_autotune_cache=False)
    def test_mm_views(self):
        def fn(a, b):
            return torch.mm(a.view(32, 32), b.view(32, 32))

        self.common(
            fn,
            (
                torch.randn([32, 32]).transpose(0, 1),
                torch.randn([1, 32, 32]).transpose(0, 1),
            ),
            check_lowp=False,
        )
        expected_kernel = 0
        # codegen mm kernel from template
        self.assertEqual(
            torch._inductor.metrics.generated_kernel_count, expected_kernel
        )

    @torch._dynamo.config.patch(assume_static_by_default=False)
    @skipIfWindows
    def test_dtype_sympy_expr(self):
        @torch._dynamo.optimize_assert("inductor")
        def fn(a):
            y = a[..., :-1, :].contiguous()
            return y

        result = fn(torch.randn([1, 2, 16, 4]).requires_grad_())
        result.sum().backward()

    @skipIfWindows
    def test_dropout2(self):
        n = 100000
        weight = torch.ones(
            n, device=self.device, dtype=torch.float32, requires_grad=True
        )
        ones = torch.ones(n, device=self.device, dtype=torch.float32)

        @torch._dynamo.optimize_assert("inductor")
        def run(x, train=True):
            return F.dropout(x * weight, 0.33, train)

        def check(r, g):
            rmean = r.mean().item()
            gmean = g.mean().item()
            rcount = len(r.nonzero())
            gcount = len(g.nonzero())

            # dropped elements should match
            self.assertTrue(same(r.nonzero(), g.nonzero()))
            self.assertEqual(rcount, gcount)

            # dropped should be close to 0.33
            self.assertGreater(rcount, 0.64 * n)
            self.assertGreater(0.68 * n, rcount)

            self.assertAlmostEqual(rmean, gmean)
            self.assertAlmostEqual(rmean, 1.0, places=2)

        r1 = run(ones, train=False)
        r1.sum().backward()
        g1 = weight.grad.clone()
        # eval mode should be all ones
        self.assertTrue(same(r1, torch.ones_like(r1)))
        self.assertTrue(same(g1, torch.ones_like(g1)))

        torch.manual_seed(1234)
        weight.grad.zero_()
        r2, (fw_code, bw_code) = run_fw_bw_and_get_code(lambda: run(ones))
        if is_halide_backend(self.device):
            self.assertEqual(fw_code.count("halide_helpers.rand"), 1)
            self.assertEqual(bw_code.count("halide_helpers.rand"), 0)
        elif self.device == GPU_TYPE:
            self.assertEqual(fw_code.count("tl.rand"), 1)
            self.assertEqual(bw_code.count("tl.rand"), 0)
        g2 = weight.grad.clone()
        check(r2, g2)

        torch.manual_seed(1234)
        weight.grad.zero_()
        r3 = run(ones)
        r3.sum().backward()
        g3 = weight.grad.clone()
        check(r3, g3)

        # second run is same result as first
        self.assertTrue(same(r2, r3))
        self.assertTrue(same(g2, g3))

    @config.patch(search_autotune_cache=False)
    @skipIfWindows
    def test_dropout3(self):
        m = torch.nn.Sequential(
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
        ).to(self.device)

        @torch._dynamo.optimize_assert("inductor")
        def run(x):
            return m(x)

        torch._inductor.metrics.generated_kernel_count = 0

        result, (fw_code, bw_code) = run_fw_bw_and_get_code(
            lambda: run(torch.randn([8, 32], device=self.device))
        )

        if is_halide_backend(self.device):
            self.assertEqual(fw_code.count("halide_helpers.rand"), 2)
            self.assertEqual(bw_code.count("halide_helpers.rand"), 0)
        elif self.device == GPU_TYPE:
            self.assertEqual(fw_code.count("tl.rand"), 2)
            self.assertEqual(bw_code.count("tl.rand"), 0)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @skipIfWindows
    def test_randint_kernel_count(self):
        @torch._dynamo.optimize_assert("inductor")
        def fn1():
            random_tensor1 = torch.randint(10, [32], device=self.device)
            random_tensor2 = torch.randint(10, [32], device=self.device)
            random_tensor3 = torch.randint(10, [32], device=self.device)
            return random_tensor1, random_tensor2, random_tensor3

        _, source_codes = run_and_get_code(fn1)
        if self.device == GPU_TYPE:
            self.assertEqual(len(source_codes), 1)
            self.assertEqual(source_codes[0].count("async_compile.triton"), 2)

    def test_roll(self):
        def fn(a):
            return (
                aten.roll(a, [-3, 10], [1, 2]),
                aten.roll(a, [5]),
            )

        self.common(
            fn,
            [
                torch.randn([2, 56, 56, 16]),
            ],
        )

    def test_argmax_min_int32(self):
        # https://github.com/pytorch/pytorch/issues/94055
        def fn(a, b):
            c = a.argmax(3)
            return torch.min(b, c)

        a = torch.rand(3, 4, 2, 1).int()
        b = torch.rand(2, 2, 1, 4, 1).int()
        self.common(fn, (a, b))

    def test_argmax_argmin1(self):
        def fn(x):
            return (aten.argmax(x), aten.argmin(x))

        self.common(
            fn,
            [
                torch.randn([8, 256, 256]),
            ],
        )

    def test_argmax_argmin2(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, 1),
                aten.argmin(x, 1),
            )

        self.common(fn, (torch.randn([144, 144]),))

    def test_argmax_argmin_with_duplicates(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, 1),
                aten.argmin(x, 1),
            )

        # Unrolled reduction
        t1 = torch.randint(2, size=(6, 6))
        self.common(fn, (t1,))

        # Persistent reduction
        t1 = torch.randint(8, size=(32, 32))
        self.common(fn, (t1,))

        # Non-persistent reduction
        t1 = torch.randint(8, size=(1028, 1028))
        self.common(fn, (t1,))

    @skip_if_halide  # nan behavior
    def test_argmax_argmin_with_nan(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, 1),
                aten.argmin(x, 1),
            )

        if self.device == "cpu":
            raise unittest.SkipTest("broken on CPU")

        # Unrolled reduction
        t1 = torch.randn((6, 6))
        t1[:, 1] = float("nan")
        t1[:, 3] = float("nan")
        self.common(fn, (t1,))

        # Persistent reduction
        t1 = torch.randn((32, 32))
        t1[:, 4] = float("nan")
        t1[:, 8] = float("nan")
        self.common(fn, (t1,))

        # Non-persistent reduction
        t1 = torch.randn((1028, 1028))
        t1[:, 40] = float("nan")
        t1[:, 100] = float("nan")
        self.common(fn, (t1,))

    def test_conv_backward(self):
        def fn(rank4_inps, rank3_inps, rank5_inps):
            out1 = aten.convolution_backward(
                *rank4_inps,
                [C],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            out2 = aten.convolution_backward(
                *rank4_inps,
                [C],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, False, False],
            )
            out3 = aten.convolution_backward(
                *rank3_inps,
                [C],
                [1],
                [0],
                [1],
                False,
                [0],
                1,
                [True, True, True],
            )
            out4 = aten.convolution_backward(
                *rank5_inps,
                [C],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                False,
                [0, 0, 0],
                1,
                [True, True, True],
            )
            return (out1, out2, out3, out4)

        B = 3
        C = 4
        H = 5
        grad_out = torch.randn(B, C, H - 2, H - 2, H - 2)
        inp = torch.randn(B, C, H, H, H)
        weight = torch.randn(C, C, 3, 3, 3)

        def shrink_rank(x, rank):
            res = x
            while res.dim() > rank:
                res = torch.select(res, -1, 0)
            return res.contiguous()

        rank4_inps = [shrink_rank(x, 4) for x in [grad_out, inp, weight]]
        rank3_inps = [shrink_rank(x, 4) for x in [grad_out, inp, weight]]
        rank5_inps = [shrink_rank(x, 5) for x in [grad_out, inp, weight]]

        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            self.common(
                fn,
                [rank4_inps, rank3_inps, rank5_inps],
            )

    @unittest.skip(
        """
        FIXME: In the case of having equally max/min elements, our implementation returns
        the last index instead of the first one
        """
    )
    def test_argmax_argmin3(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, -1),
                aten.argmin(x, -1),
            )

        self.common(
            fn,
            [torch.randint(0, 5, [10, 10])],
        )

    def test_vdd_clamp(self):
        def fn(x):
            return torch.clamp_min(x, 3)

        self.common(
            fn,
            [
                torch.randn([16], requires_grad=True) * 10,
            ],
        )

    def test_tmp_not_defined_issue1(self):
        def forward(
            primals_3,
            primals_4,
            add_tensor,
            convert_element_type_default,
            div_default,
            reciprocal_default,
        ):
            var_default = torch.ops.aten.var(
                convert_element_type_default, [2], correction=0
            )
            sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, div_default)
            mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, reciprocal_default)
            mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_3)
            add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_2, primals_4)
            convert_element_type_default_1 = add_tensor_2.to(dtype=torch.float32)
            convert_element_type_default_2 = convert_element_type_default_1.to(
                dtype=torch.float32
            )
            var_default_1 = torch.ops.aten.var(
                convert_element_type_default_2, [2], correction=0
            )
            broadcast_in_dim_default_2 = var_default_1.reshape(1, 512, 1)
            sum_default_1 = convert_element_type_default_2.sum(2)
            add_tensor_3 = torch.ops.aten.add.Tensor(broadcast_in_dim_default_2, 1e-05)
            return (var_default, sum_default_1, add_tensor_3)

        inps = [
            (torch.Size([1024]), torch.float32),
            (torch.Size([1024]), torch.float32),
            (torch.Size([1, 512, 1024]), torch.float32),
            (torch.Size([1, 512, 1024]), torch.float32),
            (torch.Size([1, 512, 1]), torch.float32),
            (torch.Size([1, 512, 1]), torch.float32),
        ]
        inps = [torch.randn(shape, dtype=dtype) for (shape, dtype) in inps]
        self.common(forward, inps, atol=1e-05, rtol=2e-05)

    @unittest.skipIf(
        os.environ.get("BUILD_ENVIRONMENT", "").startswith("parallelnative"),
        "TODO: debug this with asan",
    )
    @skip_if_gpu_halide
    def test_tmp_not_defined_issue2(self):
        def forward(arg38_1, arg81_1, getitem_17, new_zeros_default_4):
            div_tensor_7 = torch.ops.aten.div.Tensor(getitem_17, arg81_1)
            mul_tensor_24 = torch.ops.aten.mul.Tensor(div_tensor_7, arg38_1)
            sum_default_7 = torch.ops.aten.sum.default(mul_tensor_24)
            return (new_zeros_default_4, sum_default_7)

        dtype = torch.float32
        args = [
            ((1, 88, 40, 40), (140800, 1600, 40, 1), dtype),
            ((), (), dtype),
            ((1, 88, 40, 40), (140800, 1600, 40, 1), dtype),
            ((3,), (1,), dtype),
        ]
        args = [
            rand_strided(shape, stride, dtype).requires_grad_(True).add(1)
            for shape, stride, dtype in args
        ]
        self.common(forward, args, atol=1e-5, rtol=1e-5)

    @requires_gpu()
    @skip_if_halide  # cascading accuracy issues due rsqrt fallback
    def test_tmp_not_defined_issue3(self):
        from torch import device

        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[1001]",
            primals_3: "f32[1001, 64]",
            primals_4: "f32[4190]",
            primals_5: "f32[4190]",
            primals_6: "f32[1739, 4190]",
            primals_48: "f32[6144, 4191]",
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0
            lift_fresh_copy: "i64[4190]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant0
            )

            index: "f32[6144, 4190]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy]
            )

            _tensor_constant1: "i64[6]" = self._tensor_constant1
            lift_fresh_copy_1: "i64[6]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant1
            )
            index_1: "f32[6144, 6]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy_1]
            )
            primals_48 = lift_fresh_copy_1 = None
            permute: "f32[6, 1001]" = torch.ops.aten.permute.default(primals_1, [1, 0])
            addmm: "f32[6144, 1001]" = torch.ops.aten.addmm.default(
                primals_2, index_1, permute
            )
            amax: "f32[6144, 1]" = torch.ops.aten.amax.default(addmm, [-1], True)
            sub: "f32[6144, 1001]" = torch.ops.aten.sub.Tensor(addmm, amax)
            exp: "f32[6144, 1001]" = torch.ops.aten.exp.default(sub)
            sum_1: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
            div: "f32[6144, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1)

            full_default: "i32[6144, 1001]" = torch.ops.aten.full.default(
                [6144, 1001],
                1,
                dtype=torch.int32,
                layout=torch.strided,
                device=device(type=GPU_TYPE, index=0),
                pin_memory=False,
            )

            iota: "i32[1001]" = torch.ops.prims.iota.default(
                1001,
                start=0,
                step=1,
                dtype=torch.int32,
                device=device(type=GPU_TYPE),
                requires_grad=False,
            )

            mul: "i32[6144, 1001]" = torch.ops.aten.mul.Tensor(full_default, iota)
            iota_1: "i32[6144]" = torch.ops.prims.iota.default(
                6144,
                start=0,
                step=1001,
                dtype=torch.int32,
                device=device(type=GPU_TYPE, index=0),
                requires_grad=False,
            )
            view: "i32[6150144]" = torch.ops.aten.reshape.default(mul, [-1])
            view_1: "f32[6150144]" = torch.ops.aten.reshape.default(div, [-1])
            _embedding_bag = torch.ops.aten._embedding_bag.default(
                primals_3, view, iota_1, False, 0, False, view_1
            )
            getitem: "f32[6144, 64]" = _embedding_bag[0]
            getitem_1: "i32[6150144]" = _embedding_bag[1]
            getitem_2: "i32[6144]" = _embedding_bag[2]
            getitem_3: "i32[0]" = _embedding_bag[3]
            unsqueeze: "f32[6144, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem, 1)
            var_mean = torch.ops.aten.var_mean.correction(
                index, [1], correction=0, keepdim=True
            )
            getitem_4: "f32[6144, 1]" = var_mean[0]
            getitem_5: "f32[6144, 1]" = var_mean[1]
            add: "f32[6144, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
            rsqrt: "f32[6144, 1]" = torch.ops.aten.rsqrt.default(add)
            sub_1: "f32[6144, 4190]" = torch.ops.aten.sub.Tensor(index, getitem_5)
            mul_1: "f32[6144, 4190]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt)
            mul_2: "f32[6144, 4190]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
            add_1: "f32[6144, 4190]" = torch.ops.aten.add.Tensor(mul_2, primals_5)
            permute_1: "f32[4190, 1739]" = torch.ops.aten.permute.default(
                primals_6, [1, 0]
            )

            return [
                index,
                index_1,
                addmm,
                amax,
                sum_1,
                iota_1,
                view,
                view_1,
                getitem_1,
                getitem_2,
                getitem_3,
                unsqueeze,
                getitem_5,
                rsqrt,
                add_1,
                permute_1,
            ]

        kwargs = aot_graph_input_parser(forward, device=GPU_TYPE)
        self.common(forward, [], kwargs=kwargs)

    @skip_if_gpu_halide
    @config.patch("halide.scheduler_cpu", "Mullapudi2016")
    def test_misaligned_address_issue1(self):
        def forward(sub_tensor_1, unsqueeze_default):
            gather_default = torch.ops.aten.gather.default(
                sub_tensor_1, 1, unsqueeze_default
            )
            return gather_default

        args = [
            ((1, 1000), (1000, 1), torch.float32),
            ((1, 1), (1, 1), torch.int64),
        ]
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        self.common(forward, args)

    def test_invalid_operand_issue1(self):
        def forward(arg0_1, arg1_1, arg3_1, squeeze, view_1, slice_1):
            slice_scatter = torch.ops.aten.slice_scatter.default(
                slice_1, arg3_1, 1, 1, 9223372036854775807
            )
            slice_scatter_1 = torch.ops.aten.slice_scatter.default(
                arg1_1, slice_scatter, 0, 0, 9223372036854775807
            )
            slice_2 = torch.ops.aten.slice.Tensor(
                slice_scatter_1, 0, 0, 9223372036854775807
            )
            select_scatter = torch.ops.aten.select_scatter.default(
                slice_2, squeeze, 1, 0
            )
            slice_scatter_2 = torch.ops.aten.slice_scatter.default(
                slice_scatter_1, select_scatter, 0, 0, 9223372036854775807
            )
            view = torch.ops.aten.view.default(slice_scatter_2, [-1, 128])
            embedding = torch.ops.aten.embedding.default(arg0_1, view, 1)
            return [embedding, view_1]

        args = [
            ((50005, 768), (768, 1), torch.float32),
            ((8, 128), (128, 1), torch.int64),
            ((8, 127), (127, 1), torch.int64),
            ((8,), (1,), torch.int64),
            ((1024,), (1,), torch.int64),
            ((8, 128), (128, 1), torch.int64),
        ]
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        self.common(forward, args)

    def test_sizehint_issue1(self):
        def forward(x):
            return torch.nn.functional.unfold(
                x, kernel_size=[4, 4], dilation=1, padding=0, stride=[4, 4]
            )

        args = [((2, 24, 56, 56), (75264, 3136, 56, 1), torch.float32, False)]
        args = [
            rand_strided(sh, st, dt).requires_grad_(rg) for (sh, st, dt, rg) in args
        ]
        self.common(forward, args)

    @skipIfWindows
    def test_zero_dim_reductions(self):
        for kd in [True, False]:
            inps0 = (torch.zeros(2, 0, device=self.device, dtype=torch.float16), 1, kd)
            failed_ops = [aten.argmin, aten.argmax, aten.max, aten.min]
            for fo in failed_ops:
                with self.assertRaisesRegex(
                    IndexError, "Expected reduction dim 1 to have non-zero size"
                ):
                    mod = make_fx(fo)(*inps0)
                    _ = compile_fx_inner(mod, inps0)

            pass_ops = [
                lambda *x: fn(*x) for fn in [aten.sum, aten.prod, aten.any, aten.all]
            ]
            for po in pass_ops:
                compiled = torch._dynamo.optimize("inductor")(po)
                expected = po(*inps0)
                actual = compiled(*inps0)

            self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))

    def test_unfold_zero_dimension_tensor(self):
        def forward(x):
            return torch.unfold_copy(dimension=1, input=x, size=0, step=7)

        x = torch.rand([1, 0], dtype=torch.float32)

        y = forward(x)
        compiled_y = torch.compile(forward, fullgraph=True)(x)

        self.assertEqual(y, compiled_y)

    def test_zero_element_mutation(self):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.LeakyReLU(negative_slope=5.2955089, inplace=True)

            def forward(self, inputs):
                return self.layer1(inputs)

        ip_size = [0]
        input_tensor = torch.randn(ip_size)

        mymodel = CustomModel()
        self.common(mymodel, (input_tensor,))

    def test_lerp(self):
        # non-contiguous inputs for lerp
        def fn0(i0, i1):
            x1 = i0.transpose(-2, -3)
            return torch.lerp(i1, x1, 70000)

        # contiguous inputs for lerp
        def fn1(i0, i1):
            return torch.lerp(i1, i0, 70000)

        self.common(fn0, [torch.rand(10, 3, 10), torch.rand(3, 10, 10)])
        self.common(fn1, [torch.rand(3, 10, 10), torch.rand(3, 10, 10)])

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8318
    def test_unspec_inputs(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Testing mixed devices")

        def fn(x, y):
            return x + y, x * y, x / y

        opt = torch._dynamo.optimize("inductor")(fn)
        dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]

        for d in dtypes:
            inputs = (
                rand_strided((2, 3), (3, 1), dtype=torch.float32, device=GPU_TYPE),
                rand_strided((), (), dtype=d, device="cpu"),
            )
            self.assertTrue(same(opt(*inputs), fn(*inputs)))
            inputs = (inputs[1], inputs[0])
            self.assertTrue(same(opt(*inputs), fn(*inputs)))

    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_list_clearing(self):
        if self.device == "cpu":
            contexts = [contextlib.nullcontext]
        else:
            contexts = [
                contextlib.nullcontext,
                lambda: config.patch({"triton.cudagraphs": True}),
            ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

                def fn(x, y):
                    a = x + y
                    return (a @ a,)

                fn_fx = make_fx(fn)(inps[0], inps[1])
                fn_compiled = compile_fx_inner(fn_fx, inps)

                test_self = self
                matmul_seen = False

                class TestRefMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}

                        nonlocal inps
                        nonlocal inp_refs
                        nonlocal test_self
                        nonlocal matmul_seen

                        # by matmul, inputs should be deallocated
                        # TODO: should not be necessary, ref-cycle ?
                        gc.collect()
                        if func is aten.mm.out:
                            matmul_seen = True
                            test_self.assertEqual(len(inps), 0)
                            test_self.assertIsNone(inp_refs[0]())
                            test_self.assertIsNone(inp_refs[1]())

                        return func(*args, **kwargs)

                with TestRefMode():
                    fn_compiled(inps)

                # do an extra run to make sure we are deallocating on warmup and record
                if self.device == GPU_TYPE:
                    inps.extend(
                        [
                            torch.rand([5, 5]).to(self.device),
                            torch.rand([5, 5]).to(self.device),
                        ]
                    )
                    inp_refs.extend([weakref.ref(inp) for inp in inps])
                    matmul_seen = False

                    with TestRefMode():
                        fn_compiled(inps)

                # for some reason, TorchDispatch doesnt capture the
                # cuda mm call (even without cudagraphs)
                if self.device == "cpu":
                    self.assertTrue(matmul_seen)
                else:
                    self.assertEqual(len(inps), 0)

    def test_dtype_mismatch_issue(self):
        def fn(x):
            attn = torch.nn.functional.pad(x, [0, 1])
            return attn.softmax(dim=-1)

        x = torch.rand(128, 32, 63)
        self.common(fn, (x,))

    def test_vectorized_ops_masked(self):
        def fn(x):
            index = torch.arange(64, device=x.device)
            mask = index.view(1, 1, 64) < 63
            indices = [None, None, index]
            return torch.ops.aten._unsafe_masked_index(x, mask, indices, 7)

        x = torch.rand(128, 32, 63)
        self.common(fn, (x,))

    def test_vectorized_ops_masked_var_novec(self):
        def fn(x):
            index = torch.arange(10, device=x.device)
            mask = (index < 5).view(1, 1, 1, 10)
            indices = [None, None, None, index]
            return torch.ops.aten._unsafe_masked_index(x, mask, indices, 7)

        x = torch.rand(1, 1, 8, 8)
        self.common(fn, (x,))

    def test_diagonal_copy(self):
        def fn(x):
            return torch.diagonal_copy(x)

        for x in (torch.randn(2, 3), torch.randn(2, 2), torch.randn(3, 2)):
            self.common(fn, (x,))

    def test_kwargs(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("histogramdd only supports cpu")

        def fn(x, y):
            return torch.histogramdd(
                x,
                bins=[3, 3],
                weight=y,
            )

        self.common(
            fn,
            [torch.randn((4, 2)), torch.randn(4)],
        )

    # Shape padding causes the inputs to all get specialized, so the codegen
    # test fails
    @expectedFailureCodegenDynamic
    @requires_gpu()
    @torch._inductor.config.patch("shape_padding", True)
    def test_shape_padding(self):
        dtypes = [
            torch.float16,
            torch.float32,
        ]

        b, m, n, k = 7, 11, 13, 15

        def gen(*shape, dtype=torch.float32):
            return torch.randn(*shape, device=GPU_TYPE, dtype=dtype) / k + 1.0

        for dtype in dtypes:
            x = gen(m, k, dtype=dtype)
            y = gen(k, n, dtype=dtype)
            z = gen(n, dtype=dtype)
            self.common(lambda x, y: torch.mm(x, y), (x, y))
            self.common(lambda x, y: torch.matmul(x, y), (x, y))
            self.common(lambda x, y, z: torch.addmm(z, x, y), (x, y, z))

        for dtype in dtypes:
            x = gen(b, m, k, dtype=dtype)
            y = gen(b, k, n, dtype=dtype)
            z = gen(n, dtype=dtype)
            self.common(lambda x, y: torch.bmm(x, y), (x, y))
            self.common(lambda x, y: torch.matmul(x, y), (x, y))
            self.common(lambda x, y, z: torch.baddbmm(z, x, y), (x, y, z))

    @requires_gpu()
    @torch._inductor.config.patch("layout_optimization", True)
    def test_inductor_layout_optimization_input_mutations(self):
        # channel dim must be > 64 for inductor to do layout optimization and use NHWC
        mod = nn.Conv2d(3, 128, 1, stride=1, bias=False).to(GPU_TYPE)

        def f(x):
            x.mul_(2)
            out = mod(x)
            return out

        f_compiled = torch.compile(f)
        x_ref = torch.rand(2, 3, 128, 128, device=GPU_TYPE)
        x_test = x_ref.clone().detach()
        with torch.no_grad():
            out_ref = f(x_ref)
            out_test = f_compiled(x_test)
            self.assertEqual(out_ref, out_test)
            self.assertEqual(out_ref.shape, out_test.shape)
            # Importantly, since inductor._config.keep_output_stride is True,
            # the outputs should have matching strides here.
            self.assertEqual(out_ref.stride(), out_test.stride())
            self.assertEqual(x_ref, x_test)

    def test_int_input_dynamic_shapes(self):
        @torch.compile(dynamic=True)
        def fn(x, i):
            y = x * i
            return y

        # Constant must not get matched as constant
        self.common(fn, [torch.randn(3, 1, 1, 1, 1), 9132])

    def test_sqrt_dynamic_shapes(self):
        # TIMM convit_base model: https://github.com/pytorch/pytorch/issues/97877.
        # TODO: support cuda path.
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("sqrt dynamic shapes only supports cpu")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                B, N, C = x.shape
                return self.get_rel_indices(N)

            def get_rel_indices(self, num_patches: int) -> torch.Tensor:
                img_size = int(num_patches**0.5)
                ind = torch.arange(img_size)
                return ind

        self.common(
            Model(),
            [
                torch.randn(8, 4, 4),
            ],
        )

    def test_rsqrt_dynamic_shapes(self):
        # From HF hf_BigBird model.
        @torch.compile(dynamic=True)
        def fn(a, b):
            r = 1 / math.sqrt(a.size(1))
            return torch.bmm(a, b) / r

        self.common(
            fn,
            [
                torch.randn(2, 4, 4),
                torch.randn(2, 4, 4),
            ],
        )

    def test_index_dynamic_shapes(self):
        # Repro from vision_maskrcnn
        def fn(arg0_1):
            unsqueeze = arg0_1.unsqueeze(0)
            sym_size = arg0_1.size(1)
            ceil = math.ceil(sym_size * 1.8735363483428955)
            iota = torch.ops.prims.iota.default(
                ceil,
                start=0,
                step=1,
                dtype=torch.int64,
                device=arg0_1.device,
                requires_grad=False,
            )
            convert_element_type_1 = iota.to(torch.float32)
            sym_size_1 = arg0_1.size(2)
            floor_1 = math.floor(sym_size_1 * 1.8735363483428955)
            ceil_1 = math.ceil(floor_1)
            iota_1 = torch.ops.prims.iota.default(
                ceil_1,
                start=0,
                step=1,
                dtype=torch.int64,
                device=arg0_1.device,
                requires_grad=False,
            )
            convert_element_type_3 = iota_1.to(torch.float32)
            sub_2 = (convert_element_type_1 + 0.5) * (sym_size / ceil) - 0.5
            clamp_min = sub_2.clamp_min(0.0)
            sub_3 = (convert_element_type_3 + 0.5) * (sym_size_1 / floor_1) - 0.5
            clamp_min_1 = sub_3.clamp_min(0.0)
            convert_element_type_4 = clamp_min.to(torch.int64)
            sub_4 = sym_size - 1
            clamp_max = clamp_min.ceil().clamp_max(sub_4)
            convert_element_type_5 = clamp_max.to(torch.int64)
            convert_element_type_6 = clamp_min_1.to(torch.int64)
            unsqueeze_2 = convert_element_type_4.unsqueeze(1)
            index = torch.ops.aten.index.Tensor(
                unsqueeze, [None, None, unsqueeze_2, convert_element_type_6]
            )
            index_1 = torch.ops.aten.index.Tensor(
                unsqueeze,
                [
                    None,
                    None,
                    convert_element_type_5.unsqueeze(1),
                    convert_element_type_6,
                ],
            )
            sub_6 = clamp_min.unsqueeze(1) - unsqueeze_2
            mul_10 = (index * (1.0 - sub_6) + index_1 * (sub_6)) * (
                1.0 - (clamp_min_1 - convert_element_type_6)
            )
            select = torch.ops.aten.select.int(mul_10, 0, 0)
            return (select,)

        x = torch.randn(15, 20, 3)
        self.common(
            fn,
            [x],
        )

    @skipIfWindows
    def test_setitem_with_int_parameter(self):
        x = torch.zeros(7, device=self.device)

        def fn(n, a):
            a[n] = -1
            return a

        cnts = CompileCounterWithBackend("inductor")
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        for n in range(2, x.shape[0]):
            opt_fn(n, x)
            self.assertEqual(x[n], -1)

        # If assume_static_by_default is set, the calls above will trigger
        # 3 function compilation:
        #   1. assuming 'n' is static (equals 2)
        #   2. making 'n' dynamic, but with the guard 'end <= x.shape[0]'
        #      (from: torch._inductor.ir.SliceView.create)
        frame_count = 2 if torch._dynamo.config.assume_static_by_default else 1
        self.assertEqual(cnts.frame_count, frame_count)

        # Negative index triggers new compilation.
        opt_fn(-x.shape[0], x)
        self.assertEqual(x[0], -1)
        self.assertEqual(cnts.frame_count, frame_count + 1)

    @config.patch(profiler_mark_wrapper_call=True)
    @skipIfWindows
    def test_profiler_mark_wrapper_call(self):
        from torch.profiler import profile

        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(a, b):
            return a + b

        a = torch.rand((100,))
        b = torch.rand((100,))
        with profile() as prof:
            fn(a, b)
        assert any(
            "inductor_wrapper_call" in e.name for e in prof.profiler.function_events
        )

    def test_insignificant_strides(self):
        def f(x):
            tmp = x + 1
            return tmp.view(-1, 1, 2)

        x = torch.arange(8, device=self.device, dtype=torch.float32)
        out = f(x)
        compiled_out = torch.compile(f)(x)

        self.assertEqual(out.stride(), compiled_out.stride())
        self.assertEqual(out, compiled_out)

    @unittest.skipIf(IS_X86 and not HAS_AVX2, "Requires AVX2")
    def test_pixel_shuffle_channels_last(self):
        def fn(x):
            x = torch.nn.functional.pixel_shuffle(x, 2)
            x = torch.nn.functional.relu(x)
            return x

        self.common(
            fn,
            (torch.randn(1, 16, 64, 72).to(memory_format=torch.channels_last),),
        )

    def test_where_broadcast(self):
        # https://github.com/pytorch/pytorch/issues/93374
        def fn(x, p1, p0):
            o = torch.where(x, p1, p0)
            return o

        # https://github.com/pytorch/pytorch/issues/94725
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "_tensor_constant0", torch.randn([], dtype=torch.float32)
                )

            def forward(self, arg0_1, arg1_1):
                convert_element_type = torch.ops.prims.convert_element_type.default(
                    arg1_1, torch.bool
                )
                bitwise_not = torch.ops.aten.bitwise_not.default(convert_element_type)
                _tensor_constant0 = self._tensor_constant0
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(
                    _tensor_constant0
                )
                where = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy, arg0_1)
                return (where, bitwise_not)

        self.common(
            fn,
            (torch.tensor([[True]]), torch.rand(13, 7, 3), torch.rand(1, 1)),
        )

        args = [
            torch.randn(1, 4, 64, 64),
            torch.zeros(1, 1, 64, 64, dtype=torch.uint8),
        ]
        args[1][:, :, :32, :32] = 1
        eager_args = [x.clone() for x in args]
        eager_mod = Repro()
        mod = make_fx(eager_mod, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        inductor_out = compiled(args)
        eager_out = eager_mod(*eager_args)
        self.assertEqual(inductor_out, eager_out)

    @skipIfRocm
    def test_require_stride_expanded(self):
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16.unsqueeze(0), arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            return (convolution,)

        self.common(
            forward,
            (
                None,
                rand_strided(
                    (64, 3, 11, 11),
                    (363, 121, 11, 1),
                    torch.float32,
                    device=self.device,
                ).to(memory_format=torch.channels_last),
                rand_strided(
                    (1, 3, 224, 224),
                    (150528, 50176, 224, 1),
                    torch.float32,
                    device=self.device,
                )
                .to(memory_format=torch.channels_last)
                .squeeze(0),
            ),
            atol=1e-3,
            rtol=0.001,
        )

        # expanded dim should not cause copy in require_stride_order
        assertGeneratedKernelCountEqual(self, 0)

    @requires_gpu()
    @parametrize("use_block_ptr", (False, True))
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @skipIfRocm
    def test_sdpa(self, use_block_ptr):
        def foo(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
            view = torch.ops.aten.view.default(arg3_1, [23760, 128])
            arg3_1 = None
            mm = torch.ops.aten.mm.default(view, arg4_1)
            view = arg4_1 = None
            view_1 = torch.ops.aten.view.default(mm, [3, 99, 80, 8])
            mm = None
            view_2 = torch.ops.aten.view.default(view_1, [3, 99, 80, 8])
            view_1 = None
            permute = torch.ops.aten.permute.default(view_2, [0, 3, 1, 2])
            view_2 = None
            view_3 = torch.ops.aten.view.default(permute, [3, 8, 99, 80])
            permute = None

            clone = torch.ops.aten.clone.default(
                view_3, memory_format=torch.contiguous_format
            )
            view_3 = None

            expand = torch.ops.aten.expand.default(clone, [3, 8, 99, 80])
            clone = None
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    arg0_1, arg1_1, arg2_1, expand, False
                )
            )
            arg0_1 = arg1_1 = arg2_1 = expand = None
            getitem = _scaled_dot_product_efficient_attention[0]
            _scaled_dot_product_efficient_attention = None
            return (getitem,)

        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        DEVICE = torch.device(f"{GPU_TYPE}:0")
        DTYPE = torch.float16
        B = 3
        H = 8
        Q = 99
        K = 80
        D = 32
        C_bias = 128

        # inputs
        query = torch.randn((B, H, Q, D), device=DEVICE, dtype=DTYPE)
        key = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
        value = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
        bias = torch.randn((B, Q, K, C_bias), device=DEVICE, dtype=DTYPE)
        weights = torch.randn((C_bias, H), device=DEVICE, dtype=DTYPE)
        inps = (query, key, value, bias, weights)

        with config.patch("triton.use_block_ptr", use_block_ptr):
            # Check accuracy
            self.common(
                foo,
                inps,
                atol=0.02,
                rtol=1e4,
            )

            # Check code for block pointers
            foo_opt = torch._dynamo.optimize("inductor")(foo)
            code = run_and_get_triton_code(foo_opt, *inps)
            have_block_ptr = code.count("tl.make_block_ptr") > 0
            if not is_halide_backend(self.device):
                self.assertEqual(have_block_ptr, use_block_ptr)

    @requires_gpu()
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support mem_eff_attention",
    )
    def test_sdpa_unaligned_mask(self):
        def foo(
            arg0_1: "f32[8, 8, 16, 16]",
            arg1_1: "f32[8, 8, 15, 16]",
            arg2_1: "f32[8, 8, 15, 16]",
            arg3_1: "f32[1, 1, 16, 15]",
        ):
            constant_pad_nd: "f32[1, 1, 16, 16]" = (
                torch.ops.aten.constant_pad_nd.default(arg3_1, [0, 1], 0.0)
            )
            arg3_1 = None
            slice_1: "f32[1, 1, 16, 15]" = torch.ops.aten.slice.Tensor(
                constant_pad_nd, -1, 0, 15
            )
            constant_pad_nd = None
            expand: "f32[8, 8, 16, 15]" = torch.ops.aten.expand.default(
                slice_1, [8, 8, 16, 15]
            )
            slice_1 = None
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    arg0_1, arg1_1, arg2_1, expand, False
                )
            )
            arg0_1 = arg1_1 = arg2_1 = expand = None
            getitem: "f32[8, 8, 16, 16]" = _scaled_dot_product_efficient_attention[0]
            _scaled_dot_product_efficient_attention = None
            return (getitem,)

        query = torch.rand(8, 8, 16, 16, device=GPU_TYPE)
        key = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        value = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        bias = torch.rand(1, 1, 16, 15, device=GPU_TYPE)
        self.common(
            foo,
            (query, key, value, bias),
            atol=0.02,
            rtol=1e4,
        )

    @requires_gpu()
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support mem_eff_attention",
    )
    @config.patch(freezing=True)
    def test_sdpa_unaligned_mask_freezing(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.arg3_1 = torch.rand(1, 1, 16, 15, device=GPU_TYPE)

            def forward(
                self,
                arg0_1: "f32[8, 8, 16, 16]",
                arg1_1: "f32[8, 8, 15, 16]",
                arg2_1: "f32[8, 8, 15, 16]",
            ):
                arg3_1 = self.arg3_1
                constant_pad_nd: "f32[1, 1, 16, 16]" = (
                    torch.ops.aten.constant_pad_nd.default(arg3_1, [0, 1], 0.0)
                )
                arg3_1 = None
                slice_1: "f32[1, 1, 16, 15]" = torch.ops.aten.slice.Tensor(
                    constant_pad_nd, -1, 0, 15
                )
                constant_pad_nd = None
                expand: "f32[8, 8, 16, 15]" = torch.ops.aten.expand.default(
                    slice_1, [8, 8, 16, 15]
                )
                slice_1 = None
                _scaled_dot_product_efficient_attention = (
                    torch.ops.aten._scaled_dot_product_efficient_attention.default(
                        arg0_1, arg1_1, arg2_1, expand, False
                    )
                )
                arg0_1 = arg1_1 = arg2_1 = expand = None
                getitem: "f32[8, 8, 16, 16]" = _scaled_dot_product_efficient_attention[
                    0
                ]
                _scaled_dot_product_efficient_attention = None
                return (getitem,)

        query = torch.rand(8, 8, 16, 16, device=GPU_TYPE)
        key = torch.rand(8, 8, 15, 16, device=GPU_TYPE)
        value = torch.rand(8, 8, 15, 16, device=GPU_TYPE)

        mod = Mod()
        out_eager = mod(query, key, value)

        with torch.no_grad():
            out_compiled = torch.compile(mod)(query, key, value)
            self.assertEqual(out_eager, out_compiled, atol=0.02, rtol=1e4)

    def test_where_with_logical_op(self):
        def fn_and(x, y):
            return torch.where(torch.logical_and(x, y), 1.0, 0.0)

        def fn_or(x, y):
            return torch.where(torch.logical_or(x, y), 1.0, 0.0)

        self.common(
            fn_and,
            (torch.randn(32), torch.randn(32)),
        )
        self.common(
            fn_or,
            (torch.randn(32), torch.randn(32)),
        )

    @skipIfRocm
    def test_conv_with_as_strided(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv = torch.nn.Conv2d(
                    256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
                )

            def forward(self, x):
                convolution = self.kv(x)
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # as_strided inputs are depend on input's size and stide.
                as_strided = torch.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                as_strided_1 = torch.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                clone = torch.ops.aten.clone.default(
                    as_strided_1, memory_format=torch.contiguous_format
                )
                return clone

        self.common(
            Model(),
            (torch.randn(8, 256, 16, 16),),
            check_lowp=not is_halide_backend(self.device),
        )

    def test_inplace_where_pointwise(self):
        # https://github.com/pytorch/pytorch/issues/96446
        def fn(a, b):
            a[0] = 2
            return a * b

        self.common(fn, (torch.rand(1), torch.rand(2)))

    def test_view_on_aliased(self):
        # https://github.com/pytorch/pytorch/issues/96728
        def fn1(a, b):
            a = a.max(0).values
            c = torch.cat((a, b))
            c = c.round()
            b >= a[0]  # noqa: B015
            return c

        some_const = torch.tensor(6324)

        def fn2():
            a = torch.tensor([[0.6324]])
            ret = torch.cat((a, a), dim=0)
            some_const >= a[0]  # noqa: B015
            return ret

        self.common(fn1, (torch.tensor([[4.0]]), torch.tensor([5.0])))
        self.common(fn2, ())

    def test_argmax_to_float(self):
        # https://github.com/pytorch/pytorch/issues/97127
        def fn():
            a = torch.zeros([2, 2])
            b = a.argmax(0)
            return b.float().mean()

        self.common(fn, ())

    def test_const_int32_to_float(self):
        # https://github.com/pytorch/pytorch/issues/97124
        def fn():
            a = torch.zeros([1, 2], dtype=torch.int32)
            a = a + a
            b = a.to(dtype=torch.float32)
            return b * 0.8

        self.common(fn, ())

    def test_getitem(self):
        out_features = ["p3", "p4", "p5", "p6", "p7"]
        in_feature = "p5"

        def fn(a):
            return a[out_features.index(in_feature)]

        x = [
            torch.rand([1, 256, 100, 152], device=self.device),
            torch.rand([1, 256, 50, 76], device=self.device),
            torch.rand([1, 256, 25, 38], device=self.device),
        ]
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        same(fn(x), opt_fn(x))

    def test_pad_view(self):
        def fn(a):
            y = torch.nn.functional.pad(a, (0, 0, 0, 1))
            y = y.view(*y.size()[:-2], y.size(-1), y.size(-2))
            return y

        x = torch.rand(48, 3, 512, 512)
        self.common(fn, (x,))

    def test_pad_cast(self):
        def fn(x):
            return torch.nn.functional.pad(x.to(torch.float32), (0, 3, 0, 0))

        for dtype in [torch.int32, torch.int64]:
            self.common(fn, (torch.ones(1, 1, 13, dtype=dtype),))

    @unittest.skipIf(not HAS_CPU, "requires C++ compiler")
    @skip_if_halide  # bf16
    def test_data_type_propogation(self):
        from torch._dynamo.utils import detect_fake_mode
        from torch._inductor.codegen.common import boolean_ops
        from torch._inductor.compile_fx import shape_env_from_inputs
        from torch._inductor.debug import DebugContext
        from torch._inductor.decomposition import decompositions
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        def get_data_type(node: torch.fx.Node):
            if OptimizationContext.key in node.meta:
                return node.meta[OptimizationContext.key].dtype
            else:
                return None

        def func(arg0_1):
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(
                arg0_1, [3, 3], [2, 2], [1, 1]
            )
            arg0_1 = None
            getitem = max_pool2d_with_indices[0]
            max_pool2d_with_indices = None
            return (getitem,)

        example_inputs = [
            torch.randn(10, 32, 20, 20, dtype=torch.bfloat16).to(
                memory_format=torch.channels_last
            )
        ]

        gm = make_fx(func, decomposition_table=decompositions, tracing_mode="fake")(
            *example_inputs
        )

        shape_env = shape_env_from_inputs(example_inputs)

        fake_mode = detect_fake_mode(example_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
        else:
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                *example_inputs
            )
        with V.set_fake_mode(fake_mode):
            graph = GraphLowering(
                gm,
                shape_env=shape_env,
            )
            with V.set_graph_handler(graph), V.set_debug_handler(DebugContext()):
                graph.run(*example_inputs)
                graph.compile_to_module()
                scheduler_node = graph.scheduler.nodes[0]
                DataTypePropagation.propagate_scheduler_node(scheduler_node)
                root_graph = scheduler_node._body.root_block.graph
                for node in root_graph.nodes:
                    if node.op == "placeholder":
                        self.assertEqual(get_data_type(node), None)
                    elif node.target in boolean_ops():
                        self.assertEqual(get_data_type(node), torch.bool)
                    elif node.target in (
                        "constant",
                        "to_dtype",
                        "index_expr",
                    ):
                        self.assertEqual(get_data_type(node), node.args[-1])
                    elif node.target in (
                        "get_index",
                        "index_expr",
                    ):
                        self.assertEqual(get_data_type(node), torch.int64)
                    elif node.target in (
                        "load",
                        "store",
                    ):
                        self.assertEqual(
                            get_data_type(node), V.graph.get_dtype(node.args[1])
                        )
                    elif node.target == "reduction":
                        _, _, dtype, _, _, _, _ = node.args
                        self.assertEqual(get_data_type(node), dtype)
                    elif node.target.startswith("masked_subblock"):
                        """
                        masked_subblocks:
                        opcode       name       target     args                        kwargs
                        -----------  ---------  ---------  --------------------------  --------
                        placeholder  ops        ops        ()                          {}
                        call_module  get_index  get_index  ('index2',)                 {}
                        call_method  load       load       (ops, 'arg0_1', get_index)  {}
                        call_method  to_dtype   to_dtype   (ops, load, torch.float32)  {}
                        output       output     output     (to_dtype,)                 {}
                        """
                        self.assertEqual(get_data_type(node), torch.float)
                    elif node.target == "and_":
                        """
                        and_'s input is boolean_ops:
                        -----------  ---------  ---------  --------------------------  --------
                        call_method  and__22           and_              (ops, ge_15, lt_15)
                        -----------  ---------  ---------  --------------------------  --------
                        """
                        self.assertEqual(get_data_type(node), torch.bool)
                    elif node.target == "maximum":
                        """
                        maximum's input is maximum or masked_subblock:
                        -----------  ---------  ---------  --------------------------  --------
                        call_method  maximum_6         maximum           (ops, masked_subblock8, maximum_5)
                        -----------  ---------  ---------  --------------------------  --------
                        """
                        self.assertEqual(get_data_type(node), torch.float)
                    elif node.target == "output":
                        self.assertEqual(get_data_type(node), torch.bfloat16)

    # Calling div only torch.SymInt arguments is not yet supported.
    # To support this behavior, we need to allow const-propping tensors that store symint data.
    # For now, dynamo will explicitly graph break when it encounters user code with this behavior.
    @expectedFailureCodegenDynamic
    @skip_if_gpu_halide  # accuracy error
    @skipIfWindows  # TODO: need to fix
    def test_AllenaiLongformerBase_repro(self):
        def fn(query, scores, window_overlap):
            batch_size, seq_len, num_heads, _ = query.size()
            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            diagonal_attention_scores = scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            diagonal_attention_scores[:, :-1, :, window_overlap:] = scores[
                :, :, :window_overlap, : window_overlap + 1
            ]
            input_tensor = diagonal_attention_scores.view(
                batch_size, num_heads, seq_len, 2 * window_overlap + 1
            ).transpose(2, 1)
            beginning_input = input_tensor[:, :window_overlap, :, : window_overlap + 1]
            input_tensor[:, :window_overlap, :, : window_overlap + 1] = torch.full_like(
                beginning_input, -float("inf")
            )
            return input_tensor

        args = [
            ((4, 1024, 12, 64), (768, 3072, 64, 1)),
            ((48, 3, 512, 513), (787968, 262656, 513, 1)),
        ]
        args = [rand_strided(sh, st) for (sh, st) in args]
        args.append(256)

        if is_cpp_backend(self.device):
            opt_fn = torch._dynamo.optimize("inductor")(fn)
            _, code = run_and_get_cpp_code(opt_fn, *args)
            FileCheck().check_count(
                "static_cast<int64_t>(256)",
                2,
                exactly=True,
            ).run(code)

        self.common(fn, args)

    def test_cumsum_pattern_matcher_issue(self):
        def fn(input_ids) -> torch.Tensor:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size, seq_length = input_shape
            past_key_values_length = 0
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=input_ids.device
            )
            attention_mask = attention_mask.long()
            return torch.cumsum(attention_mask, dim=1)

        x = torch.randn(2, 2)
        self.common(fn, (x,), atol=0, rtol=0)

    @staticmethod
    def _check_resize_common(
        self, fn, x, size_or_y, memory_format, inplace, deterministic
    ):
        x = x.to(self.device)
        x_ref_arg = x.clone()
        x_opt_arg = x.clone()
        x_numel = x.numel()
        torch._dynamo.reset_code_caches()
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        correct = fn(x_ref_arg, size_or_y, memory_format)
        actual = opt_fn(x_opt_arg, size_or_y, memory_format)

        def get_numel(size_or_y):
            if isinstance(size_or_y, torch.Tensor):
                return size_or_y.numel()
            else:
                # assume shape
                return functools.reduce(lambda x, y: x * y, size_or_y, 1)

        if deterministic:
            nele_check = correct.numel()
        else:
            nele_check = min(x_numel, get_numel(size_or_y))

        correct_values = correct.as_strided((nele_check,), (1,))
        actual_values = actual.as_strided((nele_check,), (1,))
        self.assertTrue(same(correct_values, actual_values, equal_nan=deterministic))
        correct_strides = correct.stride()
        actual_strides = actual.stride()
        self.assertEqual(correct_strides, actual_strides)

    @staticmethod
    def _cases_resize_common():
        sizes = [
            ((2,), (1, 3, 2, 3)),
            ((100,), (1, 3, 2, 3)),
            ((1, 3, 2, 3), (1, 3, 2, 3)),
            ((2,), (1, 3, 2, 3, 1)),
            ((100,), (1, 3, 2, 3, 1)),
            ((1, 3, 2, 3, 1), (1, 3, 2, 3, 1)),
            ((2, 0, 1), (2, 2)),
        ]
        for x_size, y_size in sizes:
            memory_formats = [torch.contiguous_format]
            if len(y_size) == 4:
                memory_formats.append(torch.channels_last)
            if len(y_size) == 5:
                memory_formats.append(torch.channels_last_3d)
            for memory_format in memory_formats:
                x = torch.randn(*x_size)
                yield x, y_size, memory_format
                # check some non-contiguous tensors
                if x.numel() == 100:
                    x_strided = x[::2].reshape(25, 2).transpose(0, 1)
                    yield x_strided, y_size, memory_format

    def test_resize(self):
        def fn(x, size, memory_format):
            # NOTE: Tensor.resize() =/= aten::resize()
            return torch.ops.aten.resize(x, size, memory_format=memory_format)

        for deterministic in [True, False]:
            with DeterministicGuard(
                deterministic, fill_uninitialized_memory=deterministic
            ):
                for x, y_size, memory_format in CommonTemplate._cases_resize_common():
                    CommonTemplate._check_resize_common(
                        self,
                        fn,
                        x,
                        y_size,
                        memory_format,
                        inplace=False,
                        deterministic=deterministic,
                    )

    @staticmethod
    def _cases_resize_as_common():
        for x, y_size, memory_format in CommonTemplate._cases_resize_common():
            # each sizes /memory_format combintation tested in 2 ways:
            # 1. y is contiguous fn gets memory_format kwargs
            # 2. y has memory_format contiguity and fn gets preserve kwarg
            # 3. y has some other strides (not contiguous or channels last) and fn gets preserve
            yield x, torch.randn(*y_size), memory_format
            yield x, torch.randn(*y_size).contiguous(
                memory_format=memory_format
            ), torch.preserve_format
            yield x, torch.randn(*y_size).permute(
                tuple(reversed(range(len(y_size))))
            ), torch.preserve_format

    @skipIfXpu
    def test_resize_as(self):
        def fn(x, y, memory_format):
            return torch.ops.aten.resize_as(x, y, memory_format=memory_format)

        for deterministic in [True, False]:
            with DeterministicGuard(
                deterministic, fill_uninitialized_memory=deterministic
            ):
                for x, y, memory_format in CommonTemplate._cases_resize_as_common():
                    CommonTemplate._check_resize_common(
                        self,
                        fn,
                        x,
                        y,
                        memory_format,
                        inplace=False,
                        deterministic=deterministic,
                    )

    def test_inplace_resize_as(self):
        def fn(x, y):
            x.resize_as_(y)
            return x

        x = torch.randn(2, 3)
        y = torch.randn(200, 300)
        x_clone = x.clone()
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        same(fn(x, y), opt_fn(x_clone, y))

    def test_erfc(self):
        def fn(x):
            return torch.erfc(x)

        self.common(fn, (torch.randn(8, 8),))

    @skip_if_halide  # erfinv not implemented
    def test_erfinv(self):
        def fn(x):
            return torch.erfinv(x)

        # domain for erfinv is (-1, 1)
        x = torch.empty(8, 8).uniform_(-1, 1)
        self.common(fn, (x,))

    def test_uint(self):
        def fn(z):
            x = torch.tensor(5, device=z.device, dtype=torch.uint8)
            y = torch.neg(x)
            return x < y

        self.common(fn, (torch.randn(26),))

    def test_scaled_dot_product_attention(self):
        if self.device == "cuda" and not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            raise unittest.SkipTest("Can't run flash attention on this platform")
        if self.device == "cuda" and TEST_WITH_ROCM:
            raise unittest.SkipTest(
                "Flash attention support is incomplete on this platform"
            )

        def fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=0.125,
            )[:2]

        self.common(
            fn,
            (
                torch.randn(4, 2, 4, 2),
                torch.randn(4, 2, 4, 2),
                torch.randn(4, 2, 4, 2),
            ),
            atol=2e-4,  # to pass lowp check on GPU
            rtol=1e-2,  # to pass lowp check on GPU
        )

    @skipIfRocm
    @expectedFailureXPU
    def test_scaled_dot_product_efficient_attention(self):
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # The first two values should be the same, attention output
        # and logsumexp since dropout is not being set
        def fn(q, k, v, attn_bias, compute_log_sumexp):
            return aten._scaled_dot_product_efficient_attention(
                q, k, v, attn_bias, compute_log_sumexp
            )[:2]

        self.common(
            fn,
            (
                torch.randn(4, 4, 36, 36),
                torch.randn(4, 4, 36, 36),
                torch.randn(4, 4, 36, 36),
                torch.randn(4, 4, 36, 36),
                False,
            ),
            check_lowp=False,
        )

    def test_fft_real_input(self):
        def fn(x):
            return torch.fft.fftn(x)

        self.common(fn, (torch.randn((16, 16, 16)),), check_lowp=False)

    def test_fft_real_input_real_output(self):
        def fn(x):
            return torch.fft.fftn(x).real

        self.common(fn, (torch.randn((16, 16, 16)),), check_lowp=False)

    def test_bucketize(self):
        def fn(input, boundaries, out_int32, right):
            return torch.bucketize(input, boundaries, out_int32=out_int32, right=right)

        input = torch.rand((64, 64)) * 2 - 1
        boundaries = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])

        for out_int32 in [True, False]:
            for right in [True, False]:
                out_int32 = True
                right = False
                self.common(fn, (input, boundaries, out_int32, right), check_lowp=False)

    def test_bucketize_default_kwargs(self):
        def fn(input, offsets):
            return torch.bucketize(input, offsets)

        input = torch.tensor(
            [-1.0, -0.9, -0.8, -0.5, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 0.91]
        )
        offsets = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])

        self.common(fn, (input, offsets), check_lowp=False)

    def test_bucketize_int(self):
        def fn(input, offsets, out_int32, right):
            return torch.bucketize(input, offsets, out_int32=out_int32, right=right)

        input = torch.randint(0, 102, (64, 64))
        offsets = torch.arange(10, dtype=torch.int32) ** 2 + 1

        for out_int32 in [True, False]:
            for right in [True, False]:
                self.common(fn, (input, offsets, out_int32, right), check_lowp=False)

    @patch.object(config.triton, "autotune_pointwise", True)
    def test_bucketize_add_autotune(self):
        # Causes a @pointwise(size_hints) where size_hints is 2D

        def fn(input, offsets, add_value):
            return torch.bucketize(input, offsets) + add_value

        input = torch.rand((16, 16, 64, 64))
        boundaries = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9])
        add_value = torch.randint(0, 1024, (16, 16, 64, 64)).to(
            memory_format=torch.channels_last
        )

        self.common(fn, (input, boundaries, add_value), check_lowp=False)

        assertGeneratedKernelCountEqual(self, 1)

    def test_bucketize_computed_offsets(self):
        def fn(inp, offsets):
            return torch.bucketize(inp, offsets + 0.01)

        inp = torch.tensor(
            [-1.0, -0.9, -0.8, -0.5, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 0.91]
        )
        offsets = torch.tensor([-0.9, -0.8, 0.1, 0.2, 0.5, 0.9]) - 0.01

        self.common(fn, (inp, offsets), check_lowp=False)

    @requires_gpu()
    @config.patch(assume_aligned_inputs=False)
    def test_config_option_dont_assume_alignment(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x.cos()

        # Inductor specializes on the (unguarded) alignment of the initial input.
        # Make sure that for different configurations, nothing breaks.
        for offset in (0, 1, 2, 3, 4):
            base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)
            inp = torch.as_strided(base, (64, 64), (64, 1), offset)
            torch._dynamo.reset()
            fn_c = torch.compile(fn)

            ref = fn(inp)
            res = fn_c(inp)
            self.assertEqual(ref, res)

            for offset2 in (0, 1, 2, 3, 4):
                base2 = torch.randn(
                    64 * 64 + 64, dtype=torch.float32, device=self.device
                )
                inp2 = torch.as_strided(base2, (64, 64), (64, 1), offset2)
                ref2 = fn(inp2)
                res2 = fn_c(inp2)
                self.assertEqual(ref2, res2, atol=1e-5, rtol=1e-5)

    @requires_gpu()
    @config.patch(assume_aligned_inputs=False)
    def test_config_option_dont_assume_alignment_recompiles(self):
        # Inputs:
        #  1. (32, 32) shape
        #  2. (64, 64) shape -> causes a recompile
        #  3. (64, 64) shape with different storage offset -> should NOT cause a recompile
        failed_guards = []

        def fail(guard):
            nonlocal failed_guards
            failed_guards.append(guard)

        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x.cos()

        base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)

        inp1 = torch.as_strided(base, (32, 32), (32, 1), 4)
        inp2 = torch.as_strided(base, (64, 64), (64, 1), 4)
        inp3 = torch.as_strided(base, (64, 64), (64, 1), 5)

        torch._dynamo.reset()

        fn_c = torch._dynamo.optimize("inductor", guard_fail_fn=fail)(fn)

        ref1 = fn(inp1)
        res1 = fn_c(inp1)
        self.assertEqual(ref1, res1)
        self.assertEqual(0, len(failed_guards))

        ref2 = fn(inp2)
        res2 = fn_c(inp2)
        self.assertEqual(ref2, res2)
        # if dynamic shapes isn't already turned on, we might have a guard failure as we turn
        # on dynamic shapes
        self.assertLessEqual(len(failed_guards), 1)
        failed_guard_count_iteration_2 = len(failed_guards)

        failed_guards = []
        ref3 = fn(inp3)
        res3 = fn_c(inp3)
        self.assertEqual(ref3, res3)
        # we might still have the dynamics shapes failure, but offset change shouldn't be guarded on
        # see Note: [Input Alignment handling in Inductor]
        self.assertLessEqual(len(failed_guards), failed_guard_count_iteration_2)

    @requires_gpu()
    @config.patch(assume_aligned_inputs=False)
    def test_config_option_dont_assume_alignment_cudagraphs(self):
        def fn(x):
            return x.cos() * x.sin()

        fn_c = torch.compile(fn, mode="reduce-overhead", dynamic=True)

        for size, stride, offset in (
            ((32, 32), (32, 1), 4),
            ((48, 48), (48, 1), 4),
            ((64, 64), (64, 1), 5),
        ):
            torch.manual_seed(42)
            base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=self.device)
            torch.manual_seed(42)
            base_ref = torch.randn(
                64 * 64 + 64, dtype=torch.float32, device=self.device
            )

            inp = torch.as_strided(base, size, stride, offset)
            inp_ref = torch.as_strided(base_ref, size, stride, offset)

            inp.requires_grad_(True)
            inp_ref.requires_grad_(True)

            res = fn_c(inp)
            ref = fn(inp_ref)
            self.assertEqual(ref, res)

            res.sum().backward()
            ref.sum().backward()
            self.assertEqual(base.grad, base_ref.grad)

    @config.patch(implicit_fallbacks=True)
    def test_custom_op_1(self):
        import torch.library

        def foo_cpu(x):
            return 3 * x

        def foo_cuda(x):
            return 3 * x

        def foo_xpu(x):
            return 3 * x

        def foo_meta(x):
            return torch.empty_like(x)

        define_custom_op_for_test("foo", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.foo(a)
            c = torch.cos(b)
            return c

        self.common(fn, (torch.randn((16, 32)),), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_custom_op_2(self):
        import torch.library

        def foo_cpu(x, scale: float):
            return scale * x, torch.cos(x)

        def foo_cuda(x, scale: float):
            return scale * x, torch.cos(x)

        def foo_xpu(x, scale: float):
            return scale * x, torch.cos(x)

        def foo_meta(x, scale: float):
            return torch.empty_like(x), torch.empty_like(x)

        define_custom_op_2_for_test("foo2", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        def fn(x, scale: float):
            a = torch.nn.functional.relu(x)
            return torch.ops.test.foo2(a, scale)

        self.common(fn, (torch.randn((16, 32)), 2.0), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_custom_op_3(self):
        import torch.library

        def foo_cpu(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        def foo_cuda(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        def foo_xpu(x):
            result = torch.zeros_like(x[0])
            for t in x:
                result += t
            return result

        def foo_meta(x):
            return torch.empty_like(x[0])

        define_custom_op_3_for_test("foo3", foo_cpu, foo_cuda, foo_xpu, foo_meta)

        def fn(x):
            return torch.ops.test.foo3(x)

        self.common(
            fn,
            ([torch.randn((16, 32)), torch.randn((16, 32)), torch.randn((16, 32))],),
            check_lowp=False,
        )

    @requires_gpu()
    @torch._inductor.config.patch("layout_optimization", True)
    @torch._inductor.config.patch("keep_output_stride", False)
    @config.patch(implicit_fallbacks=True)
    def test_custom_op_fixed_layout_sequential(self):
        import torch.library

        mod = nn.Conv2d(3, 128, 1, stride=1, bias=False).to(device=GPU_TYPE)
        inp = torch.rand(2, 3, 128, 128, device=GPU_TYPE)
        expected_stride = mod(inp).stride()

        def bar_cpu(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        def bar_cuda(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        def bar_xpu(x):
            self.assertEqual(x.stride(), expected_stride)
            return x.clone()

        def bar_meta(x):
            return torch.empty_like(x)

        define_custom_op_for_test(
            "bar",
            bar_cpu,
            bar_cuda,
            bar_xpu,
            bar_meta,
            tags=[torch._C.Tag.needs_fixed_stride_order],
        )

        def fn(x):
            z = mod(x)
            output = torch.ops.test.bar(z)
            return output

        with torch.no_grad():
            # With keep_output_stride False, inductor would normally have different layout from eager execution
            # But because our custom op needs fixed layout, the assertions in the custom op will pass
            self.common(fn, (inp,), check_lowp=False)

    @requires_gpu()
    @config.patch(implicit_fallbacks=True)
    def test_custom_op_fixed_layout_channels_last(self):
        class Block(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

                self.in_layers = nn.Sequential(
                    nn.Dropout(p=0.1),
                )

            def helper(self, x):
                out = F.gelu(x)
                out = self.in_layers(out)
                return out

            def forward(self, x):
                out = self.helper(x)
                out = torch.ops.test.baz(out)
                return out

        model = Block()
        model = model.to(GPU_TYPE).to(memory_format=torch.channels_last)
        input_t = torch.randn([1, 320, 128, 128], dtype=torch.float32, device=GPU_TYPE)
        input_t = input_t.to(memory_format=torch.channels_last)
        expected_strides = model.helper(input_t).stride()

        def baz_cpu(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        def baz_cuda(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        def baz_xpu(x):
            self.assertEqual(expected_strides, x.stride())
            return x.clone()

        def baz_meta(x):
            return torch.empty_like(x)

        define_custom_op_for_test(
            "baz",
            baz_cpu,
            baz_cuda,
            baz_xpu,
            baz_meta,
            tags=[torch._C.Tag.needs_fixed_stride_order],
        )

        with torch.no_grad():
            net = torch.compile(model)
            out = net(input_t)

    @skip_if_gpu_halide  # cuda error
    def test_buffer_use_after_remove(self):
        # https://github.com/pytorch/pytorch/issues/102857

        def rotvec_to_rotmat(rotvec) -> torch.Tensor:
            """Simplified rotvec to rotmat code from RoMa
            (https://github.com/naver/roma/blob/06e4b0cdc1c802a60a012bb19c581d6600c63358/roma/mappings.py#L371)
            """
            theta = torch.norm(rotvec, dim=-1)
            axis = rotvec / theta[..., None]
            kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            one_minus_cos_theta = 1 - cos_theta
            xs = kx * sin_theta
            ys = ky * sin_theta
            zs = kz * sin_theta
            xyc = kx * ky * one_minus_cos_theta
            xzc = kx * kz * one_minus_cos_theta
            yzc = ky * kz * one_minus_cos_theta
            xxc = kx**2 * one_minus_cos_theta
            yyc = ky**2 * one_minus_cos_theta
            zzc = kz**2 * one_minus_cos_theta
            R_rodrigues = torch.stack(
                [
                    1 - yyc - zzc,
                    xyc - zs,
                    xzc + ys,
                    xyc + zs,
                    1 - xxc - zzc,
                    -xs + yzc,
                    xzc - ys,
                    xs + yzc,
                    1 - xxc - yyc,
                ],
                dim=-1,
            ).reshape(-1, 3, 3)
            R = R_rodrigues
            return R

        def f(coord, rot, trans):
            rot_mat = rotvec_to_rotmat(rot)
            coord = torch.einsum("...ij,...bj->...bi", rot_mat, coord) + trans
            return coord.sum()

        foo_c = torch.compile(f, dynamic=True)

        def run(fn):
            coord = torch.ones((2, 3), device=self.device)
            rot = nn.Parameter(torch.ones((2, 3), device=self.device))
            trans = nn.Parameter(torch.ones((2, 3), device=self.device))

            U = fn(coord, rot, trans)
            U.backward()

            return U, rot, trans

        U_e, rot_e, trans_e = run(f)
        U, rot, trans = run(foo_c)

        self.assertEqual(U, U_e)
        self.assertEqual(rot.grad, rot_e.grad)
        self.assertEqual(trans.grad, trans_e.grad)

    # If we serve from the cache, the init hook isn't called
    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    @skipIfWindows
    def test_inner_fn_str_and_stride(self):
        def f(x):
            x = x + 1
            x = test_operators.realize(x)
            x = x * 2
            x = test_operators.realize(x)
            return x

        x = torch.rand(3, 2, device=self.device).t()
        ref = f(x)
        called = False

        def hook_fn(scheduler, nodes):
            nonlocal called
            called = True

            if self.device != "cpu":
                self.assertEqual(len(nodes), 3)
                _, mul_buf, _ = nodes
                self.assertTrue(
                    all(
                        V.graph.sizevars.size_hints(buf.get_stride()) == (1, 2)
                        for buf in nodes
                    )
                )
                # before the fix, the wrong index expression
                # 'i1 + 3 * i0' is cached.
                self.assertTrue(
                    "i0 + 2 * i1" in mul_buf.data.inner_fn_str()
                    or "i0 + i1 * s1" in mul_buf.data.inner_fn_str()
                )

        with add_scheduler_init_hook(hook_fn):
            actual = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(ref, actual)
        self.assertTrue(called)

    @skip_if_gpu_halide  # cuda error
    def test_mutations_loop_fusion(self):
        def fn(tensor, index, source):
            out = tensor.index_add(0, index, source, alpha=2.0) / 2
            return out

        device = "cpu"
        tensor = torch.rand((1,), dtype=torch.double, device=device)
        index = torch.tensor([0], dtype=torch.long, device=device)
        source = torch.rand((1,), dtype=torch.double, device=device)
        self.common(
            fn,
            (
                tensor,
                index,
                source,
            ),
        )

    @config.patch(
        "triton.autotune_pointwise", True
    )  # needed to introduce config that exceed max shared memory usage
    @serialTest()
    def test_large_block_sizes(self):
        """
        Inductor will try triton configs like x = 64 and y = 1024 which will
        result in out of shared memory if dtype is fp32.

        Currently inductor will skip such bad configs and pick the best one
        from the remaining configs.
        """
        if not _has_sufficient_memory(self.device, 3 * 2**24 * 65 * 4):
            raise unittest.SkipTest("insufficient memory")

        @torch.compile
        def fn(x, y):
            return x.t() + y

        # Use shape (2**24, 65) rather than (2**24, 128) potentially avoid OOM in
        # CI while still keep the same up-rounded size-hints.
        a = torch.randn(2**24, 65, device=self.device)
        b = torch.randn(65, 2**24, device=self.device)
        fn(a, b)

    # Skipped on ROCm until https://github.com/ROCm/triton/issues/443 resolved
    def test_fuse_large_params(self):
        def pt2_optimizer_step(optimizer):
            @torch.compile()
            def f():
                optimizer.step()

            f()

        params = [
            torch.rand(10, 10, dtype=torch.float32, device=self.device)
            for _ in range(194)
        ]
        for p in params:
            p.grad = torch.rand_like(p)

        o = torch.optim.AdamW(params)
        pt2_optimizer_step(o)

    @skip_if_gpu_halide
    def test_adaptive_avg_pool1d_argmax(self):
        # https://github.com/pytorch/pytorch/issues/113013
        def fn(x):
            x = torch.adaptive_avg_pool1d(input=x, output_size=2)
            x = torch.argmax(input=x)
            return x

        x = torch.rand([4, 4, 3], dtype=torch.float64)
        self.common(fn, (x,))

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    # We only support dtypeview for abi_conpatible aoti
    @torch._inductor.config.patch(abi_compatible=True)
    def test_dtypeview(self):
        if TEST_WITH_ASAN:
            return

        # https://github.com/pytorch/pytorch/issues/126338
        def fn(x, y, x_dtype, x2):
            x = x.view(x_dtype)
            y = y.view(x_dtype) + 1
            x2 = x2.view(x_dtype) + 1
            return x @ y, x2 @ x

        test_dtypes = [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        for test_dtype_x in test_dtypes:
            for test_dtype_y in test_dtypes:
                # @ operation needs arguments to be the same dtype
                for view_dtype in test_dtypes:
                    try:
                        # print(f"({test_dtype_x}, {test_dtype_y}, {view_dtype})")
                        x = rand_strided(
                            (2, 2), (2, 1), device=self.device, dtype=test_dtype_x
                        )
                        y = rand_strided(
                            (2, 2), (2, 1), device=self.device, dtype=test_dtype_y
                        )
                        x2 = x.clone()
                        fn(x, y, view_dtype, x2)
                    except Exception as e:
                        continue
                    self.common(
                        fn,
                        (x, y, view_dtype, x2),
                        reference_in_float=False,
                        check_lowp=False,
                    )

    @torch._inductor.config.patch(abi_compatible=True)
    def test_dtypeview_fusion(self):
        @torch.compile
        def fn(x):
            x = x + 1
            x = torch.ops.aten.view.dtype(x, torch.int16)
            x = x * 2
            return x

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([1024], dtype=torch.float16, device=self.device)
        self.common(fn, (x,), reference_in_float=False)
        assertGeneratedKernelCountEqual(self, 1)

    @expectedFailureCodegenDynamic
    def test_reinterpret_dtypeview(self):
        @torch.compile
        def fn(x, x2):
            return x.view([10, 10]).view(torch.int32), x2.view(torch.int32).view(
                [10, 10]
            )

        x = torch.randn([100, 1], device=self.device)
        x2 = x.clone()
        self.common(fn, (x, x2), reference_in_float=False, check_lowp=False)
        x = torch.randn([100, 1], device=self.device)
        x2 = x.clone()
        _, code = run_and_get_code(fn, x, x2)
        FileCheck().check("aten.view.dtype(reinterpret_tensor").run(code[0])

    def test_float16_to_int16(self):
        def fn(x):
            x_view = x.view(dtype=torch.int16)
            return x_view.mul(2)

        x = torch.ones(4, dtype=torch.float16, device=self.device)
        ref = fn(x)
        actual = torch.compile(fn)(x)
        self.assertEqual(ref, actual)

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_bfloat16_to_int16(self):
        def fn(a, b):
            x = a + b
            x_view = x.view(dtype=torch.int16)
            return x_view.mul(2)

        a = torch.ones(4, dtype=torch.bfloat16, device=self.device)
        b = torch.ones(4, dtype=torch.bfloat16, device=self.device)
        ref = fn(a, b)
        actual = torch.compile(fn)(a, b)
        self.assertEqual(ref, actual)

    def test_float32_to_int32(self):
        def fn(a, b):
            x = a + b
            x_view = x.view(dtype=torch.int32)
            return x_view.mul(2)

        a = torch.ones(4, dtype=torch.float32, device=self.device)
        b = torch.ones(4, dtype=torch.float32, device=self.device)
        ref = fn(a, b)
        actual = torch.compile(fn)(a, b)
        self.assertEqual(ref, actual)

    @skipIfWindows
    def test_randint_int64_mod(self):
        # This used to not compile due to a wrong return type of randint64_cpu
        # See https://github.com/pytorch/pytorch/issues/117435
        def fn(n):
            return (
                torch.randint(
                    low=-5, high=5, size=(n,), dtype=torch.int64, device=self.device
                )
                % 10
            )

        res = torch.compile(fn)(20)
        self.assertTrue(torch.all((0 <= res) & (res < 10)).item())

    @torch._inductor.config.patch(force_shape_pad=True)
    @skip_if_gpu_halide  # correctness issue
    def test_should_pad_bench_for_bmm(self):
        B = 2
        M = 1024
        N = 1024
        K = 1024 + 1  # a size that requires padding

        mat1 = torch.rand(B, M, K, device=self.device)
        mat2 = torch.rand(B, K, N, device=self.device)

        should_pad = pad_mm.should_pad_bench(None, mat1, mat2, torch.ops.aten.bmm)

        self.assertTrue(should_pad)

    @parametrize(
        "name, op",
        [
            subtest((name, getattr(torch.special, name)), name=name)
            for name in torch.special.__all__
            if name not in {"softmax", "log_softmax", "logsumexp"}
        ],
    )
    def test_pointwise(self, name, op):
        dtype = torch.float32
        check_lowp = True
        if self.device == GPU_TYPE and name in {
            "airy_ai",
            "bessel_i0",
            "bessel_i1",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "erfcx",
            "gammainc",
            "gammaincc",
            "i1",
            "i1e",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "ndtri",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "spherical_bessel_j0",
            "zeta",
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_w",
            "legendre_polynomial_p",
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "laguerre_polynomial_l",
        }:
            # <func>_cuda not implemented for Half
            check_lowp = False

        if is_halide_backend(self.device) and name in (
            "erfinv",
            "airy_ai",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "digamma",
            "gammainc",
            "gammaincc",
            "gammaln",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "i0",
            "i0e",
            "i1",
            "i1e",
            "laguerre_polynomial_l",
            "legendre_polynomial_p",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "multigammaln",
            "ndtri",
            "polygamma",
            "psi",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
            "spherical_bessel_j0",
            "zeta",
        ):
            raise unittest.SkipTest(f"halide does not support {name}")

        if name in {"gammainc", "gammaincc"}:
            args = (
                torch.randn(8, 8, dtype=dtype, device=self.device),
                torch.empty(8, 8, dtype=dtype, device=self.device).uniform_(1, 2),
            )

            def fn(x, y):
                return op(x, y)

        elif name in {"xlog1py", "xlogy", "zeta"}:
            args = (
                torch.randn(8, 8, dtype=dtype, device=self.device),
                torch.empty(8, 8, dtype=dtype, device=self.device).uniform_(1, 2),
            )

            def fn(x, y):
                return op(x, y)

        elif name == "multigammaln":
            args = (
                torch.empty(8, 8, dtype=dtype, device=self.device).uniform_(1, 2),
                2,
            )

            def fn(x, p):
                return op(x, p)

        elif name == "polygamma":
            args = (
                1,
                torch.empty(8, 8, dtype=dtype, device=self.device).uniform_(1, 10),
            )

            def fn(n, x):
                return op(n, x)

        elif "_polynomial_" in name:
            args = (
                torch.randn(8, 8, dtype=dtype, device=self.device),
                2,
            )

            def fn(x, n):
                return op(x, n)

        else:
            args = (torch.randn(8, 8, dtype=dtype, device=self.device),)

            def fn(x):
                return op(x)

        self.common(fn, args, check_lowp=check_lowp, atol=1e-4, rtol=1e-4)

    # codegen test fails with no dynamic for loop in dynamic shape tests
    @expectedFailureCodegenDynamic
    def test_view_uint8_through_differing_bitwidths(self):
        # https://github.com/pytorch/pytorch/issues/120998
        def fn(x, view_dtype):
            return x.view(view_dtype).view(torch.uint8)

        view_dtypes = [torch.int16, torch.int32, torch.int64]
        for dtype in view_dtypes:
            x = torch.randint(0, 2**4, [4096, 4096], dtype=torch.uint8)
            self.common(
                fn,
                (
                    x,
                    dtype,
                ),
            )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_with_unbacked_symints(self):
        @torch.compile()
        def f(sz, x):
            s0, s1 = sz.tolist()
            r0, r1 = torch.ops.aten.split_with_sizes.default(x, [s0, s1])
            return torch.ops.aten.sort.default(r1)

        N = 7312
        S0 = 420
        S1 = N - S0

        result = f(torch.tensor([S0, S1]), torch.randn(N))
        self.assertTrue(len(result) == 2)

        @torch.compile()
        def f2(x):
            y = torch.arange(x.item())
            return torch.ops.aten.split_with_sizes.default(y, [5, 5, 10])

        result = f2(torch.tensor([20]))
        self.assertTrue(len(result) == 3)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_unbacked_symints(self):
        # https://github.com/pytorch/pytorch/issues/122937
        @torch.compile()
        def f(x):
            y = torch.arange(x.item())
            return torch.split(y, [5, 5, 10])

        result = f(torch.tensor([20]))
        self.assertTrue(len(result) == 3)

    def test_complex_memory_overlap(self):
        t = rand_strided((8, 1500, 1), (1504, 1, 1), device=self.device)
        self.assertFalse(complex_memory_overlap(t))

    def test_generate_rand_fp8(self):
        """
        PyTorch can not generate fp8 tensors with a normal distribution because of
        missing needed kernels.

        We work around that in rand_strided by generating an fp16 tensor first and
        then do casting.
        """
        t = rand_strided((2, 3), (3, 1), device=self.device, dtype=torch.float8_e4m3fn)
        self.assertTrue(t.dtype is torch.float8_e4m3fn)

    @skipIfWindows
    def test_large_grid(self):
        # https://github.com/pytorch/pytorch/issues/123210
        def fn(primals_5):
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            primals_5 = None
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            return clone

        s0 = 16777472
        s1 = 8
        compiled_fn = torch._dynamo.optimize()(fn)
        actual = compiled_fn(torch.ones(s0, s1, device=self.device))
        self.assertTrue((actual == 1).all())

    @skip_if_gpu_halide
    def test_pattern_matcher_multi_user(self):
        # Reproducer for https://github.com/pytorch/pytorch/issues/129685

        def forward(float_1, view_1):
            logits = float_1 / 64.0
            loss = torch.nn.functional.cross_entropy(logits, view_1, ignore_index=5)
            logsumexp = logits.logsumexp(dim=-1)
            return [loss, logsumexp]

        a = torch.randn(512, 4096, requires_grad=True)
        b = torch.randint(size=(512,), low=0, high=4095)

        self.common(forward, (a, b))

    def test_mul_index_expr(self):
        # Minified repro from https://github.com/pytorch/pytorch/issues/111884
        def forward():
            iota = torch.ops.prims.iota.default(
                16,
                start=0,
                step=1,
                dtype=torch.int64,
                device=self.device,
                requires_grad=False,
            )
            unsqueeze = torch.ops.aten.unsqueeze.default(iota, -1)
            mul = torch.ops.aten.mul.Tensor(unsqueeze, iota)
            unsqueeze = iota = None
            neg = torch.ops.aten.neg.default(mul)
            mul = None
            div = torch.ops.aten.div.Tensor(neg, 16)
            neg = None
            return (div,)

        self.common(forward, ())


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


if HAS_CPU:

    class SweepInputsCpuTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cpu")

    SweepInputsCpuTest.populate()

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(CommonTemplate, CpuTests, "cpu")

if HAS_GPU and not TEST_WITH_ASAN:

    class SweepInputsGPUTest(SweepInputs2, TestCase):
        gen = InputGen(10, GPU_TYPE)

    SweepInputsGPUTest.populate()

    class GPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, GPUTests, GPU_TYPE)

    class TritonCodeGenTests(TestCase):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        device_type = GPU_TYPE
        device = GPU_TYPE

        class NoOpCompilerBackend:
            def __init__(self):
                self.example_args = None
                self.model = None

            def noop_backend(
                self,
                model_: torch.fx.GraphModule,
                example_inputs_: typing.List[torch.Tensor],
            ):
                """
                The Noop backend does not compile the fx graph it is given.
                Instead, it transforms the fx graph so that its functions are
                aten operations. It then saves this graph.
                """
                from torch._inductor.decomposition import select_decomp_table
                from torch._subclasses import FakeTensorMode
                from torch.fx import Interpreter

                fake_mode = FakeTensorMode()

                def interpret(*args, **kwargs):
                    return Interpreter(model_).run(*args[0:], **kwargs)

                fake_flat_tensor_args = [
                    fake_mode.from_tensor(x) for x in example_inputs_
                ]
                fw_module = make_fx(interpret, select_decomp_table())(
                    *fake_flat_tensor_args
                )
                self.model = fw_module
                self.example_args = fake_flat_tensor_args
                return lambda x: example_inputs_

        def get_kernels(self, fn, args) -> typing.List[CachingAutotuner]:
            from torch._inductor.debug import DebugContext
            from torch._inductor.graph import GraphLowering
            from torch._inductor.virtualized import V

            cxt = TritonCodeGenTests.NoOpCompilerBackend()
            torch._dynamo.optimize(backend=cxt.noop_backend)(fn)(*args)
            graph = GraphLowering(cxt.model)
            kernels = []
            with V.set_graph_handler(graph), V.set_debug_handler(DebugContext()):
                graph.run(*(cxt.example_args))
                mod = graph.compile_to_module()

                for val in mod.__dict__.values():
                    if isinstance(
                        val, torch._inductor.runtime.triton_heuristics.CachingAutotuner
                    ):
                        kernels.append(val)

            return kernels

        def test_divisible_by_16_covers_numel_args(self):
            torch._dynamo.reset()

            def fn(a: torch.Tensor) -> torch.Tensor:
                return torch.sum(a)

            kernels = self.get_kernels(fn, [torch.randn([256, 256], device=GPU_TYPE)])
            if config.triton.multi_kernel:
                self.assertTrue(
                    len(kernels) == 4,
                    "SUM should result in four kernels when multi-kernel is enabled",
                )
            else:
                self.assertTrue(len(kernels) == 2, "SUM should result in two kernels")

            # kernel0 reduces from 256 to (xnumel=8, rnumel=8192), which means it reduces 256 by 256 into an array of
            # size 8 by accumulating 8192 elements at once note that rnumel is equal to 512 * 16, so rnumel which is
            # at slot 3 should be in the divisible by 16 descriptor
            arguments_that_are_divisible_by_16_in_kernel0 = (
                kernels[0].triton_meta["configs"][0].divisible_by_16
            )
            self.assertEqual(arguments_that_are_divisible_by_16_in_kernel0, (0, 1, 3))

            # kernel1 reduces from 8 elements to a single scalar.
            # Since multi-kernel generate 2 variants for each kernel. The second
            # persistent-reduction has index 2.
            kernel1_index = 2 if config.triton.multi_kernel else 1
            arguments_that_are_divisible_by_16_in_kernel1 = (
                kernels[kernel1_index].triton_meta["configs"][0].divisible_by_16
            )
            self.assertEqual(arguments_that_are_divisible_by_16_in_kernel1, (0, 1))
            torch._dynamo.reset()

        @config.patch(assume_aligned_inputs=False)
        def test_codegen_config_option_dont_assume_alignment(self):
            def fn(x: torch.Tensor) -> torch.Tensor:
                return x.sin() + x.cos()

            # We want code that assumes alignment if the initial input is 16-byte aligned
            for offset in (0, 1, 2, 3, 4):
                base = torch.randn(64 * 64 + 64, dtype=torch.float32, device=GPU_TYPE)
                inps = torch.as_strided(base, (64, 64), (64, 1), offset)
                torch._dynamo.reset()
                kernels = self.get_kernels(fn, [inps])
                arguments_that_are_divisible_by_16 = (
                    kernels[0].triton_meta["configs"][0].divisible_by_16
                )

                #             NO_ALIGN ALIGN     ALIGN
                # def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr)

                if offset % 4 == 0:
                    expected_aligned = (0, 1, 2)
                else:
                    expected_aligned = (1, 2)
                self.assertEqual(arguments_that_are_divisible_by_16, expected_aligned)

            # If input isn't a view, storage offset != , inductor will assume alignment.
            torch._dynamo.reset()
            inp = torch.randn((64, 64), device=GPU_TYPE)
            kernels = self.get_kernels(fn, [inp])
            arguments_that_are_divisible_by_16 = (
                kernels[0].triton_meta["configs"][0].divisible_by_16
            )
            self.assertEqual(arguments_that_are_divisible_by_16, (0, 1, 2))

        def test_optimize_indexing_dtype(self):
            def fn(x: torch.Tensor) -> torch.Tensor:
                return aten.upsample_bilinear2d.vec(x, None, True, [2.0, 2.0])

            fn_opt = torch._dynamo.optimize("inductor")(fn)
            inps = [torch.randn(2, 4, 16, 16, device=GPU_TYPE)]
            code = run_and_get_triton_code(fn_opt, *inps)
            self.assertTrue("to(tl.int32)" in code)
            self.assertFalse("to(tl.int64)" in code)

            self.assertEqual(fn_opt(*inps), fn(*inps))

        @config.patch({"fx_graph_remote_cache": False})
        def test_optimize_indexing_dtype_with_constraint(self):
            def fn1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                x = torch.arange(0, b.shape[0], device=GPU_TYPE)
                y = ((x + x) / 3).int()
                return a[y.to(torch.int64)]

            def fn2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                torch._check_is_size(b.shape[0])
                torch._check(b.shape[0] >= 2)
                torch._check(b.shape[0] <= 100)
                return fn1(a, b)

            fn1_opt = torch._dynamo.optimize("inductor")(fn1)
            fn2_opt = torch._dynamo.optimize("inductor")(fn2)

            a = torch.rand([100, 100], device=GPU_TYPE)
            b1 = torch.rand([102], device=GPU_TYPE)
            b2 = torch.rand([100], device=GPU_TYPE)
            torch._dynamo.mark_dynamic(b1, 0)
            torch._dynamo.mark_dynamic(b2, 0)
            inps1 = [a, b1]
            inps2 = [a, b2]

            # Run fn2 first since it has more restrictive bounds -- to avoid cache hit
            code2 = run_and_get_triton_code(fn2_opt, *inps2)
            code1 = run_and_get_triton_code(fn1_opt, *inps1)

            # The function with the constrained tensor should be optimized, but
            # the other should not:
            self.assertTrue("to(tl.int64)" in code1)
            self.assertTrue("to(tl.int32)" in code2)
            self.assertFalse("to(tl.int64)" in code2)

            self.assertEqual(fn1_opt(*inps1), fn1(*inps1))
            self.assertEqual(fn2_opt(*inps2), fn1(*inps2))

        def test_constant_folding_deallocation(self):
            import torch._inductor

            def fn():
                li = []
                for i in range(10):
                    x = torch.full([100], i)
                    x = x + 1
                    li.append(x)

                return li

            mod = make_fx(fn)()

            live_tensors = WeakTensorKeyDictionary()
            max_live_tensors = 0

            class LiveTensors(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    nonlocal live_tensors
                    nonlocal max_live_tensors

                    kwargs = kwargs if kwargs else {}
                    for arg in pytree.arg_tree_leaves(*args, **kwargs):
                        if isinstance(arg, torch.Tensor):
                            live_tensors[arg] = True

                    out = func(*args, **kwargs)
                    if not isinstance(out, torch.Tensor):
                        return out

                    live_tensors[out] = True
                    max_live_tensors = max(max_live_tensors, len(live_tensors))
                    return out

            mode = LiveTensors()
            from torch._inductor.fx_passes.joint_graph import UniformValueConstantFolder

            with mode:
                UniformValueConstantFolder(mod).run()

            # there are a couple extra tensors created in `insertable_tensor_check`
            self.assertTrue(max_live_tensors == 3)

        # See https://github.com/pytorch/pytorch/issues/100348
        def test_inductor_detach_view(self):
            def fn(x: torch.Tensor) -> torch.Tensor:
                a = x * 2
                return a, a.detach()

            fn_opt = torch._dynamo.optimize("inductor")(fn)
            inp = torch.ones(2, 2, requires_grad=True, device=GPU_TYPE)
            inp_ref = inp.clone().detach().requires_grad_(True)
            out_ref = fn(inp_ref)
            out = fn_opt(inp)
            out_ref[0].sum().backward()
            out[0].sum().backward()
            self.assertEqual(inp.grad, inp_ref.grad)

        @requires_gpu()
        @unittest.skipIf(
            not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
            "Does not support mem_eff_attention",
        )
        def test_sdpa_inference_mode_aot_compile(self):
            class TestSDPA(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(
                    self,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    attn_mask: torch.Tensor,
                ):
                    return torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                    )

            q = torch.rand([10, 4, 128, 64], device=GPU_TYPE, dtype=torch.bfloat16)
            k = torch.rand([10, 4, 128, 64], device=GPU_TYPE, dtype=torch.bfloat16)
            v = torch.rand([10, 4, 128, 64], device=GPU_TYPE, dtype=torch.bfloat16)
            attn_mask = (
                torch.rand([10, 4, 128, 128], device=GPU_TYPE, dtype=torch.bfloat16)
                < 0.9
            )

            inputs = (q, k, v, attn_mask)

            import torch.export._trace as export_trace

            with torch.inference_mode():
                traced = export_trace._export_to_torch_ir(
                    TestSDPA(),
                    inputs,
                    disable_constraint_solver=True,
                    restore_fqn=False,
                )
                torch._inductor.aot_compile(traced, inputs)

        def test_optimize_indexing_assert(self):
            def has_indirect(code, tl_fn: str):
                self.assertTrue(
                    tl_fn in code,
                    msg=f"{tl_fn} not present:\n{code}",
                )
                for line in code.split("\n"):
                    if tl_fn in line:
                        stmt = line.split(tl_fn)[-1]
                        # indirect indexing involves a `tmp` variable
                        self.assertTrue(
                            "tmp" in stmt,
                            msg=f"Indirect indexing not present in code:\n{line}",
                        )

            def has_assert(code, lower: bool, upper: bool):
                self.assertIn(
                    "device_assert", code, msg=f"No device asert found:\n{code}"
                )
                for line in code.split("\n"):
                    if "device_assert" in line:
                        self.assertTrue(
                            ("0 <= " in line) is lower,
                            msg=f"Lower bound {'' if lower else 'not '}elided:{line}",
                        )
                        self.assertTrue(
                            (" < " in line) is upper,
                            msg=f"Upper bound {'' if upper else 'not '}elided:{line}",
                        )

            def fn(x: torch.Tensor) -> torch.Tensor:
                s = 1.0 * torch.arange(x.shape[0], device=x.device)
                return x[s.long()]

            # aten.index
            for dynamic in (False, True):
                fn_opt = torch.compile(fn, dynamic=dynamic)

                x = torch.randn(8, device=GPU_TYPE)
                code = run_and_get_triton_code(fn_opt, x)
                self.assertEqual(fn_opt(x), fn(x), msg=f"{dynamic=}")

                # Check that there's indirect indexing...
                has_indirect(code, tl_fn="tl.load")
                if not dynamic:
                    # We elide the assert for static shapes
                    self.assertNotIn("device_assert", code)
                else:
                    # ...but we generate an upper bound for dynamic shapes
                    has_assert(code, lower=False, upper=True)

            def fn(a, z, b, idx0, idx1):
                idx2 = torch.arange(a.shape[-1], device=a.device)
                a.index_put_((z, idx0, idx1, idx2), b, accumulate=True)
                return a

            # aten.index_put
            for dynamic in (False, True):
                fn_opt = torch.compile(fn, dynamic=dynamic)
                a = torch.randn(1, 32, 32, 4, device=GPU_TYPE)
                z = torch.zeros((), dtype=torch.int64, device=GPU_TYPE)
                b = torch.randn(33, 1, device=GPU_TYPE)
                idx0 = torch.randint(32, (33,), device=GPU_TYPE).view(33, 1, 1)
                idx1 = torch.randint(32, (33,), device=GPU_TYPE).view(33, 1)
                inps = (a.clone(), z, b, idx0, idx1)
                code = run_and_get_triton_code(fn_opt, *inps)

                # Correctness
                out_opt = fn_opt(a.clone(), z, b, idx0, idx1)
                out = fn(a.clone(), z, b, idx0, idx1)
                self.assertEqual(out_opt, out, msg=f"{dynamic=}")

                # We have an indirect store via atomic_add
                has_indirect(code, tl_fn="tl.atomic_add")
                # We cannot elide he assert in this case
                has_assert(code, lower=True, upper=True)

        def test_not_materialize_pointwise_reduction(self):
            def fn(a, b):
                return (a - b).sum(dim=-1).amax(dim=-1)

            N = 16
            K = 7
            fn_opt = torch._dynamo.optimize("inductor")(fn)
            inps = [
                torch.randn(N, 1, K, device=GPU_TYPE),
                torch.randn(1, N, K, device=GPU_TYPE),
            ]
            code = run_and_get_triton_code(fn_opt, *inps)
            self.assertEqual(
                code.count("tl.store"), 2 if config.triton.multi_kernel else 1
            )
            self.assertTrue("out_ptr1" in code)
            self.assertFalse("out_ptr0" in code)
            self.assertEqual(fn_opt(*inps), fn(*inps))

        def test_numpy_on_gpu(self):
            x = np.arange(10, dtype=np.float32)

            @torch.compile
            def fn(x):
                return np.sin(x)

            def fn_gpu(x):
                with torch.device(GPU_TYPE):
                    return fn(x)

            r = fn_gpu(x)
            code = run_and_get_triton_code(fn_gpu, x)
            self.assertIn("tl_math.sin", code)
            self.assertEqual(type(r), np.ndarray)
            self.assertEqual(r, np.sin(x))

        def test_numpy_autograd(self):
            def my_torch(x):
                y = torch.cat([torch.sin(x) ** 2, torch.max(x)[None]])
                return y.sum()

            def my_np(x):
                y = np.concatenate([np.sin(x) ** 2, np.max(x)[None]])
                return np.sum(y)

            @torch.compile
            def wrapper(x):
                return torch.compiler.wrap_numpy(my_np)(x)

            @torch.compile
            def wrapper2(x):
                x = x.numpy()
                y = my_np(x)
                return torch.from_numpy(y)

            x_np = torch.arange(8, dtype=torch.float32, requires_grad=True)
            x = torch.arange(8, dtype=torch.float32, requires_grad=True)
            out_np = wrapper(x_np)
            out = my_torch(x)
            self.assertEqual(out, out_np)

            x2_np = torch.arange(8, dtype=torch.float32, requires_grad=True)
            out2_np = wrapper2(x2_np)
            self.assertEqual(out, out2_np)

            out_np.backward()
            out.backward()
            self.assertEqual(x.grad, x_np.grad)

            out2_np.backward()
            self.assertEqual(x.grad, x2_np.grad)

        # Disable constant propagation, so we isolate value range analysis
        @patch.object(config, "constant_and_index_propagation", False)
        @patch.object(config, "joint_graph_constant_folding", False)
        def test_cant_optimize_compute(self):
            def ones():
                return torch.ones([4], device=GPU_TYPE)

            def suffix(inp):
                return (inp.to(torch.int64) + 1).to(torch.float64)

            ten = torch.rand([4], device=GPU_TYPE)

            for foo in (
                lambda x: x + 2147483657,
                lambda x: torch.where(x < 0, ones(), ones() - 2) * (-(2 ** (40))),
                lambda x: x + ten,
                lambda x: x + ten.sum(),
            ):

                def fn():
                    return suffix(foo(ones()))

                fn_opt = torch._dynamo.optimize("inductor")(fn)
                code = run_and_get_triton_code(fn_opt)

                # this cannot be optimized away, value too large
                self.assertTrue("to(tl.int64)" in code)
                self.assertEqual(fn_opt(), fn())

        # Disable constant propagation, so we isolate value range analysis
        @patch.object(config, "constant_and_index_propagation", False)
        @patch.object(config, "joint_graph_constant_folding", False)
        def test_optimize_compute(self):
            def ones():
                return torch.ones([4], device=GPU_TYPE)

            def suffix(inp):
                return (inp.to(torch.int64) + 1).to(torch.float64)

            for foo in (
                lambda x: x + 500,
                lambda x: torch.where(x < 0, ones(), ones() - 2) * (-(2 ** (20))),
                lambda x: x / 30,
            ):

                def fn():
                    return suffix(foo(ones()))

                fn_opt = torch._dynamo.optimize("inductor")(fn)
                code = run_and_get_triton_code(fn_opt)

                # this can be optimized away, value too large
                self.assertTrue("to(tl.int64)" not in code)
                self.assertTrue("to(tl.int32)" in code)

                self.assertEqual(fn_opt(), fn())

        # https://github.com/pytorch/pytorch/issues/130335
        def test_ctr_not_moved_to_cuda_when_used_in_index_put(self):
            @torch.compile
            def f(x, mask):
                x[:, mask] = -math.inf
                return x

            x_tmp = torch.randn(512, 19, device=GPU_TYPE)
            x = x_tmp.permute(1, 0).view(-1, 128, 4)[:, :, 1:]

            mask_tmp = torch.ones(128, 3, dtype=torch.int32, device=GPU_TYPE)
            mask = mask_tmp == mask_tmp
            f(x, mask)
            code = run_and_get_triton_code(f, x, mask)
            # What we are testing here:
            # inductor has a pass to move tensor constructors on cpu to cuda
            # (the -math.inf will become a scalar-tensor input to index_put_())
            # we are asserting that when inductor allocates this tensor,
            # it does not move the tensor constructor to cuda and keeps it on CPU.
            self.assertFalse("empty_strided_cuda(()" in code)

        @config.patch("triton.use_block_ptr", False)
        def test_evict_last_non_coalesced_loads(self):
            @torch.compile
            def f(a, b):
                return (a * b).sum(dim=-1)

            N = 512
            inps = (
                torch.randn(N, N, N, device=GPU_TYPE).permute(2, 1, 0),
                torch.randn(N, N, N, device=GPU_TYPE).permute(1, 2, 0),
            )
            code = run_and_get_triton_code(f, *inps)
            lines = [line for line in code.split("\n") if "tl.load" in line]
            if config.triton.multi_kernel:
                # the first 2 lines are generated for the persistent reduction
                # variant.
                self.assertExpectedInline(
                    "\n".join(lines),
                    """\
    tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (262144*r2)), rmask, other=0.0)
        tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + (262144*r2)), rmask, eviction_policy='evict_first', other=0.0)""",
                )
            else:
                self.assertExpectedInline(
                    "\n".join(lines),
                    """\
        tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + (262144*r2)), rmask, eviction_policy='evict_first', other=0.0)""",
                )

        @config.patch("triton.use_block_ptr", True)
        def test_evict_last_non_coalesced_loads_block_ptr(self):
            @torch.compile
            def f(a, b):
                return (a * b).sum(dim=-1)

            N = 512
            inps = (
                torch.randn(N, N, N, device=GPU_TYPE).permute(2, 1, 0),
                torch.randn(N, N, N, device=GPU_TYPE).permute(1, 2, 0),
            )
            code = run_and_get_triton_code(f, *inps)
            lines = [line for line in code.split("\n") if "tl.load" in line]

            if config.triton.multi_kernel:
                # the first 2 lines are generated for the persistent reduction
                # variant.
                self.assertExpectedInline(
                    "\n".join(lines),
                    """\
    tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(tl.make_block_ptr(in_ptr1, shape=[262144, 512], strides=[1, 262144], block_shape=[XBLOCK, RBLOCK], order=[0, 1], offsets=[xoffset, roffset]), boundary_check=[1], padding_option='zero')
        tmp0 = tl.load(in_ptr0 + (x1 + (512*x0) + (262144*r2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(block_ptr0, boundary_check=[1], padding_option='zero', eviction_policy='evict_first')""",  # noqa: B950 line too long
                )
            else:
                self.assertExpectedInline(
                    "\n".join(lines),
                    """\
        tmp0 = tl.reshape(tl.load(block_ptr0, boundary_check=[3], padding_option='zero', eviction_policy='evict_last'), [XBLOCK, RBLOCK])
        tmp1 = tl.load(block_ptr1, boundary_check=[1], padding_option='zero', eviction_policy='evict_first')""",  # noqa: B950 line too long
                )

        # Disable index propagation, so the indirect indexing isn't optimized away
        @patch.object(config, "constant_and_index_propagation", False)
        def test_computed_indirect_mask(self):
            def fn(x, n):
                tmp = torch.arange(n, device=x.device)
                return x[tmp] + 1

            x = torch.randn(8, device=GPU_TYPE)
            fn_opt = torch.compile(fn)
            code = run_and_get_triton_code(fn_opt, x, 8)
            # load should be masked
            self.assertTrue(
                "tl.load(in_ptr0 + (tmp0), xmask" in code
                or "tl.load(in_ptr0 + (tmp0), (xmask).to(tl.int1)" in code
            )
            self.assertEqual(fn(x, 8), fn_opt(x, 8))

        def test_kernel_names_descriptive(self):
            @torch._dynamo.optimize("inductor")
            def fn1(x):
                return x.cos().sin()

            @torch._dynamo.optimize("inductor")
            def fn2(x):
                x = torch.mm(x, x)
                x = torch.softmax(x, dim=1)
                return x

            mod = nn.Sequential(
                nn.Linear(4, 4),
                nn.LayerNorm(4),
                nn.ReLU(),
            ).to(device=GPU_TYPE)

            @torch._dynamo.optimize("inductor")
            def fn3(x):
                return mod(x)

            func_and_kernel_aten = [
                (fn1, "triton_poi_fused_cos_sin", (torch.randn(8, device=GPU_TYPE),)),
                (
                    fn2,
                    "triton_poi_fused__softmax",
                    (torch.randn(4, 4, device=GPU_TYPE),),
                ),
                (
                    fn3,
                    "triton_poi_fused_native_layer_norm_relu",
                    (torch.randn(4, 4, device=GPU_TYPE),),
                ),
            ]
            func_and_kernel_torch = [
                (fn1, "triton_poi_fused_cos_sin", (torch.randn(8, device=GPU_TYPE),)),
                (
                    fn2,
                    "triton_poi_fused_softmax",
                    (torch.randn(4, 4, device=GPU_TYPE),),
                ),
                (
                    fn3,
                    "triton_poi_fused_layer_norm_relu"
                    if torch._dynamo.config.inline_inbuilt_nn_modules
                    else "triton_poi_fused_LayerNorm_ReLU",
                    (torch.randn(4, 4, device=GPU_TYPE),),
                ),
            ]

            def test_funcs(func_and_kernel):
                with torch.no_grad():
                    for fn, kernel_name, inps in func_and_kernel:
                        code = run_and_get_triton_code(fn, *inps)
                        if kernel_name not in code:
                            print(code)
                        self.assertTrue(kernel_name in code)

            test_funcs(func_and_kernel_aten)
            patch.object(config.triton, "descriptive_names", "torch")(test_funcs)(
                func_and_kernel_torch
            )

        @patch.object(config, "profile_bandwidth", True)
        def test_bandwidth_profiler(self):
            @torch._dynamo.optimize("inductor")
            def fn(x):
                x = x.cos()
                x = x.cos()
                x = torch.mm(x, x)
                x = x.sin()
                x = x.relu()
                return x

            inp = torch.randn(4, 4, device=GPU_TYPE)
            code = run_and_get_triton_code(fn, inp)
            fn(inp)
            self.assertTrue("start_graph" in code)
            self.assertTrue("end_graph" in code)

        def test_split_op_with_sym(self):
            def fn(x: torch.Tensor) -> torch.Tensor:
                # split(tensor, sympy.Integer), split(tensor, sympy.Expr)
                return torch.split(x, x.shape[0]), torch.split(x, x.shape[0] // 2)

            for dynamic_shapes in [True, False]:
                with torch._dynamo.config.patch(dynamic_shapes=dynamic_shapes):
                    torch._dynamo.reset()
                    fn_opt = torch._dynamo.optimize("inductor", dynamic=dynamic_shapes)(
                        fn
                    )
                    inps = torch.randn([5, 5])
                    fn_opt(inps)

        @skipIfRocm
        @unittest.skipIf(IS_FBCODE, "fbcode system python does not provide torch")
        def test_indirect_device_assert(self):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            test_path = os.path.join(dir_path, "indirect_assert_helper.py")
            fns = ("first_arg", "store", "second_arg", "same_pm_one", "same_pp_one")

            def test(fn, ndims, dyn_shape, one_size=False):
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        test_path,
                        fn,
                        str(ndims),
                        str(dyn_shape),
                        str(one_size),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
                )
                stderr = proc.communicate()[1]
                self.assertTrue(
                    any(
                        "out of bounds" in err.decode("utf-8")
                        for err in stderr.splitlines()
                    ),
                    f"{fn}, {ndims}, {dyn_shape}, {one_size}",
                )

            for fn, ndims, dyn_shape in itertools.product(fns, (2, 3), (True, False)):
                test(fn, ndims, dyn_shape)

            test("first_arg", 2, False, True)

            for fn, dyn_shape in itertools.product(
                ("upper1", "upper2", "lower1", "lower2"), (True, False)
            ):
                test(fn, 2, dyn_shape)

        @patch("torch._inductor.config.comment_origin", True)
        @patch("torch._functorch.config.max_dist_from_bw", 0)
        def test_inductor_sequence_nr(self):
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=(1, 1),
                        stride=1,
                        padding="same",
                        bias=True,
                    )
                    self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                    self.relu1 = torch.nn.ReLU()
                    self.loss_fn = torch.nn.L1Loss()

                def forward(self, x, target):
                    y = x
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu1(x)
                    x = x + y
                    x = torch.flatten(x)
                    output = self.loss_fn(x, target)
                    return (output,)

            def get_triton_codegen(optimized_module, args):
                def run_with_backward():
                    result = optimized_module(*args)
                    result[0].backward()
                    return result

                res, (fwd_code, bwd_code) = run_and_get_code(run_with_backward)
                return fwd_code, bwd_code

            x = torch.rand(100, 16, 32, 32, requires_grad=True, device=GPU_TYPE)
            target = torch.rand(1, device=GPU_TYPE)
            args = [x, target]
            model = Model().to(device=GPU_TYPE)
            opt_model = torch.compile(model)
            fwd_code, bwd_code = get_triton_codegen(opt_model, args)

            bwd_seq_nr_set = set()
            fwd_seq_nr_set = set()
            for idx, code in enumerate([fwd_code, bwd_code]):
                seq_nr_set = bwd_seq_nr_set if idx > 0 else fwd_seq_nr_set
                prefix = "BWD" if idx > 0 else "FWD"
                for line in code.split("\n"):
                    if "seq_nr" in line:
                        res = re.search(r"seq_nr:(\d+)", line)
                        if res:
                            seq_nr_set.add(int(res.group(1)))
            self.assertTrue(bwd_seq_nr_set.issubset(fwd_seq_nr_set))

        @config.patch(
            {
                "coordinate_descent_tuning": True,
                "triton.unique_kernel_names": True,
                "benchmark_kernel": True,
            }
        )
        @skipIfRocm
        @expectedFailureXPU
        @unittest.skipIf(
            torch.cuda.is_available() and torch.cuda.get_device_capability() < (9, 0),
            "Triton does not support fp8 on A100",
        )
        def test_red_followed_by_transposed_pointwise(self):
            bs = 26624
            dim = 1024

            @torch.compile(dynamic=False)
            def f(in1, in2, a, b, scale_a, scale_b):
                out = torch.nn.functional.silu(in1) * in2
                out_row = (out / out.amax(dim=1, keepdim=True)).to(torch.float8_e4m3fn)
                out_col = (out / out.amax(dim=0, keepdim=True)).to(torch.float8_e4m3fn)

                # setup strides for _scaled_mm
                out_row = out_row.contiguous()
                out_col = out_col.t().contiguous().t()

                return (
                    torch._scaled_mm(
                        out_row, a, scale_a, scale_b, out_dtype=torch.bfloat16
                    ),
                    torch._scaled_mm(
                        b, out_col, scale_a, scale_b, out_dtype=torch.bfloat16
                    ),
                )

            in1 = torch.randn((bs, dim), dtype=torch.bfloat16, device=GPU_TYPE)
            in2 = torch.randn((bs, dim), dtype=torch.bfloat16, device=GPU_TYPE)
            a = (
                torch.randn((dim, dim), dtype=torch.bfloat16, device=GPU_TYPE)
                .t()
                .to(torch.float8_e4m3fn)
            )
            b = torch.randn((dim, bs), dtype=torch.bfloat16, device=GPU_TYPE).to(
                torch.float8_e4m3fn
            )
            # Scales
            scale_a = torch.tensor(1.0, device=GPU_TYPE)
            scale_b = torch.tensor(1.0, device=GPU_TYPE)

            # warmup
            _, (wrapper,) = run_and_get_code(f, in1, in2, a, b, scale_a, scale_b)

            # Previously indcutor decide reduction hint for a reduction kernel without considering
            # the pointwise nodes. That will cause the third reduction kernel in this wrapper to be a
            # persistent inner reduction and cause bad perf.
            #
            # We fix that by making the third reduction a non-persistent reduction
            # and improve the perf by 4.14x (451us -> 109us)
            self.assertEqual(3, wrapper.count("def triton_red_"))
            self.assertEqual(0, wrapper.count("def triton_per_"))

            if DO_PERF_TEST:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as p:
                    for _ in range(1000):
                        f(in1, in2, a, b, scale_a, scale_b)

                print(p.key_averages().table(max_name_column_width=200))

    class RNNTest(TestCase):
        device_type = GPU_TYPE

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(16, 16, batch_first=True)

            def forward(self, x):
                return self.gru(x)

        @expectedFailureXPU
        def test_rnn_compile_safe(self):
            device = torch.device(GPU_TYPE)
            model = RNNTest.Model().to(device)
            model = torch._dynamo.optimize("inductor")(model)
            x = torch.rand(1024, 20, 16).to(device)
            model(x)

    class NanCheckerTest(TestCase):
        @config.patch("nan_asserts", True)
        def test_nan_checker_pass(self):
            def f(x):
                return torch.softmax(x, dim=-1)

            x = torch.randn(2, 1024, device=GPU_TYPE)
            ref = f(x)
            actual, (code,) = run_and_get_code(torch.compile(f), x)
            self.assertTrue(torch.allclose(ref, actual))
            self.assertTrue("# make sure graph inputs are not nan/inf" in code)
            self.assertTrue(
                re.search(r"assert not .*\.isnan\(\)\.any\(\).item\(\)", code)
                is not None
            )
            self.assertTrue(
                re.search(r"assert not .*\.isinf\(\)\.any\(\).item\(\)", code)
                is not None
            )

        @config.patch("nan_asserts", True)
        def test_nan_checker_fail(self):
            def f(x):
                return torch.softmax(x, dim=-1)

            x = torch.randn(2, 1024, device=GPU_TYPE)
            x[0, 0] = float("nan")
            with self.assertRaises(AssertionError):
                torch.compile(f)(x)


if HAS_CPU:

    class TestFull(TestCase):
        @skipIfWindows
        def test_full_dtype(self):
            pytypes = (
                bool,
                int,
                float,
                # TODO: Triton's JITFunction._type_of has no support for complex
                # complex,
            )

            dtypes = (
                torch.bool,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
                None,
                # torch.complex64,
                # torch.complex128,
            )

            def fn(pytype, dtype):
                if pytype is bool:
                    fill_value = True
                elif pytype is int:
                    fill_value = 42
                elif pytype is float:
                    fill_value = 42.0
                else:
                    raise AssertionError(f"Unexpected Python type: {pytype}")

                return torch.full(
                    (4, 6), fill_value, dtype=dtype, device=torch.device("cpu")
                )

            fn_opt = torch._dynamo.optimize("inductor")(fn)

            for pytype, dtype in itertools.product(pytypes, dtypes):
                with enable_python_dispatcher():
                    with torch.no_grad():
                        ret_opt = fn_opt(pytype, dtype)

                self.assertEqual(ret_opt, fn(pytype, dtype))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
