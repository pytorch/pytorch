# Owner(s): ["module: inductor"]
"""Tests for strict numerics mode."""

import ast
import hashlib
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import unittest
from contextlib import nullcontext
from unittest import mock


# Native reductions register during import, so enable the rollout first.
os.environ["PYTORCH_SUM_INNER_TREE"] = "1"

import torch
from torch._dynamo import compiled_autograd
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._higher_order_ops import foreach_map
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._native.ops.reductions.inner_tree_plan import (
    compute_inner_tree_params,
    vec_size,
)
from torch._native.registry import native_decomp_table
from torch.testing._internal.common_cuda import IS_SM100, IS_SM89, IS_SM90, SM90OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
    OpDTypes,
    ops,
    skipCUDAIf,
    skipCUDAIfNoCudnn,
)
from torch.testing._internal.common_methods_invocations import (
    foreach_binary_op_db,
    op_db,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    LazyVal,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
)
from torch.testing._internal.inductor_pointwise_coverage import (
    coverage_case,
    CoverageRecord,
    record_result,
    write_event,
    write_manifest,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    generate_elementwise_binary_with_scalar_samples,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map
from torch.utils._triton import has_triton_reduction_ordering


def _singleton_input(device):
    return torch.as_strided(torch.full((1,), -0.0, device=device), (1, 1), (1, 0))


SUM_CASES = (
    ("persistent_fp16", (64, 256), 1, torch.float16),
    ("looped_bf16", (8, 12000), 1, torch.bfloat16),
    ("split_fp32", (8, 65536), 1, torch.float32),
    ("persistent_fp64", (64, 256), 1, torch.float64),
    ("looped_fp64", (8, 12000), 1, torch.float64),
    ("split_fp64", (8, 65536), 1, torch.float64),
)
INNER_TREE_CALL = "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)"

SUM_VARIANTS = (
    ("autotune", (64, 256), 1, torch.float32, False, {"max_autotune": True}),
)

PROD_CASES = (
    ("persistent_fp16", (64, 256), 1, torch.float16),
    ("looped_fp32", (8, 12000), 1, torch.float32),
    ("split_fp32", (8, 65536), 1, torch.float32),
)

DYNAMIC_CASES = (("plan_change", (512, 65537), {}),)

OUT_OF_SCOPE_CASES = (
    ("multidim", lambda z: torch.sum(z, (0, 1))),
    ("dtype", lambda z: torch.sum(z, 1, dtype=torch.float64)),
)

LAYOUT_CASES = (
    (
        "outer_strided",
        lambda d: (torch.randn(128, 300, device=d)[::2], -1),
        True,
    ),
    ("singleton_dim1", lambda d: (_singleton_input(d), 1), True),
    (
        "noncollapsible",
        lambda d: (torch.randn(16, 8, 300, device=d)[::2], -1),
        False,
    ),
)

SIGNED_ZERO_CASES = (
    ("multirow_fp32", 4, 1, True, torch.float32),
    ("persistent_fp32", 64, 128, False, torch.float32),
    ("persistent_fp64", 64, 128, False, torch.float64),
)

FUSION_CASES = (
    "nested",
    "mix",
    "multi_kernel",
    "multi_output",
)

NUMERICS_MODES = ("default", "strict_pointwise", "strict_reduction", "strict")
EFFECTIVE_NUMERICS = (
    "eager_numerics.division_rounding",
    "eager_numerics.disable_ftz",
    "emulate_precision_casts",
)


def _numerics_options(numerics, enabled):
    return {
        "numerics": numerics,
        "strict_pointwise": False,
        "strict_reduction": False,
        **dict.fromkeys(EFFECTIVE_NUMERICS, enabled),
    }


def _effective_numerics():
    return {
        "eager_numerics.division_rounding": config.eager_numerics.division_rounding,
        "eager_numerics.disable_ftz": config.eager_numerics.disable_ftz,
        "emulate_precision_casts": config.emulate_precision_casts,
    }


class ElementwiseSymbolicLayoutTest(TestCase):
    @dtypes(torch.int32, torch.int64)
    @parametrize("cpp_fake", (False, True))
    @parametrize("scalar", (False, True))
    @parametrize("op", ("add", "mul", "div", "le", "eq", "ne", "bitwise_and"))
    def test_contiguous_broadcast_unbacked_strides(
        self, device, dtype, cpp_fake, scalar, op
    ):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        mode = mode_cls(shape_env=shape_env)
        if not cpp_fake:
            mode.cache_enabled = False
        with mode:
            x = torch.empty_strided((u, 1), (1, 8), device=device, dtype=dtype)
            y = (
                1
                if scalar
                else torch.empty_strided(
                    (1, 1), (2, 1), device=device, dtype=torch.int64
                )
            )
            output = getattr(torch, op)(x, y)
        for size in (0, 1, 2, 3):
            x = torch.empty_strided(
                (size, 1), (1, 8), device=device, dtype=dtype
            ).fill_(1)
            y = (
                1
                if scalar
                else torch.empty_strided(
                    (1, 1), (2, 1), device=device, dtype=torch.int64
                ).fill_(1)
            )
            expected = getattr(torch, op)(x, y)
            actual = tuple(
                int(s.node.expr.subs(u.node.expr, size))
                if isinstance(s, torch.SymInt)
                else s
                for s in output.stride()
            )
            with self.subTest(size=size):
                self.assertEqual(actual, expected.stride())
                self.assertEqual(output.dtype, expected.dtype)

    @dtypes(
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("cpp_fake", (False, True))
    @parametrize("arguments", ("scalar_base", "scalar_exponent", "tensors"))
    @parametrize("scalar", (1.0, 2.0, 3.0))
    def test_pow_unbacked_strides(self, device, dtype, cpp_fake, arguments, scalar):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        with mode_cls(shape_env=shape_env):
            x = torch.empty_strided((u, 2), (1, u), device=device, dtype=dtype)
            y = torch.empty_strided((u, 2), (2, 1), device=device, dtype=dtype)
            a = scalar if arguments == "scalar_base" else x
            b = (
                scalar
                if arguments == "scalar_exponent"
                else y
                if arguments == "tensors"
                else x
            )
            output = torch.pow(a, b)
        for size in (0, 1, 2, 3):
            x = torch.empty_strided(
                (size, 2), (1, size), device=device, dtype=dtype
            ).fill_(1)
            y = torch.empty_strided(
                (size, 2), (2, 1), device=device, dtype=dtype
            ).fill_(2)
            a = scalar if arguments == "scalar_base" else x
            b = (
                scalar
                if arguments == "scalar_exponent"
                else y
                if arguments == "tensors"
                else x
            )
            actual = tuple(
                int(s.node.expr.subs(u.node.expr, size))
                if isinstance(s, torch.SymInt)
                else s
                for s in output.stride()
            )
            with self.subTest(size=size):
                self.assertEqual(actual, torch.pow(a, b).stride())

    @parametrize("cpp_fake", (False, True))
    def test_unbacked_strides_contiguous_device_policy(self, device, cpp_fake):
        from unittest.mock import patch

        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        with (
            mode_cls(shape_env=shape_env),
            patch("torch._refs.is_noncontiguous_supported", return_value=False),
        ):
            x = torch.empty_strided((u, 2), (1, u), device=device)
            output = torch._refs.neg(x)
        self.assertEqual(output.stride(), (2, 1))

    @parametrize("cpp_fake", (False, True))
    @parametrize(
        "dtype_pair",
        (
            ("float16", "float16"),
            ("int32", "float32"),
            ("float16", "float32"),
            ("float32", "float64"),
        ),
    )
    @parametrize("pred_stride", ((0, 0, 0), (0, 3, 1), (1, 3, 0)))
    @parametrize("pred_dtype", (torch.bool, torch.uint8))
    def test_where_unbacked_strides(
        self, device, cpp_fake, dtype_pair, pred_stride, pred_dtype
    ):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        left_dtype, right_dtype = (getattr(torch, name) for name in dtype_pair)
        with mode_cls(shape_env=shape_env):
            pred = torch.empty_strided(
                (u, 2, 2), pred_stride, device=device, dtype=pred_dtype
            )
            x = torch.empty_strided(
                (u, 2, 2), (0, 3, 1), device=device, dtype=left_dtype
            )
            y = torch.empty_strided(
                (u, 2, 2), (1, 3, 0), device=device, dtype=right_dtype
            )
            output = torch.where(pred, x, y)
        for size in (0, 1, 2):
            pred = torch.empty_strided(
                (size, 2, 2), pred_stride, device=device, dtype=pred_dtype
            )
            x = torch.empty_strided(
                (size, 2, 2), (0, 3, 1), device=device, dtype=left_dtype
            )
            y = torch.empty_strided(
                (size, 2, 2), (1, 3, 0), device=device, dtype=right_dtype
            )
            actual = tuple(
                int(s.node.expr.subs(u.node.expr, size))
                if isinstance(s, torch.SymInt)
                else s
                for s in output.stride()
            )
            with self.subTest(size=size):
                self.assertEqual(actual, torch.where(pred, x, y).stride())

    @dtypes(torch.int32, torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("cpp_fake", (False, True))
    @parametrize("op", ("eq", "ne", "lt", "le", "gt", "ge"))
    def test_scalar_unbacked_strides(self, device, dtype, cpp_fake, op):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        with mode_cls(shape_env=shape_env):
            x = torch.empty_strided((u, 2, 2), (0, 3, 1), device=device, dtype=dtype)
            output = getattr(torch, op)(x, 1.5)
        for size in (0, 1, 2):
            x = torch.empty_strided((size, 2, 2), (0, 3, 1), device=device, dtype=dtype)
            actual = tuple(
                int(s.node.expr.subs(u.node.expr, size))
                if isinstance(s, torch.SymInt)
                else s
                for s in output.stride()
            )
            with self.subTest(size=size):
                self.assertEqual(actual, getattr(torch, op)(x, 1.5).stride())

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    @parametrize("cpp_fake", (False, True))
    @parametrize("op", ("bitwise_and", "bitwise_or", "bitwise_xor"))
    def test_bitwise_unbacked_strides(self, device, dtype, cpp_fake, op):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        with mode_cls(shape_env=shape_env):
            x = torch.empty_strided((u, 2, 2), (0, 3, 1), device=device, dtype=dtype)
            y = torch.empty_strided(
                (u, 2, 2), (1, 3, 0), device=device, dtype=torch.int64
            )
            output = getattr(torch, op)(x, y)
        for size in (0, 1, 2):
            x = torch.empty_strided((size, 2, 2), (0, 3, 1), device=device, dtype=dtype)
            y = torch.empty_strided(
                (size, 2, 2), (1, 3, 0), device=device, dtype=torch.int64
            )
            actual = tuple(
                int(s.node.expr.subs(u.node.expr, size))
                if isinstance(s, torch.SymInt)
                else s
                for s in output.stride()
            )
            with self.subTest(size=size):
                self.assertEqual(actual, getattr(torch, op)(x, y).stride())

    @parametrize("cpp_fake", (False, True))
    @parametrize(
        "dtype_pair",
        (
            ("int32", "float32"),
            ("float16", "float32"),
            ("bfloat16", "float16"),
            ("float32", "float64"),
        ),
    )
    @parametrize("op", ("add", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"))
    def test_mixed_dtype_unbacked_strides(self, device, cpp_fake, dtype_pair, op):
        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        left_dtype, right_dtype = (getattr(torch, name) for name in dtype_pair)
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        layouts = (
            ((0, 3, 1), (1, 3, 0)),
            ((1, 2 * u, u), (4, 1, 2)),
            ((u + 1, 3, 1), (1, u + 2, 3)),
        )
        for left_stride, right_stride in layouts:
            with mode_cls(shape_env=shape_env):
                x = torch.empty_strided(
                    (u, 2, 2), left_stride, device=device, dtype=left_dtype
                )
                y = torch.empty_strided(
                    (u, 2, 2), right_stride, device=device, dtype=right_dtype
                )
                output = getattr(torch, op)(x, y)
            for size in (0, 1, 2):

                def evaluate(value):
                    return (
                        int(value.node.expr.subs(u.node.expr, size))
                        if isinstance(value, torch.SymInt)
                        else value
                    )

                x = torch.empty_strided(
                    (size, 2, 2),
                    tuple(map(evaluate, left_stride)),
                    device=device,
                    dtype=left_dtype,
                )
                y = torch.empty_strided(
                    (size, 2, 2),
                    tuple(map(evaluate, right_stride)),
                    device=device,
                    dtype=right_dtype,
                )
                with self.subTest(
                    left_stride=left_stride, right_stride=right_stride, size=size
                ):
                    self.assertEqual(
                        tuple(map(evaluate, output.stride())),
                        getattr(torch, op)(x, y).stride(),
                    )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("cpp_fake", (False, True))
    def test_unbacked_output_strides(self, device, dtype, cpp_fake):
        from functools import lru_cache

        from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        layouts = {
            "rank3_holes": lambda m, n: ((m, n, 2), (n + 1, m + 1, 2)),
            "rank3_stride_order": lambda m, n: ((m, n, 2), (2 * n + 1, 2 * m + 1, 3)),
            "rank4_holes": lambda m, n: ((2, m, 2, n), (11, 3, n + 1, m + 1)),
            "transpose": lambda m, n: ((m, n), (1, m)),
            "prefix": lambda m, n: ((2, 2, m, n), (2 * m * n, m * n, 1, m)),
            "channels_last": lambda m, n: ((2, m, 2, n), (2 * m * n, 1, m * n, m)),
            "channels_singleton": lambda m, n: ((2, m, 1, n), (m * n, 1, 9 * m * n, m)),
            "holes": lambda m, n: ((m, n), (2, 3 * m + 1)),
            "dynamic_order": lambda m, n: ((m, n), (n + 1, m + 1)),
        }
        operations = {
            "add": lambda x: torch.add(x, x),
            "mul": lambda x: torch.mul(x, x),
            "neg": torch.neg,
            "add_scalar": lambda x: torch.add(x, 1),
            "mul_scalar": lambda x: torch.mul(x, 2),
            "sub_scalar": lambda x: torch.sub(x, 1),
            "div_scalar": lambda x: torch.div(x, 2),
            "pow_scalar": lambda x: torch.pow(x, 2),
            "clamp": lambda x: torch.clamp(x, 0, 1),
            "round": torch.round,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "max": lambda x: torch.maximum(x, x),
            "eq": lambda x: torch.eq(x, x),
            "addcmul": lambda x: torch.addcmul(x, x, x),
            "addcdiv": lambda x: torch.addcdiv(x, x, x),
            "where": lambda x: torch.where(
                torch.empty_strided(
                    x.shape, x.stride(), device=x.device, dtype=torch.bool
                ),
                x,
                x,
            ),
            "lerp": lambda x: torch.lerp(x, x, 0.5),
        }
        mode_cls = CppFakeTensorMode if cpp_fake else FakeTensorMode
        for layout, geometry in layouts.items():
            shape_env = ShapeEnv()
            m, n = (shape_env.create_unbacked_symint() for _ in range(2))
            shape, stride = geometry(m, n)
            with mode_cls(shape_env=shape_env):
                fake = torch.empty_strided(shape, stride, device=device, dtype=dtype)
                outputs = {name: op(fake) for name, op in operations.items()}
            for a, b in itertools.product((0, 1, 2), repeat=2):
                shape, stride = geometry(a, b)
                x = torch.empty_strided(shape, stride, device=device, dtype=dtype)
                replacements = {m.node.expr: a, n.node.expr: b}

                @lru_cache(None)
                def evaluate(expr):
                    if expr in replacements:
                        return replacements[expr]
                    args = tuple(evaluate(arg) for arg in expr.args)
                    return expr if args == expr.args else expr.func(*args)

                for name, op in operations.items():
                    with self.subTest(layout=layout, op=name, m=a, n=b):
                        actual = tuple(
                            int(evaluate(s.node.expr))
                            if isinstance(s, torch.SymInt)
                            else s
                            for s in outputs[name].stride()
                        )
                        self.assertEqual(actual, op(x).stride())


instantiate_device_type_tests(ElementwiseSymbolicLayoutTest, globals())


class ElementwiseStrideCodegenTest(TestCase):
    @skipCUDAIf(not HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @parametrize("rank", (1, 2, 3, 4, 5))
    @parametrize("mode", ("unary", "binary", "scalar", "cast"))
    def test_native_strides(self, device, rank, mode):
        import ctypes

        import sympy

        from torch._inductor.codecache import CppCodeCache
        from torch.utils._sympy.functions import ElementwiseOutputStride
        from torch.utils._sympy.printers import CppPrinter

        inputs = 2 if mode == "binary" else 1
        allow_fast = mode != "scalar"
        preserve_dense = mode == "cast"
        cases = []
        for m, n in itertools.product((0, 1, 2), repeat=2):
            if rank == 1:
                cases.append(((m,), (n + 1,), (n + 2,)))
            else:
                shape = (*(2,) * (rank - 2), m, n)
                prefix = tuple(m * n * 2**i for i in reversed(range(rank - 2)))
                cases.append((shape, (*prefix, n + 1, m + 1), (*prefix, n + 2, m + 2)))
                cases.append((shape, (*prefix, 1, m), (*prefix, n, 1)))
        if rank == 4:
            for m, n in itertools.product((0, 1, 2), repeat=2):
                shape = (2, m, 2, n)
                step = (2 * m * n, 1, m * n, m)
                cases.append((shape, step, step))
        shape = (1,) * rank
        step = (1 << 35, *(1,) * (rank - 1))
        cases.append((shape, step, step))
        if rank > 1:
            shape = (0, *(1,) * (rank - 2), 1 << 35)
            step = tuple(torch.empty(shape, device=device).stride())
            cases.append((shape, step, step))
        rows = []
        expected = []
        for shape, step, other_step in cases:
            x = torch.empty_strided(shape, step, device=device, dtype=torch.float32)
            if mode == "binary":
                y = torch.empty_strided(
                    shape, other_step, device=device, dtype=torch.float32
                )
                result = torch.add(x, y)
            elif mode == "scalar":
                result = torch.add(x, 1)
            elif mode == "cast":
                result = x.to(torch.float64)
            else:
                result = torch.neg(x)
            rows.append(
                [*shape, *step, *other_step, int(allow_fast), int(preserve_dense)]
                if inputs == 2
                else [*shape, *step, int(allow_fast), int(preserve_dense)]
            )
            expected.append(result.stride())
        data = torch.tensor(rows, device=device, dtype=torch.int64)
        output = torch.empty((len(rows), rank), device=device, dtype=torch.int64)
        if device.startswith("cuda"):
            import triton
            import triton.language as tl

            from torch._inductor.runtime.triton_helpers import elementwise_output_stride

            @triton.jit
            def stride_kernel(data, output, RANK: tl.constexpr, INPUTS: tl.constexpr):
                row = tl.program_id(0)
                sizes = ()
                for i in tl.static_range(RANK):
                    sizes += (tl.load(data + row * (RANK * (INPUTS + 1) + 2) + i),)
                strides = ()
                for t in tl.static_range(INPUTS):
                    entry = ()
                    for i in tl.static_range(RANK):
                        entry += (
                            tl.load(
                                data
                                + row * (RANK * (INPUTS + 1) + 2)
                                + RANK * (t + 1)
                                + i
                            ),
                        )
                    strides += (entry,)
                allow_fast = tl.load(
                    data + row * (RANK * (INPUTS + 1) + 2) + RANK * (INPUTS + 1)
                )
                preserve_dense = tl.load(
                    data + row * (RANK * (INPUTS + 1) + 2) + RANK * (INPUTS + 1) + 1
                )
                for i in tl.static_range(RANK):
                    value = elementwise_output_stride(
                        sizes, strides, i, allow_fast, preserve_dense
                    )
                    tl.store(output + row * RANK + i, value)

            stride_kernel[(len(rows),)](data, output, rank, inputs)
        else:
            width = rank * (inputs + 1) + 2
            args = tuple(
                sympy.Symbol(f"data[row * {width} + {i}]", integer=True)
                for i in range(width)
            )
            statements = []
            for dim in range(rank):
                expr = ElementwiseOutputStride(
                    dim, rank, args[-2], args[-1], *args[:-2]
                )
                statements.append(
                    f"output[row * {rank} + {dim}] = {CppPrinter().doprint(expr)};"
                )
            code = (
                '#include <cstdint>\nextern "C" void evaluate(const int64_t* data, int64_t* output, int64_t rows) { for (int64_t row=0; row<rows; ++row) {'
                + "\n".join(statements)
                + "}}"
            )
            lib = CppCodeCache.load(code)
            lib.evaluate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            lib.evaluate(data.data_ptr(), output.data_ptr(), len(rows))
        self.assertEqual(
            output, torch.tensor(expected, device=device, dtype=torch.int64)
        )


instantiate_device_type_tests(
    ElementwiseStrideCodegenTest, globals(), only_for=("cpu", "cuda")
)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None,
    "requires NVIDIA CUDA and Triton",
)
class CompiledSymbolicLayoutTest(TestCase):
    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("views", ("repeated", "distinct", "base", "input"))
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_output_aliases(self, device, dtype, views, numerics):
        def fn(x, mask):
            n = torch.nonzero(mask).shape[0]
            y = x[:n].transpose(0, 1)
            if views != "input":
                y = torch.neg(y)
            if views == "base":
                return y, y.transpose(0, 1)
            if views == "distinct":
                return y.transpose(0, 1), y[:, :1]
            return y.transpose(0, 1), y.transpose(0, 1)

        counter = CompileCounterWithBackend("inductor")
        with config.patch(
            {
                "numerics": numerics,
                "force_disable_caches": True,
                "triton.cudagraphs": False,
            }
        ):
            compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
            for n in (0, 1, 2, 3):
                x = torch.ones((4, 2), device=device, dtype=dtype)
                mask = torch.arange(4, device=device) < n
                expected = fn(x.clone(), mask)
                actual = compiled(x.clone(), mask)
                for a, b in zip(actual, expected):
                    self.assertEqual(
                        _pointwise_comparison_bits(a), _pointwise_comparison_bits(b)
                    )
                    self.assertEqual(a.stride(), b.stride())
                self.assertTrue(torch._C._is_alias_of(*actual))
                expected[0].fill_(7)
                actual[0].fill_(7)
                self.assertEqual(
                    _pointwise_comparison_bits(actual[1]),
                    _pointwise_comparison_bits(expected[1]),
                )
        self.assertGreater(counter.frame_count, 0)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("op", ("add", "mul", "neg"))
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_layouts(self, device, dtype, op, numerics):
        layouts = {
            "rank3_holes": lambda m, n: ((m, n, 2), (n + 1, m + 1, 2)),
            "rank3_stride_order": lambda m, n: ((m, n, 2), (2 * n + 1, 2 * m + 1, 3)),
            "rank4_holes": lambda m, n: ((2, m, 2, n), (11, 3, n + 1, m + 1)),
            "transpose": lambda m, n: ((m, n), (1, m)),
            "prefix": lambda m, n: ((2, 2, m, n), (2 * m * n, m * n, 1, m)),
            "channels_last": lambda m, n: ((2, m, 2, n), (2 * m * n, 1, m * n, m)),
            "channels_singleton": lambda m, n: ((2, m, 1, n), (m * n, 1, 9 * m * n, m)),
            "holes": lambda m, n: ((m, n), (2, 3 * m + 1)),
            "dynamic_order": lambda m, n: ((m, n), (n + 1, m + 1)),
        }
        for layout, geometry in layouts.items():
            torch._dynamo.reset()

            def fn(mask1, mask2):
                m = torch.nonzero(mask1).shape[0]
                n = torch.nonzero(mask2).shape[0]
                size, stride = geometry(m, n)
                x = torch.empty_strided(size, stride, device=device, dtype=dtype)
                x.fill_(0.5)
                out = torch.neg(x) if op == "neg" else getattr(torch, op)(x, x)
                return out, out + 1, out.stride()

            counter = CompileCounterWithBackend("inductor")
            metrics.reset()
            with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
                compiled = torch.compile(
                    fn, backend=counter, fullgraph=True, dynamic=True
                )
                for m, n in itertools.product((0, 1, 2), repeat=2):
                    with self.subTest(layout=layout, m=m, n=n):
                        mask1 = torch.arange(4, device=device) < m
                        mask2 = torch.arange(4, device=device) < n
                        expected, actual = fn(mask1, mask2), compiled(mask1, mask2)
                        for a, b in zip(actual[:2], expected[:2]):
                            self.assertEqual(
                                _pointwise_comparison_bits(a),
                                _pointwise_comparison_bits(b),
                            )
                            self.assertEqual(a.stride(), b.stride())
                        self.assertEqual(actual[2], expected[2])
                        self.assertEqual(actual[2], actual[0].stride())
            self.assertEqual(counter.frame_count, 1)
            self.assertGreater(metrics.generated_kernel_count, 0)


instantiate_device_type_tests(CompiledSymbolicLayoutTest, globals(), only_for="cuda")


@instantiate_parametrized_tests
class StrictNumericsConfigTest(TestCase):
    @parametrize("geometry", ("complex64", "complex128", "strided", "singleton_gaps"))
    def test_cuda_pointwise_byte_chunks(self, geometry):
        from torch._inductor.inductor_prims import (
            _cuda_pointwise_chunk_index,
            _cuda_pointwise_chunks,
        )

        if geometry == "singleton_gaps":
            shape, strides = (3, 2), ((8, 24), (16, 2**31), (2**31, 2**30))
            chunks = _cuda_pointwise_chunks(shape, strides)
            self.assertEqual(len(chunks), 6)
            for physical in range(0, 48, 8):
                matches = []
                for chunk in chunks:
                    coords, remainder, local, count = _cuda_pointwise_chunk_index(
                        physical, chunk
                    )
                    if (
                        physical >= chunk.offsets[0]
                        and remainder == 0
                        and all(c < n for c, n in zip(coords, chunk.shape))
                    ):
                        matches.append((local, count))
                self.assertEqual(matches, [(0, 1)])
            return
        itemsize = 16 if geometry == "complex128" else 8
        input_stride = 16 if geometry == "strided" else itemsize
        count = (2**27 if input_stride == 16 else 2**28) + 259
        chunks = _cuda_pointwise_chunks(
            (count,), ((itemsize,), (input_stride,), (input_stride,))
        )
        self.assertEqual(
            [c.shape for c in chunks], [(count // 2,), (count - count // 2,)]
        )
        self.assertEqual(
            chunks[1].offsets,
            (
                count // 2 * itemsize,
                count // 2 * input_stride,
                count // 2 * input_stride,
            ),
        )
        for offset in (0, 127, 128, 255, 256):
            coords, remainder, local, size = _cuda_pointwise_chunk_index(
                chunks[1].offsets[0] + offset * itemsize, chunks[1]
            )
            self.assertEqual(
                (coords, remainder, local, size),
                ([offset], 0, offset, count - count // 2),
            )

    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA Triton backend")
    @parametrize("strict_pointwise", (False, True))
    @parametrize("device_type", ("cuda", "hip", "xpu"))
    def test_compile_options_require_scoped_libdevice_support(
        self, strict_pointwise, device_type
    ):
        import triton

        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        tuner = mock.Mock(device_props=mock.Mock(type=device_type))
        cfg = triton.Config({})
        meta = {
            "num_warps": 4,
            "num_stages": 1,
            "debug": False,
            "strict_pointwise": strict_pointwise,
        }
        with mock.patch(
            "triton.backends.nvidia.compiler.CUDAOptions",
            new=type("CUDAOptions", (), {}),
        ):
            if device_type == "cuda" and strict_pointwise:
                with self.assertRaisesRegex(
                    RuntimeError, "libdevice_fp64_fusion_roots"
                ):
                    CachingAutotuner._create_compile_options(tuner, cfg, meta)
            else:
                options = CachingAutotuner._create_compile_options(tuner, cfg, meta)
                self.assertNotIn("libdevice_fp64_fusion_roots", options)

    @parametrize("numerics", NUMERICS_MODES)
    @config.patch({"triton.enable_pdl": False})
    def test_config_patch_enables_eager_numerics(self, numerics):
        from torch._inductor.codegen.triton import TritonKernel

        enabled = numerics in ("strict_pointwise", "strict")
        reduction_enabled = numerics in ("strict_reduction", "strict")
        with config.patch(_numerics_options("strict", False)):
            with config.patch(numerics=numerics):
                self.assertEqual(config.strict_pointwise, enabled)
                self.assertEqual(config.strict_reduction, reduction_enabled)
                self.assertEqual(
                    TritonKernel.triton_meta_common()["strict_pointwise"], enabled
                )
                self.assertEqual(
                    _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, enabled)
                )
            self.assertTrue(config.strict_pointwise)
            self.assertTrue(config.strict_reduction)
            self.assertEqual(
                _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, True)
            )
        with config.patch(_numerics_options(numerics, True)):
            self.assertEqual(config.strict_pointwise, enabled)
            self.assertEqual(config.strict_reduction, reduction_enabled)
            self.assertEqual(
                _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, True)
            )

    @parametrize("enabled", (False, True))
    def test_strict_pointwise_enables_eager_numerics(self, enabled):
        with config.patch(_numerics_options("default", False)):
            with config.patch(strict_pointwise=enabled):
                self.assertEqual(config.numerics, "default")
                self.assertEqual(
                    _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, enabled)
                )
            self.assertFalse(config.strict_pointwise)
            self.assertEqual(
                _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, False)
            )

    @parametrize("numerics", NUMERICS_MODES)
    def test_env_enables_eager_numerics(self, numerics):
        enabled = numerics in ("strict_pointwise", "strict")
        reduction_enabled = numerics in ("strict_reduction", "strict")
        env = os.environ.copy()
        env["TORCHINDUCTOR_NUMERICS"] = numerics
        env["TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING"] = "0"
        env["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = "0"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "from torch._inductor import config; "
                    "print(config.strict_reduction, config.strict_pointwise, "
                    "config.eager_numerics.division_rounding, "
                    "config.eager_numerics.disable_ftz, "
                    "config.emulate_precision_casts)"
                ),
            ],
            env=env,
            text=True,
        )
        self.assertEqual(
            output.split(),
            [str(reduction_enabled)] + [str(enabled)] * (1 + len(EFFECTIVE_NUMERICS)),
        )


def _compile_without_decompositions(gm, inputs):
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, inputs, decompositions={})


class StrictNumericsFallbackTest(TestCase):
    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_foreach_inplace_tensor_list(self, device, dtype, numerics):
        def fn(a, b):
            return torch._foreach_add_([a, b], 1)

        a = torch.arange(4, device=device, dtype=dtype)
        b = torch.ones_like(a)
        eager_args = (a.clone(), b.clone())
        expected = fn(*eager_args)
        actual = torch.compile(fn, fullgraph=True, options={"numerics": numerics})(a, b)
        self.assertEqual(actual, expected)
        self.assertEqual((a, b), eager_args)

    @ops(
        [op for op in op_db if op.name == "logit"],
        allowed_dtypes=(torch.bool, torch.int64),
    )
    @parametrize("numerics", NUMERICS_MODES)
    def test_logit_integer_promotion(self, device, dtype, op, numerics):
        for sample in op.sample_inputs(device, dtype, requires_grad=False):

            def fn(x):
                return op.op(x, *sample.args, **sample.kwargs)

            actual = torch.compile(fn, fullgraph=True, options={"numerics": numerics})(
                sample.input
            )
            expected = fn(sample.input)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual, expected)

    @dtypes(torch.float32, torch.float64)
    @parametrize("numerics", NUMERICS_MODES)
    def test_log_ndtr_without_decompositions(self, device, dtype, numerics):
        x = _min_max_specials(dtype, device)
        with config.patch(numerics=numerics):
            actual = torch.compile(
                torch.special.log_ndtr,
                backend=_compile_without_decompositions,
                fullgraph=True,
            )(x)
        self.assertTrue(_outputs_equal(actual, torch.special.log_ndtr(x)))

    @dtypes(torch.float32, torch.float64)
    @parametrize("numerics", ("default", "strict_reduction"))
    @parametrize("decimals", (-3, 3))
    def test_round_decimals_without_decompositions(
        self, device, dtype, numerics, decimals
    ):
        def fn(x):
            return torch.round(x, decimals=decimals)

        x = torch.arange(-1000, 1000, device=device, dtype=dtype) / 777
        with config.patch(numerics=numerics):
            actual = torch.compile(
                fn, backend=_compile_without_decompositions, fullgraph=True
            )(x)
        self.assertTrue(_outputs_equal(actual, fn(x)))

    @ops(
        [op for op in op_db if op.name in ("fmin", "fmax")],
        allowed_dtypes=(torch.float32,),
    )
    @parametrize("numerics", NUMERICS_MODES)
    def test_without_decompositions(self, device, dtype, op, numerics):
        x = _min_max_specials(dtype, device)
        y = x.roll(1)
        with config.patch(numerics=numerics):
            result = torch.compile(
                op.op, backend=_compile_without_decompositions, fullgraph=True
            )(x, y)
        self.assertTrue(_outputs_equal(result, op.op(x, y)))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None,
    "requires NVIDIA CUDA and Triton",
)
class StrictNumericsCompileTest(TestCase):
    def _assert_bitwise_equal(self, actual, expected, *, msg=None):
        from torch.utils._pytree import tree_map

        def bits(t):
            return t.dtype, t.shape, _pointwise_comparison_bits(t)

        self.assertEqual(
            tree_map(bits, actual),
            tree_map(bits, expected),
            msg=msg,
        )

    @largeTensorTest(
        lambda self, device, dtype, kind: 6 << 30 if kind == "split" else 0,
        inductor=True,
    )
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("kind", ("native", "split", "symbolic", "dynamic"))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_generated_sum_strided_program_grid(self, device, dtype, kind):
        from contextlib import ExitStack

        from torch._inductor.kernel import cuda_native_sum as kernels

        grids = (
            kernels.cuda_native_sum_grid,
            kernels.cuda_dynamic_sum_grid,
            kernels.cuda_symbolic_sum_grid,
        )
        meta = {"M": 1 << 31, "MAX_CTAS": 1, "FINAL": True}
        self.assertLessEqual(grids[0](meta)[0], 65535)
        self.assertLessEqual(grids[2](1 << 31, 1, 1, 1, meta)[0], 65535)
        meta["FINAL"] = False
        self.assertLessEqual(grids[1](1 << 31, 2, meta)[0], 65535)
        self.assertLessEqual(grids[2](1, 1 << 31, 2, 2, meta)[0], 65535)

        if kind == "split":
            stride = (1 << 30) // torch.empty((), dtype=dtype).element_size()
            x = torch.empty_strided((5, 7), (stride, 1), device=device, dtype=dtype)
            x.fill_(1)
        elif kind == "dynamic":
            x = torch.randn(262144, device=device, dtype=dtype)
        else:
            x = torch.randn((127, 7), device=device, dtype=dtype)
        mask = torch.ones(x.size(0), device=device, dtype=torch.bool)

        def fn(x, mask):
            if kind in ("dynamic", "symbolic"):
                count = mask.nonzero().size(0)
                x = x.as_strided((count, *x.shape[1:]), x.stride())
            return x.sum(0)

        with ExitStack() as stack:
            for grid in grids:
                original = grid.fn

                def capped_grid(*args, _original=original, **kwargs):
                    gx, gy, gz = _original(*args, **kwargs)
                    return kwargs["min"](gx, 2), gy, gz

                stack.enter_context(mock.patch.object(grid, "fn", capped_grid))
            stack.enter_context(config.patch(numerics="strict", fx_graph_cache=False))
            actual = torch.compile(fn, fullgraph=True)(x, mask)
        self._assert_bitwise_equal(actual, fn(x, mask))

    @dtypes(torch.float32, torch.float64)
    @parametrize("inner_tree", (False, True))
    def test_generated_sum_native_override_route(self, device, dtype, inner_tree):
        table = {torch.ops.aten.sum.dim_IntList: object()} if inner_tree else {}
        x = torch.ones((2, 257), device=device, dtype=dtype)

        def fn(x):
            return x.sum(1)

        metrics.reset()
        with (
            mock.patch(
                "torch._native.registry.native_decomp_table", return_value=table
            ),
            config.patch(numerics="strict", fx_graph_cache=False),
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self._assert_bitwise_equal(actual, fn(x))
        self.assertEqual(INNER_TREE_CALL in "\n".join(codes), inner_tree)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("width", (3, 5))
    @parametrize("layout", ("vector", "reduce_all", "reduce_outer", "reduce_inner"))
    @largeTensorTest("6GB", inductor=True)
    def test_generated_sum_byte_offset_split(self, device, dtype, width, layout):
        stride = (1 << 30) // torch.empty((), dtype=dtype).element_size()
        shape, strides = (
            ((width,), (stride,)) if layout == "vector" else ((width, 2), (stride, 1))
        )
        x = torch.empty_strided(shape, strides, device=device, dtype=dtype)
        x.fill_(0)
        large = 1e4 if dtype == torch.float16 else 1e20
        x[0], x[1], x[2] = 1, large, -large
        if width == 5:
            x[3], x[4] = 2, 3
        axis = (
            0 if layout == "reduce_outer" else 1 if layout == "reduce_inner" else None
        )

        def fn(x):
            return x.sum(dim=axis)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("width", (3, 5))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @largeTensorTest("6GB", inductor=True)
    def test_generated_unbacked_sum_byte_offset_split(self, device, dtype, width):
        from torch._inductor import ir

        stride = (1 << 30) // torch.empty((), dtype=dtype).element_size()
        x = torch.empty_strided((width,), (stride,), device=device, dtype=dtype)
        x.fill_(0)
        large = 1e4 if dtype == torch.float16 else 1e20
        x[0], x[1], x[2] = 1, large, -large
        if width == 5:
            x[3], x[4] = 2, 3

        def fn(x, mask):
            count = mask.nonzero().size(0)
            return x.as_strided((count,), (stride,)).sum()

        fallback_ops = []
        original = ir.FallbackKernel.codegen
        for numerics in ("default", "strict"):
            numeric = set()

            def record(kernel, wrapper):
                numeric.add(str(kernel.op_overload))
                return original(kernel, wrapper)

            with mock.patch.object(ir.FallbackKernel, "codegen", record):
                compiled = torch.compile(
                    fn,
                    fullgraph=True,
                    options={"numerics": numerics, "fx_graph_cache": False},
                )
                for count in range(width + 1):
                    mask = torch.arange(width, device=device) < count
                    actual = compiled(x, mask)
                    if numerics == "strict":
                        self._assert_bitwise_equal(actual, fn(x, mask))
            fallback_ops.append(numeric)
        self.assertLessEqual(fallback_ops[1], fallback_ops[0])

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("axis", (0, 1, (0, 1)))
    @parametrize("transposed", (False, True))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @largeTensorTest("6GB", inductor=True)
    def test_generated_unbacked_sum_byte_offset_split_axes(
        self, device, dtype, axis, transposed
    ):
        from torch._inductor import ir

        stride = (1 << 30) // torch.empty((), dtype=dtype).element_size()
        x = torch.empty_strided((5, 2), (stride, 1), device=device, dtype=dtype)
        self.assertGreater(
            (x.size(0) - 1) * x.stride(0) * x.element_size(),
            torch.iinfo(torch.int32).max,
        )
        x.fill_(0)
        large = 1e4 if dtype == torch.float16 else 1e20
        x[0], x[1], x[2], x[3], x[4] = 1, large, -large, 2, 3

        def fn(x, mask):
            count = mask.nonzero().size(0)
            view = x.as_strided((count, 2), (stride, 1))
            return (view.t() if transposed else view).sum(axis)

        fallback_ops = []
        original = ir.FallbackKernel.codegen
        for numerics in ("default", "strict"):
            numeric = set()

            def record(kernel, wrapper):
                numeric.add(str(kernel.op_overload))
                return original(kernel, wrapper)

            with mock.patch.object(ir.FallbackKernel, "codegen", record):
                compiled = torch.compile(
                    fn,
                    fullgraph=True,
                    options={"numerics": numerics, "fx_graph_cache": False},
                )
                for count in range(6):
                    mask = torch.arange(5, device=device) < count
                    actual = compiled(x, mask)
                    if numerics == "strict":
                        self._assert_bitwise_equal(actual, fn(x, mask))
            fallback_ops.append(numeric)
        self.assertLessEqual(fallback_ops[1], fallback_ops[0])

    @largeTensorTest("6GB", inductor=True)
    @dtypes(torch.float32)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_generated_unbacked_sum_large_output_scratch(self, device, dtype):
        x = torch.zeros((256, 1 << 22), device=device, dtype=dtype)
        x[0], x[1], x[2] = 1, 1e20, -1e20
        mask = torch.ones(256, device=device, dtype=torch.bool)

        def fn(x, mask):
            count = mask.nonzero().size(0)
            return x.as_strided((count, x.size(1)), x.stride()).sum(0)

        expected = fn(x, mask)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x, mask)
        self._assert_bitwise_equal(actual, expected)
        self.assertLess(torch.cuda.max_memory_allocated(device) - baseline, 1 << 29)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_generated_unbacked_vector_sum_stride(self, device, dtype):
        x = torch.randn(1024, device=device, dtype=dtype)

        def fn(x, length_mask, stride_mask):
            length = length_mask.nonzero().size(0)
            stride = stride_mask.nonzero().size(0)
            return x.as_strided((length,), (stride,)).sum()

        with config.patch(numerics="strict", fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True)
            for length, stride in ((0, 0), (1, 0), (3, 1), (8, 3), (16, 8), (16, 16)):
                length_mask = torch.arange(16, device=device) < length
                stride_mask = torch.arange(16, device=device) < stride
                self._assert_bitwise_equal(
                    compiled(x, length_mask, stride_mask),
                    fn(x, length_mask, stride_mask),
                )

    @largeTensorTest("20GB", inductor=True)
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_generated_unbacked_sum_compound_stride(self, device, dtype):
        stride = 1 << 28
        x = torch.empty((8 * stride + 2,), device=device, dtype=dtype)
        x[0], x[1] = 1, 2
        for count in (1, 3, 8):
            x[count * stride], x[count * stride + 1] = 3, 4

        def fn(x, mask):
            count = mask.nonzero().size(0)
            return x.as_strided((2, 2), (count * stride, 1)).sum(0)

        with config.patch(numerics="strict", fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True)
            for count in (0, 1, 3, 8):
                mask = torch.arange(8, device=device) < count
                self._assert_bitwise_equal(compiled(x, mask), fn(x, mask))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("length", (1024, 262144))
    @parametrize("stride", (1, 2))
    def test_generated_unbacked_sum(self, device, dtype, length, stride):
        def fn(x, mask):
            return x[mask][::stride].sum()

        torch.manual_seed(368)
        x = torch.randn(length, device=device, dtype=dtype)
        with (
            torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
            config.patch(numerics="strict", fx_graph_cache=False),
        ):
            compiled = torch.compile(fn, fullgraph=True)
            for count in dict.fromkeys(
                (0, 1, 31, 127, 128, min(length, 4096), length // 2, length)
            ):
                mask = torch.arange(length, device=device) < count
                self._assert_bitwise_equal(compiled(x, mask), fn(x, mask))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("axis", (0, 1, (0, 1)))
    @parametrize("transposed", (False, True))
    def test_generated_unbacked_sum_geometry(self, device, dtype, axis, transposed):
        def fn(x, mask):
            selected = x[mask]
            if transposed:
                selected = selected.t()
            return selected.sum(axis)

        torch.manual_seed(369)
        x = torch.randn(4096, 33, device=device, dtype=dtype)
        with (
            torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
            config.patch(numerics="strict", fx_graph_cache=False),
        ):
            compiled = torch.compile(fn, fullgraph=True)
            for count in (0, 1, 3, 31, 127, 128, 4096):
                mask = torch.arange(x.size(0), device=device) < count
                self._assert_bitwise_equal(compiled(x, mask), fn(x, mask))

    @dtypes(torch.float32, torch.float64)
    def test_generated_unbacked_sum_stride_order(self, device, dtype):
        def fn(x, mask):
            stride = mask.nonzero().size(0)
            return x.as_strided((8, 8), (stride, 3)).sum()

        torch.manual_seed(370)
        x = torch.randn(512, device=device, dtype=dtype)
        with (
            torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
            config.patch(numerics="strict", fx_graph_cache=False),
        ):
            compiled = torch.compile(fn, fullgraph=True)
            for count in (0, 1, 2, 3, 4, 8, 16):
                mask = torch.arange(16, device=device) < count
                self._assert_bitwise_equal(compiled(x, mask), fn(x, mask))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("p", (17, 32, 128, 257))
    @parametrize("transposed", (False, True))
    def test_generated_mvlgamma_larger_p(self, device, dtype, p, transposed):
        x = torch.linspace(
            p / 2 + 1, p / 2 + 30, 257 * 3, device=device, dtype=dtype
        ).reshape(3, 257)
        if transposed:
            x = x.t()

        def fn(x):
            return torch.mvlgamma(x, p)

        with config.patch(numerics="strict", fx_graph_cache=False):
            self._assert_bitwise_equal(torch.compile(fn, fullgraph=True)(x), fn(x))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("shape", ((16, 33), (32, 32), (48, 32)))
    @parametrize(
        "scales",
        (
            (1.0, 1.0),
            (0.0, 0.25),
            (0.75, 0.3),
            (-1.5, 2.0),
            (0.0, math.nan),
            (math.inf, 0.0),
            (1e-50, 0.3),
        ),
    )
    @parametrize("strided", (False, True))
    def test_addr_native_arithmetic(
        self, device, dtype, numerics, shape, scales, strided
    ):
        torch.manual_seed(271)
        n, m = shape
        x = torch.randn(n * 2, m * 2, device=device, dtype=dtype)
        a = x[::2, 1::2] if strided else x[:n, :m].contiguous()
        b = torch.randn(n, device=device, dtype=dtype)
        c = torch.randn(m, device=device, dtype=dtype)
        a[0, :5] = torch.tensor(
            [math.nan, math.inf, -math.inf, -0.0, 0.0], device=device, dtype=dtype
        )
        beta, alpha = scales

        def fn(a, b, c):
            return torch.addr(a, b, c, beta=beta, alpha=alpha)

        expected = fn(a, b, c)
        metrics.reset()
        with config.patch(numerics=numerics, fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(a, b, c)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertGreater(metrics.generated_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_addr_unbacked_scales(self, device, dtype, numerics):
        def fn(a, b, c, scales):
            beta, alpha = scales[0].item(), scales[1].item()
            return torch.addr(a, b, c, beta=beta, alpha=alpha)

        a = torch.randn(16, 33, device=device, dtype=dtype)
        a[0, 0] = math.nan
        b = torch.randn(16, device=device, dtype=dtype)
        c = torch.randn(33, device=device, dtype=dtype)
        with config.patch(numerics=numerics, fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True)
            for beta, alpha in ((0.75, 0.3), (0.0, 0.25), (1e-50, 0.3), (1.0, 1.0)):
                scales = torch.tensor([beta, alpha], device=device, dtype=torch.float64)
                self._assert_bitwise_equal(
                    compiled(a, b, c, scales), fn(a, b, c, scales)
                )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("negative", (False, True))
    def test_generated_copy_and_negative_alias(self, device, dtype, numerics, negative):
        from torch._inductor import ir

        def fn(a, b):
            result = a.clone()
            target = result._neg_view() if negative else result
            target.copy_(b)
            return target + a, target, result

        a = torch.randn(8, 34, device=device, dtype=dtype)[:, 1::2]
        b = torch.randn_like(a)
        expected = fn(a, b)
        metrics.reset()
        with (
            config.patch(numerics=numerics, fx_graph_cache=False),
            mock.patch.object(
                ir.InplaceCopyFallback,
                "codegen",
                side_effect=AssertionError(
                    "real CUDA copies must use generated kernels"
                ),
            ),
        ):
            actual = torch.compile(fn, fullgraph=True)(a, b)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual([t.is_neg() for t in actual], [t.is_neg() for t in expected])
        self.assertTrue(torch._C._is_alias_of(actual[1], actual[2]))
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertGreater(metrics.generated_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "case",
        (
            "global",
            "global_outer",
            "aligned_inner",
            "unaligned_inner",
            "unaligned_outer",
            "singleton",
            "dtype_conversion",
            "negative_zero",
        ),
    )
    def test_generated_cuda_sum_order(self, device, dtype, case):
        torch.manual_seed(273)
        output_dtype = dtype
        if case == "global":
            x = torch.randn(262147, device=device, dtype=dtype)
            dims = None
        elif case == "global_outer":
            x = torch.randn(32771, 17, device=device, dtype=dtype)
            dims = (0,)
        elif case in ("aligned_inner", "unaligned_inner"):
            shift = int(case == "unaligned_inner")
            x = torch.randn(8 * 1025 + shift, device=device, dtype=dtype)[
                shift:
            ].reshape(8, 1025)
            dims = (1,)
            output_dtype = torch.float64 if dtype != torch.float64 else torch.float32
        elif case == "unaligned_outer":
            x = torch.randn(8 * 64 + 1, device=device, dtype=dtype)[1:].reshape(8, 64)
            dims = (0,)
        elif case == "singleton":
            x = torch.randn(8, 1, 33, device=device, dtype=dtype)
            dims = (1,)
        elif case == "dtype_conversion":
            x = torch.randn(8, 16, 33, device=device, dtype=dtype)
            dims = (0, 2)
            output_dtype = torch.float64 if dtype != torch.float64 else torch.float32
        else:
            x = torch.full((129, 33), -0.0, device=device, dtype=dtype)
            dims = (0,)

        def fn(x):
            return x.sum(dim=dims, dtype=output_dtype)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertGreater(metrics.generated_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_cuda_sum_runtime_alignment(self, device, dtype):
        from torch._dynamo.testing import CompileCounterWithBackend

        def fn(x):
            return x.sum(0)

        counter = CompileCounterWithBackend("inductor")
        aligned = torch.randn(8, 64, device=device, dtype=dtype)
        unaligned = torch.randn(8 * 64 + 1, device=device, dtype=dtype)[1:].reshape(
            8, 64
        )
        with config.patch(numerics="strict", fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            for x in (aligned, unaligned, aligned):
                self._assert_bitwise_equal(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("layout", ("contiguous", "channels_last", "strided"))
    @parametrize(
        "weight_bias", ((False, False), (True, False), (False, True), (True, True))
    )
    def test_generated_batch_norm_inference(self, device, dtype, layout, weight_bias):
        torch.manual_seed(274)
        x = torch.randn(4, 16, 8, 10, device=device, dtype=dtype)
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "strided":
            x = x[:, :, :, ::2]
        stat_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        mean = torch.randn(16, device=device, dtype=stat_dtype)
        var = torch.rand(16, device=device, dtype=stat_dtype) + 1
        weight = torch.randn_like(mean) if weight_bias[0] else None
        bias = torch.randn_like(mean) if weight_bias[1] else None

        def fn(x, mean, var, weight, bias):
            return torch.ops.aten._native_batch_norm_legit_no_training.default(
                x, weight, bias, mean, var, 0.1, 1e-5
            )

        expected = fn(x, mean, var, weight, bias)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x, mean, var, weight, bias)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertGreater(metrics.generated_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("shape", ((2, 3, 0, 4), (2, 3, 4, 0)))
    @parametrize("case", ("train_no_stats", "train_stats", "eval_stats"))
    def test_generated_batch_norm_empty_spatial(self, device, dtype, shape, case):
        x = torch.empty(shape, device=device, dtype=dtype)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.ones(3, device=device, dtype=stat_dtype)
        bias = torch.zeros_like(weight)
        mean = torch.randn_like(weight) if case != "train_no_stats" else None
        var = torch.rand_like(weight) if mean is not None else None

        def fn(x, weight, bias, mean, var):
            result = torch.ops.aten.native_batch_norm.default(
                x, weight, bias, mean, var, case != "eval_stats", 0.1, 1e-5
            )
            return result if mean is None else (*result, mean, var)

        expected = fn(
            x,
            weight,
            bias,
            None if mean is None else mean.clone(),
            None if var is None else var.clone(),
        )
        metrics.reset()
        actual = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})(
            x, weight, bias, mean, var
        )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_pointwise_unbacked_output_strides_backward(self, device, dtype, numerics):
        def fn(x, mask):
            n = torch.nonzero(mask).shape[0]
            return torch.neg(x[:n].transpose(0, 1))

        counter = CompileCounterWithBackend("inductor")
        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
            for n in (0, 1, 2):
                x = torch.randn(4, 2, device=device, dtype=dtype, requires_grad=True)
                mask = torch.arange(4, device=device) < n
                expected = fn(x, mask)
                expected_grad = torch.autograd.grad(
                    expected, x, torch.ones_like(expected)
                )[0]
                actual = compiled(x, mask)
                actual_grad = torch.autograd.grad(actual, x, torch.ones_like(actual))[0]
                self._assert_bitwise_equal(actual, expected)
                self._assert_bitwise_equal(actual_grad, expected_grad)
                self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(counter.frame_count, 1)

    @skipCUDAIfNoCudnn
    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize(
        "shape",
        (
            (2, 3, 4, 4),
            (3, 17, 8, 9),
            (2, 3, 128, 129),
            (16, 32, 64, 64),
            (2, 3, 4, 4, 5),
            (2, 3, 1024, 1),
        ),
    )
    @parametrize("channels_last", (False, True))
    @parametrize("momentum", (0.1, 1.0))
    @parametrize("track_running_stats", (False, True))
    def test_generated_cudnn_batch_norm_statistics(
        self, device, dtype, shape, channels_last, momentum, track_running_stats
    ):
        from torch._inductor.inductor_prims import cudnn_batch_norm_statistics

        torch.manual_seed(277)
        x = torch.randn(shape, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(
                memory_format=torch.channels_last
                if x.ndim == 4
                else torch.channels_last_3d
            )
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        mean = (
            torch.randn(shape[1], device=device, dtype=stat_dtype)
            if track_running_stats
            else None
        )
        var = torch.rand_like(mean) + 0.5 if mean is not None else None
        original = (None, None) if mean is None else (mean.clone(), var.clone())

        def fn(x, mean, var):
            return cudnn_batch_norm_statistics(x, mean, var, momentum, 1e-5)

        expected = fn(x, mean, var)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True), x, mean, var
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual((mean, var), original)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertGreater(metrics.generated_kernel_count, 0)
        source = "\n".join(code)
        self.assertIn("triton_helpers.cuda_cudnn_bn_", source)
        self.assertNotIn("aten.cudnn_batch_norm", source)
        for first, second in itertools.combinations(actual, 2):
            self.assertFalse(torch._C._is_alias_of(first, second))

    @skipCUDAIfNoCudnn
    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize(
        "shape",
        (
            (2, 3, 4, 4),
            (3, 17, 8, 9),
            (2, 3, 33, 17),
            (2, 3, 128, 129),
            (16, 32, 64, 64),
            (2, 3, 4, 4, 5),
            (2, 3, 1024, 1),
            (8, 17, 128, 129),
            (16, 4, 64, 65),
        ),
    )
    @parametrize("channels_last", (False, True))
    def test_generated_cudnn_batch_norm_backward(
        self, device, dtype, shape, channels_last
    ):
        torch.manual_seed(7)
        x = torch.randn(shape, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(
                memory_format=torch.channels_last
                if x.ndim == 4
                else torch.channels_last_3d
            )
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 0.5
        output, save_mean, invstd, reserve = torch.ops.aten.cudnn_batch_norm.default(
            x, weight, bias, mean, var, True, 0.1, 1e-5
        )
        grad = torch.randn_like(output)

        def fn(grad, x, weight, mean, var, save_mean, invstd, reserve):
            return torch.ops.aten.batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                invstd,
                True,
                1e-5,
                [True, True, True],
                reserve,
            )

        args = (grad, x, weight, mean, var, save_mean, invstd, reserve)
        expected = fn(*args)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertIn("triton_helpers.cuda_cudnn_bn_", "\n".join(code))
        self.assertNotIn("aten.cudnn_batch_norm_backward", "\n".join(code))

    @skipCUDAIfNoCudnn
    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize(
        "shape", ((2, 3, 4, 4), (3, 17, 8, 9), (2, 3, 128, 129), (16, 32, 64, 64))
    )
    @parametrize("channels_last", (False, True))
    def test_generated_cudnn_batch_norm_pair(self, device, dtype, shape, channels_last):
        torch.manual_seed(7)
        x = torch.randn(shape, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x.requires_grad_()
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(
            shape[1], device=device, dtype=stat_dtype, requires_grad=True
        )
        bias = torch.randn_like(weight, requires_grad=True)
        mean = torch.randn_like(weight, requires_grad=False)
        var = torch.rand_like(weight, requires_grad=False) + 0.5

        def fn(x, weight, bias, mean, var):
            mean, var = mean.clone(), var.clone()
            output, save_mean, invstd, _ = (
                torch.ops.aten._batch_norm_with_update.default(
                    x, weight, bias, mean, var, 0.1, 1e-5
                )
            )
            return output, save_mean, invstd, mean, var

        expected = fn(x, weight, bias, mean, var)
        grad = torch.randn_like(expected[0])
        expected_grads = torch.autograd.grad(expected[0], (x, weight, bias), grad)
        inputs = tuple(
            value.detach().clone().requires_grad_() for value in (x, weight, bias)
        )
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(*inputs, mean, var)
            actual_grads, code = run_and_get_code(
                lambda: torch.autograd.grad(actual[0], inputs, grad)
            )
        self._assert_bitwise_equal(actual, expected)
        self._assert_bitwise_equal(actual_grads, expected_grads)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertIn("triton_helpers.cuda_cudnn_bn_", "\n".join(code))
        self.assertNotIn("aten.cudnn_batch_norm_backward", "\n".join(code))

    @skipCUDAIfNoCudnn
    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize(
        "shape", ((2, 3, 4, 4), (3, 17, 8, 9), (2, 3, 128, 129), (16, 32, 64, 64))
    )
    @parametrize("channels_last", (False, True))
    @parametrize("training", (False, True))
    def test_generated_cudnn_functional_batch_norm_pair(
        self, device, dtype, shape, channels_last, training
    ):
        torch.manual_seed(7)
        x = torch.randn(shape, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x.requires_grad_()
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(
            shape[1], device=device, dtype=stat_dtype, requires_grad=True
        )
        bias = torch.randn_like(weight, requires_grad=True)
        mean = torch.randn_like(weight, requires_grad=False)
        var = torch.rand_like(weight, requires_grad=False) + 0.5

        def fn(x, weight, bias, mean, var):
            mean, var = mean.clone(), var.clone()
            output = torch.nn.functional.batch_norm(
                x, mean, var, weight, bias, training, 0.1, 1e-5
            )
            return output, mean, var

        expected = fn(x, weight, bias, mean, var)
        grad = torch.randn_like(expected[0])
        expected_grads = torch.autograd.grad(expected[0], (x, weight, bias), grad)
        inputs = tuple(
            value.detach().clone().requires_grad_() for value in (x, weight, bias)
        )
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True), *inputs, mean, var
            )
            actual_grads, back_code = run_and_get_code(
                lambda: torch.autograd.grad(actual[0], inputs, grad)
            )
        self._assert_bitwise_equal(actual, expected)
        self._assert_bitwise_equal(actual_grads, expected_grads)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        generated = "\n".join(code + back_code)
        if training:
            self.assertIn("triton_helpers.cuda_cudnn_bn_", generated)
        self.assertNotIn("aten.cudnn_batch_norm", generated)

    @skipCUDAIfNoCudnn
    @parametrize("training", (False, True))
    @parametrize("stat", ("running_mean", "running_var"))
    def test_generated_cudnn_batch_norm_running_stats_not_differentiable(
        self, device, training, stat
    ):
        x = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        weight = torch.ones(3, device=device, requires_grad=True)
        bias = torch.zeros_like(weight, requires_grad=True)
        mean = torch.zeros_like(weight, requires_grad=stat == "running_mean")
        var = torch.ones_like(weight, requires_grad=stat == "running_var")

        def fn(x, weight, bias, mean, var):
            return torch.nn.functional.batch_norm(
                x, mean, var, weight, bias, training, 0.1, 1e-5
            )

        error = f"not differentiable with respect to argument '{stat}'"
        with self.assertRaisesRegex(RuntimeError, error):
            fn(x, weight, bias, mean, var)
        with config.patch(numerics="strict"):
            with self.assertRaisesRegex(RuntimeError, error):
                torch.compile(fn, fullgraph=True)(x, weight, bias, mean, var)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "shape", ((8, 16, 33), (4, 16, 80), (1, 3, 1), (33, 3, 16), (8, 3, 1025))
    )
    def test_generated_batch_norm_statistics(self, device, dtype, shape):
        torch.manual_seed(275)
        x = torch.randn(shape, device=device, dtype=dtype)

        def fn(x):
            return torch.ops.aten.batch_norm_update_stats.default(x, None, None, 0.0)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertFalse(torch._C._is_alias_of(*actual))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("dims", (None, (0,), (0, 2)))
    @parametrize("strided", (False, True))
    def test_generated_negative_view_sum(self, device, dtype, dims, strided):
        torch.manual_seed(276)
        x = torch.randn(8, 16, 33, device=device, dtype=dtype)
        if strided:
            x = x[::2, :, ::2]
        output_dtype = torch.float64 if dtype != torch.float64 else torch.float32

        def fn(x):
            return torch._neg_view(x).sum(dim=dims, dtype=output_dtype)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("shape", ((8, 16, 33), (4, 16, 80)))
    @parametrize("functional", (False, True))
    @parametrize("momentum", (0.0, 0.1, 1.0))
    def test_generated_batch_norm_training(
        self, device, dtype, shape, functional, momentum
    ):
        torch.manual_seed(277)
        x = torch.randn(shape, device=device, dtype=dtype)
        stat_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        mean = torch.randn(shape[1], device=device, dtype=stat_dtype)
        var = torch.rand(shape[1], device=device, dtype=stat_dtype) + 1
        weight, bias = torch.randn_like(mean), torch.randn_like(mean)
        op = (
            torch.ops.aten._native_batch_norm_legit_functional.default
            if functional
            else torch.ops.aten._native_batch_norm_legit.default
        )

        def fn(x, mean, var, weight, bias):
            result = op(x, weight, bias, mean, var, True, momentum, 1e-5)
            return result, mean, var

        expected = fn(x, mean.clone(), var.clone(), weight, bias)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                x, mean.clone(), var.clone(), weight, bias
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("width", (3, 5, 7, 9, 17, 33, 65, 129))
    def test_generated_sum_non_power_of_two(self, device, dtype, width):
        torch.manual_seed(279)
        x = torch.randn(5, width, device=device, dtype=dtype)

        def fn(x):
            return x.sum(dim=1)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "shape",
        (
            (1, 3, 1, 1),
            (4, 3, 1, 1),
            (4, 16, 5, 7),
            (4, 17, 8, 9),
            (1, 65, 3, 37),
            (65, 33, 3, 37),
            (16385, 3, 1, 33),
            (16385, 64, 1, 33),
        ),
    )
    def test_generated_channels_last_bn_statistics(self, device, dtype, shape):
        torch.manual_seed(279)
        x = torch.randn(shape, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )

        def fn(x):
            return torch.ops.aten.batch_norm_update_stats.default(x, None, None, 0.0)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)
        self.assertFalse(torch._C._is_alias_of(*actual))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "layout",
        ("channels_last", "channels_last_3d", "strided", "permuted", "c_stride_one"),
    )
    @parametrize("functional", (False, True))
    @parametrize("negative", (False, True))
    def test_generated_batch_norm_training_layouts(
        self, device, dtype, layout, functional, negative
    ):
        torch.manual_seed(280)
        x = torch.randn(4, 17, 8, 9, device=device, dtype=dtype)
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "channels_last_3d":
            x = x.unsqueeze(2).contiguous(memory_format=torch.channels_last_3d)
        elif layout == "strided":
            x = x[:, :, ::2, ::2]
        elif layout == "permuted":
            x = x.transpose(2, 3)
        else:
            x = x[:, :, :1, :1].contiguous()
        if negative:
            x = torch._neg_view(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        mean = torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        var = torch.rand_like(mean) + 1
        weight, bias = torch.randn_like(mean), torch.randn_like(mean)
        op = (
            torch.ops.aten._native_batch_norm_legit_functional.default
            if functional
            else torch.ops.aten._native_batch_norm_legit.default
        )

        def fn(x, mean, var, weight, bias):
            result = op(x, weight, bias, mean, var, True, 0.1, 1e-5)
            return result, mean, var

        expected = fn(x, mean.clone(), var.clone(), weight, bias)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                x, mean.clone(), var.clone(), weight, bias
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "shape", ((8, 16, 33), (4, 16, 80), (33, 3, 16), (8, 3, 1025), (1, 65, 3))
    )
    @parametrize("training", (False, True))
    def test_generated_batch_norm_backward_fused(self, device, dtype, shape, training):
        torch.manual_seed(281)
        x = torch.randn(shape, device=device, dtype=dtype)
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, mean, var, training, 0.0, 1e-5
        )

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                training,
                1e-5,
                (True, True, True),
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("p", (1, 3, 5))
    def test_generated_mvlgamma_scalar_backward(self, device, dtype, p):
        from torch._inductor import ir

        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)

        def fn(x):
            return torch.mvlgamma(x, p)

        expected = fn(x)
        expected_grad = torch.autograd.grad(expected, x, torch.ones_like(expected))
        fallback_ops = []
        original = ir.FallbackKernel.codegen
        for numerics in ("default", "strict"):
            numeric = set()

            def record(kernel, wrapper):
                numeric.add(str(kernel.op_overload))
                return original(kernel, wrapper)

            metrics.reset()
            with mock.patch.object(ir.FallbackKernel, "codegen", record):
                actual = torch.compile(
                    fn,
                    fullgraph=True,
                    options={"numerics": numerics, "fx_graph_cache": False},
                )(x)
                actual_grad = torch.autograd.grad(actual, x, torch.ones_like(actual))
            if numerics == "strict":
                self._assert_bitwise_equal(
                    (actual, actual_grad), (expected, expected_grad)
                )
                self.assertGreater(metrics.generated_kernel_count, 0)
            fallback_ops.append(numeric)
        self.assertLessEqual(fallback_ops[1], fallback_ops[0])

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_generated_sum_fused_reciprocal(self, device, dtype):
        torch.manual_seed(282)
        x = torch.randn(10, 5, 10, device=device, dtype=dtype).transpose(0, 2)

        def fn(x):
            return 1.0 / x.sum(dim=(0, 2))

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("has_weight", (False, True))
    @parametrize("saved_invstd", (False, True))
    def test_generated_batch_norm_backward_eval_input(
        self, device, dtype, has_weight, saved_invstd
    ):
        torch.manual_seed(283)
        x = torch.randn(4, 17, 8, 9, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        mean = torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        var = torch.rand_like(mean) + 1
        weight = torch.randn_like(mean) if has_weight else None
        save_mean = torch.empty(0, device=device, dtype=stat_dtype)
        save_invstd = torch.rand_like(mean) if saved_invstd else save_mean

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                False,
                1e-5,
                (True, False, False),
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        self._assert_bitwise_equal(actual[0], expected[0])
        self.assertEqual(actual[1:], expected[1:])
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("layout", ("strided", "permuted"))
    @parametrize("negative", (False, True))
    def test_generated_batch_norm_backward_fused_layouts(
        self, device, dtype, layout, negative
    ):
        torch.manual_seed(285)
        x = torch.randn(4, 17, 8, 9, device=device, dtype=dtype)
        x = x[:, :, ::2, ::2] if layout == "strided" else x.transpose(2, 3)
        if negative:
            x = torch._neg_view(x)
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        _, mean, invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, None, None, True, 0.0, 1e-5
        )

        def fn(grad, x, weight, mean, invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                None,
                None,
                mean,
                invstd,
                True,
                1e-5,
                (True, True, True),
            )

        expected = fn(grad, x, weight, mean, invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(grad, x, weight, mean, invstd)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "shape",
        (
            (1, 3, 1, 1),
            (4, 3, 1, 1),
            (4, 16, 5, 7),
            (4, 17, 8, 9),
            (1, 65, 3, 37),
            (65, 33, 3, 37),
            (16385, 3, 1, 33),
            (16385, 64, 1, 33),
        ),
    )
    @parametrize("training", (False, True))
    def test_generated_batch_norm_backward_channels_last(
        self, device, dtype, shape, training
    ):
        torch.manual_seed(284)
        x = torch.randn(shape, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, mean, var, training, 0.0, 1e-5
        )

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                training,
                1e-5,
                (True, True, True),
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("training", (False, True))
    @parametrize("has_weight", (False, True))
    @parametrize("mask", tuple(itertools.product((False, True), repeat=3)))
    def test_generated_channels_last_bn_backward_masks(
        self, device, dtype, training, has_weight, mask
    ):
        torch.manual_seed(367)
        x = torch.randn(2, 8, 16, 16, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        mean = torch.randn(8, device=device, dtype=stat_dtype)
        var = torch.rand_like(mean) + 1
        weight = torch.randn_like(mean) if has_weight else None
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, None, mean, var, training, 0.0, 0.137
        )

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                training,
                0.137,
                mask,
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        for result, reference in zip(actual, expected):
            if reference is None:
                self.assertIsNone(result)
            else:
                self._assert_bitwise_equal(result, reference)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "mask",
        (
            (False, True, True),
            (False, True, False),
            (False, False, True),
            (True, False, False),
        ),
    )
    @parametrize("training", (False, True))
    def test_generated_batch_norm_backward_masks(self, device, dtype, mask, training):
        torch.manual_seed(286)
        x = torch.randn(33, 3, 16, device=device, dtype=dtype)
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, mean, var, training, 0.0, 1e-5
        )

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad, x, weight, mean, var, save_mean, save_invstd, training, 1e-5, mask
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        for result, reference in zip(actual, expected):
            if reference is None:
                self.assertIsNone(result)
            else:
                self._assert_bitwise_equal(result, reference)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "layout", ("grad_contiguous", "weight_strided", "channels_last_3d", "negative")
    )
    @parametrize("training", (False, True))
    def test_generated_batch_norm_backward_layout_combinations(
        self, device, dtype, layout, training
    ):
        torch.manual_seed(287)
        x = torch.randn(4, 17, 8, 9, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        if layout == "channels_last_3d":
            x = x.unsqueeze(2).contiguous(memory_format=torch.channels_last_3d)
        if layout == "negative":
            x = torch._neg_view(x)
        grad = torch.randn_like(x)
        if layout == "grad_contiguous":
            grad = grad.contiguous()
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = (
            torch.randn(x.shape[1] * 2, device=device, dtype=stat_dtype)[::2]
            if layout == "weight_strided"
            else torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        )
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, mean, var, training, 0.0, 1e-5
        )

        def fn(grad, x, weight, mean, var, save_mean, save_invstd):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                training,
                1e-5,
                (True, True, True),
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float32)
    @parametrize("channels_last", (False, True))
    def test_generated_batch_norm_wrapper_native_pair(
        self, device, dtype, channels_last
    ):
        x = (torch.arange(96, device=device) % 2 * 2).to(dtype).reshape(2, 3, 4, 4)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        weight = torch.ones(3, device=device)
        bias = torch.zeros_like(weight)
        mean = torch.zeros_like(weight)
        var = torch.ones_like(weight)

        def fn(x, weight, bias, mean, var):
            out, save_mean, invstd, reserve, new_mean, new_var = (
                torch.ops.aten._batch_norm_with_update_functional.default(
                    x, weight, bias, mean, var, 1.0, 0.0
                )
            )
            grads = torch.ops.aten.batch_norm_backward.default(
                x,
                x,
                weight,
                mean,
                var,
                save_mean,
                invstd,
                True,
                0.0,
                (True, True, True),
                reserve,
            )
            return out, grads, new_mean, new_var

        expected = fn(x, weight, bias, mean, var)

        def reserve_tensor(input, weight, bias, *args, **kwargs):
            self.assertEqual(weight.ndim, 1)
            self.assertEqual(bias.ndim, 1)
            return torch.empty(64, dtype=torch.uint8, device=input.device)

        metrics.reset()
        with (
            mock.patch(
                "torch._C._select_batch_norm_backend",
                return_value=torch._C._BatchNormBackend.Cudnn,
            ),
            mock.patch(
                "torch._decomp.decompositions._get_batch_norm_reserve_tensor",
                reserve_tensor,
            ),
            config.patch(numerics="strict", fx_graph_cache=False),
        ):
            actual = torch.compile(fn, fullgraph=True)(x, weight, bias, mean, var)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @skipCUDAIfNoCudnn
    @dtypes(torch.float16, torch.float32)
    @parametrize("channels_last", (False, True))
    def test_cudnn_batch_norm_forward_backward(self, device, dtype, channels_last):
        eps = 2**-16
        x = (torch.arange(96, device=device) % 2 * 2 - 1).to(dtype).reshape(2, 3, 4, 4)
        if dtype == torch.float32:
            x *= 1 - 2**-17
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        model = torch.nn.BatchNorm2d(3, eps=eps, momentum=1.0, device=device)
        x.requires_grad_(True)
        self.assertEqual(
            torch._C._select_batch_norm_backend(
                x,
                model.weight,
                model.bias,
                model.running_mean,
                model.running_var,
                True,
                eps,
            ),
            torch._C._BatchNormBackend.Cudnn,
        )

        def fn(x):
            return model(x)

        expected = fn(x)
        grad = torch.ones_like(x)
        expected_grads = torch.autograd.grad(
            expected, (x, model.weight, model.bias), grad
        )
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
            actual_grads = torch.autograd.grad(
                actual, (x, model.weight, model.bias), grad
            )
        self._assert_bitwise_equal(actual, expected)
        self._assert_bitwise_equal(actual_grads, expected_grads)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("channels_last", (False, True))
    @parametrize("training", (False, True))
    @torch.backends.cudnn.flags(enabled=False)
    def test_generated_batch_norm_backward_wrapper(
        self, device, dtype, channels_last, training
    ):
        torch.manual_seed(288)
        x = torch.randn(4, 17, 8, 9, device=device, dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        grad = torch.randn_like(x)
        stat_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        weight = torch.randn(x.shape[1], device=device, dtype=stat_dtype)
        bias = torch.randn_like(weight)
        mean = torch.randn_like(weight)
        var = torch.rand_like(weight) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, mean, var, training, 0.0, 1e-5
        )
        reserve = torch.empty(0, device=device, dtype=torch.uint8)

        def fn(grad, x, weight, mean, var, save_mean, save_invstd, reserve):
            return torch.ops.aten.batch_norm_backward.default(
                grad,
                x,
                weight,
                mean,
                var,
                save_mean,
                save_invstd,
                training,
                1e-5,
                (True, True, True),
                reserve,
            )

        expected = fn(grad, x, weight, mean, var, save_mean, save_invstd, reserve)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(
                grad, x, weight, mean, var, save_mean, save_invstd, reserve
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("size", (0, 1, 33, 127, 128, 1025, 131073))
    @parametrize("pattern", ("none", "all", "third", "random"))
    def test_generated_masked_sum(self, device, dtype, size, pattern):
        from torch._inductor import inductor_prims

        torch.manual_seed(290)
        x = torch.randn(size, device=device, dtype=dtype)
        index = torch.arange(size, device=device)
        if pattern == "none":
            mask = index < 0
        elif pattern == "all":
            mask = index >= 0
        elif pattern == "third":
            mask = index % 3 == 0
        else:
            mask = torch.rand(size, device=device) < 0.7

        def fn(x, mask):
            return inductor_prims.masked_sum(x, mask)

        expected = fn(x, mask)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x, mask)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "layout",
        ("strided", "transposed", "negative", "broadcast_input", "broadcast_mask"),
    )
    def test_generated_masked_sum_layouts(self, device, dtype, layout):
        from torch._inductor import inductor_prims

        torch.manual_seed(292)
        x = torch.randn(5, 7, 33, device=device, dtype=dtype)
        mask = torch.rand(x.shape, device=device) < 0.6
        if layout == "strided":
            x, mask = x[..., ::2], mask[..., ::2]
        elif layout == "transposed":
            x, mask = x.transpose(0, 2), mask.transpose(0, 2)
        elif layout == "negative":
            x = torch._neg_view(x)
        elif layout == "broadcast_input":
            x = x[0, 0, 0]
        else:
            mask = mask[:1, :, :1]

        def fn(x, mask):
            return inductor_prims.masked_sum(x, mask)

        expected = fn(x, mask)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x, mask)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_generated_masked_sum_runtime_count(self, device, dtype):
        from torch._inductor import inductor_prims

        torch.manual_seed(293)
        x = torch.randn(1025, device=device, dtype=dtype)
        index = torch.arange(x.numel(), device=device)
        counter = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        def fn(x, mask):
            return inductor_prims.masked_sum(x, mask)

        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            for count in (0, 1, 127, 128, 129, 1024, 1025):
                mask = index < count
                self._assert_bitwise_equal(compiled(x, mask), fn(x, mask))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize(
        "shape",
        (
            (5, 10),
            (16, 33),
            (65, 1025),
            (1, 33),
            (33, 1),
            (5, 0),
            (16, 128),
            (32, 4096),
            (512, 128),
            (512, 4096),
            (1, 4096),
        ),
    )
    @parametrize("layout", ("contiguous", "transposed", "strided", "expanded"))
    @parametrize("vector_stride", (1, 2))
    def test_generated_mv(self, device, dtype, shape, layout, vector_stride):
        from torch._inductor import ir

        torch.manual_seed(289)
        x = torch.randn(shape, device=device, dtype=dtype)
        if layout == "transposed":
            x = x.t().contiguous().t()
        elif layout == "strided":
            x = torch.randn(shape[0], shape[1] * 2, device=device, dtype=dtype)[:, ::2]
        elif layout == "expanded" and x.numel():
            x = x[:1, :1].expand(shape)
        vector = torch.randn(shape[1] * vector_stride, device=device, dtype=dtype)[
            ::vector_stride
        ]

        def fn(x, vector):
            return torch.mv(x, vector)

        expected = fn(x, vector)
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        metrics.reset()
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(fn, fullgraph=True)(x, vector)
        if dtype.is_complex:
            actual, expected = torch.view_as_real(actual), torch.view_as_real(expected)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(numeric, [])
        if not dtype.is_complex:
            self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float64)
    def test_generated_mv_planning_workspace(self, device, dtype):
        for _ in range(32):
            torch._C._cuda_getGemvLaunchPlan(dtype, 256, 16, 256, 1, True, 0, 0)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_reserved(device)
        for _ in range(128):
            torch._C._cuda_getGemvLaunchPlan(dtype, 256, 16, 256, 1, True, 0, 0)
        torch.cuda.synchronize(device)
        self.assertLessEqual(
            torch.cuda.memory_reserved(device) - before, 64 * 1024 * 1024
        )

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("shape", ((16, 256), (512, 4096)))
    @parametrize("column", (False, True))
    def test_generated_mv_input_alignment(self, device, dtype, shape, column):
        torch.manual_seed(17)
        element_size = torch.empty((), dtype=dtype).element_size()
        counter = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        with config.patch(numerics="strict", fx_graph_cache=False):
            compiled = torch.compile(torch.mv, backend=counter, fullgraph=True)
            for matrix_offset in range(16 // element_size):
                for vector_offset in range(16 // element_size):
                    matrix = torch.randn(
                        shape[0] * shape[1] + matrix_offset, device=device, dtype=dtype
                    )[matrix_offset:]
                    matrix = (
                        matrix.view(shape[::-1]).t() if column else matrix.view(shape)
                    )
                    vector = torch.randn(
                        shape[1] + vector_offset, device=device, dtype=dtype
                    )[vector_offset:]
                    actual, expected = (
                        compiled(matrix, vector),
                        torch.mv(matrix, vector),
                    )
                    if dtype.is_complex:
                        actual, expected = (
                            torch.view_as_real(actual),
                            torch.view_as_real(expected),
                        )
                    self._assert_bitwise_equal(actual, expected)
        self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize("shape", ((), (0,), (16, 33)))
    @parametrize("layout", ("contiguous", "transposed", "strided"))
    def test_generated_complex_constructor(self, device, dtype, shape, layout):
        from torch._inductor import ir

        torch.manual_seed(359)
        real, imag = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
        if real.ndim == 2:
            real[0, :4] = torch.tensor(
                [0.0, -0.0, float("nan"), float("inf")], device=device, dtype=dtype
            )
            imag[0, :4] = torch.tensor(
                [-0.0, 0.0, float("inf"), float("nan")], device=device, dtype=dtype
            )
            if layout == "transposed":
                real, imag = real.t(), imag.t()
            elif layout == "strided":
                real, imag = real[:, ::2], imag[:, ::2]
        numerical = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numerical.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        expected = torch.complex(real, imag)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(torch.complex, fullgraph=True)(real, imag)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(expected)
        )
        self.assertEqual(numerical, [])

    @dtypes(torch.float64)
    @parametrize("target", (torch.float16, torch.bfloat16))
    def test_generated_double_narrow_rounding(self, device, dtype, target):
        values = (
            torch.arange(0, 65536, device=device, dtype=torch.int32)
            .to(torch.uint16)
            .view(target)
        )
        following = torch.nextafter(values, torch.full_like(values, float("inf")))
        finite = torch.isfinite(values) & torch.isfinite(following)
        midpoint = (values[finite].to(dtype) + following[finite].to(dtype)) / 2
        inputs = torch.cat(
            (
                torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf"))),
                midpoint,
                torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))),
            )
        )

        def fn(x):
            return x.to(target)

        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(inputs)
        self._assert_bitwise_equal(actual, fn(inputs))

    @dtypes(torch.complex64, torch.complex128)
    @parametrize(
        "scales",
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.5, -0.625),
            (1.125 + 0.625j, -0.25 + 0.875j),
        ),
    )
    @parametrize("layout", ("contiguous", "strided", "negative", "conjugate"))
    def test_generated_complex_addr(self, device, dtype, scales, layout):
        from torch._inductor import ir

        torch.manual_seed(353)
        matrix = torch.randn(16, 33, device=device, dtype=dtype)
        first = torch.randn(16, device=device, dtype=dtype)
        second = torch.randn(33, device=device, dtype=dtype)
        torch.view_as_real(matrix).view(-1, 2)[:4] = torch.tensor(
            [
                [0.0, -0.0],
                [float("inf"), 1.0],
                [1.0, float("inf")],
                [float("nan"), 2.0],
            ],
            device=device,
            dtype=torch.view_as_real(matrix).dtype,
        )
        if layout == "strided":
            matrix = matrix.repeat_interleave(2, dim=1)[:, ::2]
            first, second = (
                first.repeat_interleave(2)[::2],
                second.repeat_interleave(2)[::2],
            )
        elif layout == "negative":
            matrix, first, second = [
                torch._neg_view(x) for x in (matrix, first, second)
            ]
        elif layout == "conjugate":
            matrix, first, second = [x.conj() for x in (matrix, first, second)]
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        def fn(matrix, first, second):
            return torch.addr(matrix, first, second, alpha=scales[0], beta=scales[1])

        expected = fn(matrix, first, second)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(fn, fullgraph=True)(matrix, first, second)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(expected)
        )
        self.assertEqual(numeric, [])

    @dtypes(torch.complex64, torch.complex128)
    @parametrize(
        "target",
        (
            torch.bool,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ),
    )
    @parametrize("shape", ((), (0,), (512,)))
    @parametrize("layout", ("contiguous", "strided", "negative", "conjugate"))
    def test_generated_complex_cast(self, device, dtype, target, shape, layout):
        from torch._inductor import ir

        torch.manual_seed(349)
        x = torch.randn(shape, device=device, dtype=dtype)
        if x.numel() > 4:
            parts = torch.view_as_real(x)
            parts[:4] = torch.tensor(
                [[0.0, -0.0], [0.0, 1.0], [1.0, float("inf")], [0.0, float("nan")]],
                device=device,
                dtype=parts.dtype,
            )
        if layout == "strided" and x.ndim:
            x = x.repeat_interleave(2)[::2]
        elif layout == "negative":
            x = torch._neg_view(x)
        elif layout == "conjugate":
            x = x.conj()
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        def fn(x):
            return x.to(target, copy=True)

        expected = fn(x)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
        if target.is_complex:
            actual, expected = torch.view_as_real(actual), torch.view_as_real(expected)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(numeric, [])

    @dtypes(torch.complex64, torch.complex128)
    @parametrize("scale", (0.0, 1.0, 3.14, -3.14))
    @parametrize("layout", ("contiguous", "strided", "negative", "conjugate"))
    def test_generated_complex_addcdiv(self, device, dtype, scale, layout):
        from torch._inductor import ir

        torch.manual_seed(347)
        inputs = [torch.randn(512, device=device, dtype=dtype) for _ in range(3)]
        for x in inputs:
            parts = torch.view_as_real(x)
            parts[:4] = torch.tensor(
                [
                    [0.0, -0.0],
                    [float("inf"), 1.0],
                    [1.0, float("inf")],
                    [float("nan"), 2.0],
                ],
                device=device,
                dtype=parts.dtype,
            )
        if layout == "strided":
            inputs = [x.repeat_interleave(2)[::2] for x in inputs]
        elif layout == "negative":
            inputs = [torch._neg_view(x) for x in inputs]
        elif layout == "conjugate":
            inputs = [x.conj() for x in inputs]
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        def fn(a, b, c):
            return torch.addcdiv(a, b, c, value=scale)

        expected = fn(*inputs)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(fn, fullgraph=True)(*inputs)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(expected)
        )
        self.assertEqual(numeric, [])

    @dtypes(torch.complex64, torch.complex128)
    @parametrize("scalar", (1.0, -0.0, 1.125 + 0.625j, complex(float("inf"), 1.0)))
    @parametrize("layout", ("contiguous", "strided", "negative", "conjugate"))
    def test_generated_complex_scalar_mul(self, device, dtype, scalar, layout):
        from torch._inductor import ir

        torch.manual_seed(337)
        x = torch.randn(512, device=device, dtype=dtype)
        parts = torch.view_as_real(x)
        parts[:4] = torch.tensor(
            [
                [0.0, -0.0],
                [float("inf"), 1.0],
                [1.0, float("inf")],
                [float("nan"), 2.0],
            ],
            device=device,
            dtype=parts.dtype,
        )
        if layout == "strided":
            x = x.repeat_interleave(2)[::2]
        elif layout == "negative":
            x = torch._neg_view(x)
        elif layout == "conjugate":
            x = x.conj()
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        def fn(x):
            return x * scalar

        expected = fn(x)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(expected)
        )
        self.assertFalse(numeric, str(numeric))

    @dtypes(torch.complex64, torch.complex128)
    @parametrize("shape", ((16, 33), (65, 1025)))
    @parametrize("column", (False, True))
    @parametrize("target", ("matrix", "vector"))
    @parametrize("lazy", ("negative", "conjugate"))
    def test_generated_complex_mv_lazy(
        self, device, dtype, shape, column, target, lazy
    ):
        from torch._inductor import ir

        torch.manual_seed(341)
        x = torch.randn(shape, device=device, dtype=dtype)
        if column:
            x = x.t().contiguous().t()
        vector = torch.randn(shape[1], device=device, dtype=dtype)
        value = x if target == "matrix" else vector
        value = torch._neg_view(value) if lazy == "negative" else value.conj()
        if target == "matrix":
            x = value
        else:
            vector = value
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        expected = torch.mv(x, vector)
        with (
            config.patch(numerics="strict", fx_graph_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", record),
        ):
            actual = torch.compile(torch.mv, fullgraph=True)(x, vector)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(expected)
        )
        self.assertFalse(numeric, str(numeric))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("output_dtype", (torch.int64, torch.bool))
    @parametrize("empty", (False, True))
    def test_generated_sum_integer_dtype(self, device, dtype, output_dtype, empty):
        x = torch.randn(0 if empty else 8, 16, 33, device=device, dtype=dtype) * 17

        def fn(x):
            return x.sum(dim=(0, 2), dtype=output_dtype)

        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, fn(x))
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("strided", (False, True))
    def test_generated_dtype_view_copy(self, device, dtype, numerics, strided):
        from torch._inductor import ir

        bit_dtype = torch.int64 if dtype == torch.float64 else torch.int32
        x = torch.randn(8, 34, device=device, dtype=dtype)
        if strided:
            x = x[:, 1::2]
        values = torch.randn_like(x).view(bit_dtype)

        def fn(x, values):
            result = x.clone()
            result.view(bit_dtype).copy_(values)
            return result + x, result

        expected = fn(x, values)
        with (
            config.patch(numerics=numerics, fx_graph_cache=False),
            mock.patch.object(
                ir.InplaceCopyFallback,
                "codegen",
                side_effect=AssertionError("real copies must use generated kernels"),
            ),
        ):
            actual = torch.compile(fn, fullgraph=True)(x, values)
        self._assert_bitwise_equal(actual, expected)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_negative_alias_cpp_wrapper(self, device, dtype):
        def fn(x, y):
            result = x.clone()
            negative = torch._neg_view(result)
            negative.copy_(y)
            return negative + x, negative, result

        x, y = [torch.randn(8, 17, device=device, dtype=dtype) for _ in range(2)]
        expected = fn(x, y)
        with config.patch(numerics="strict", cpp_wrapper=True, fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x, y)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual([v.is_neg() for v in actual], [v.is_neg() for v in expected])
        self.assertTrue(torch._C._is_alias_of(actual[1], actual[2]))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "layout",
        ("strided", "permuted", "unaligned1", "unaligned2", "global", "vectorized"),
    )
    def test_generated_general_bn_statistics(self, device, dtype, layout):
        torch.manual_seed(279)
        if layout == "global":
            x = torch.randn(16385, 3, 33, device=device, dtype=dtype)[:, :, ::2]
        elif layout == "strided":
            x = torch.randn(4, 16, 80, device=device, dtype=dtype)[:, :, ::2]
        elif layout == "vectorized":
            x = torch.randn(1, 16, 1033, device=device, dtype=dtype)[:, :, 1:1026]
        else:
            offset = {"permuted": 0, "unaligned1": 1, "unaligned2": 2}[layout]
            x = (
                torch.randn(4 * 33 * 16 + offset, device=device, dtype=dtype)[offset:]
                .reshape(4, 33, 16)
                .transpose(1, 2)
            )

        def fn(x):
            return torch.ops.aten.batch_norm_update_stats.default(x, None, None, 0.0)

        expected = fn(x)
        metrics.reset()
        with config.patch(numerics="strict", fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("layout", ("contiguous", "strided", "negative", "conjugate"))
    def test_generated_complex_gradient_accumulation(
        self, device, dtype, numerics, layout
    ):
        from torch._inductor import ir

        torch.manual_seed(297)
        x = torch.randn(8, device=device, dtype=dtype, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        parts = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                float("nan"),
                float("inf"),
                torch.finfo(real_dtype).tiny,
                2.0,
            ],
            device=device,
            dtype=real_dtype,
        )
        grad = torch.view_as_complex(torch.stack((parts, parts.flip(0)), -1))
        if layout == "strided":
            grad = grad.repeat_interleave(2)[::2]
        elif layout == "negative":
            grad = torch._neg_view(grad)
        elif layout == "conjugate":
            grad = grad.conj()
        other = grad.detach().clone()
        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        def fn(x):
            return x.clone(), x.clone()

        with config.patch(numerics=numerics, fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
            expected = fn(reference)
            with mock.patch.object(ir.FallbackKernel, "codegen", record):
                for supplied in (
                    (grad, other),
                    (grad, None),
                    (None, other),
                    (grad, other),
                ):
                    indices = [
                        i for i, value in enumerate(supplied) if value is not None
                    ]
                    grads = [supplied[i] for i in indices]
                    actual_grad = torch.autograd.grad(
                        [actual[i] for i in indices], x, grads, retain_graph=True
                    )
                    expected_grad = torch.autograd.grad(
                        [expected[i] for i in indices],
                        reference,
                        grads,
                        retain_graph=True,
                    )
                    self._assert_bitwise_equal(
                        torch.view_as_real(actual_grad[0].resolve_conj()),
                        torch.view_as_real(expected_grad[0].resolve_conj()),
                    )
        self.assertNotIn("aten.add.Tensor", numeric)

    def _assert_optional_output_gradients(
        self,
        actual,
        expected,
        base,
        expected_base,
        upstream,
        *,
        selected=0,
        backward_compiler=None,
    ):
        self._assert_bitwise_equal(actual, expected)
        original_upstream = upstream.clone()
        # Reuse the forward graph and the absent-gradient backward variant.
        for supplied_zero in (False, True, False):
            actual_outputs = actual if supplied_zero else actual[selected]
            expected_outputs = expected if supplied_zero else expected[selected]
            grad_outputs = (
                tuple(
                    upstream if i == selected else torch.zeros_like(output)
                    for i, output in enumerate(actual)
                )
                if supplied_zero
                else upstream
            )
            with (
                compiled_autograd._enable(backward_compiler)
                if backward_compiler is not None
                else nullcontext()
            ):
                actual_grads = torch.autograd.grad(
                    actual_outputs,
                    base,
                    grad_outputs,
                    allow_unused=True,
                    retain_graph=True,
                )
            expected_grads = torch.autograd.grad(
                expected_outputs,
                expected_base,
                grad_outputs,
                allow_unused=True,
                retain_graph=True,
            )
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                if expected_grad is None:
                    self.assertIsNone(actual_grad)
                else:
                    self._assert_bitwise_equal(actual_grad, expected_grad)
            self._assert_bitwise_equal(base, expected_base)
            self._assert_bitwise_equal(upstream, original_upstream)

    @dtypes(torch.float32, torch.complex64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("op", ("aten", "prims"))
    def test_where_predicate_dtype(self, device, dtype, numerics, op):
        call = torch.where if op == "aten" else torch.ops.prims.where.default
        a = torch.tensor([1, -0.0, 3], dtype=dtype, device=device)
        b = torch.tensor([10, 0.0, 30], dtype=dtype, device=device)
        mask = torch.tensor([False, True, True], device=device)
        predicate_dtypes = (torch.bool, torch.uint8, torch.float32, torch.int64)
        for predicate_dtype in predicate_dtypes:
            predicate = (
                torch.tensor([0, 2, 255], dtype=predicate_dtype, device=device)
                if predicate_dtype == torch.uint8
                else mask.to(predicate_dtype)
            )
            compiled = torch.compile(
                call, fullgraph=True, options={"numerics": numerics}
            )
            if predicate_dtype in (torch.bool, torch.uint8):
                conditions = (predicate, predicate[1]) if op == "aten" else (predicate,)
                for condition in conditions:
                    self._assert_bitwise_equal(
                        compiled(condition, a, b), call(condition, a, b)
                    )
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "(condition|predicate).*bool"
                ):
                    call(predicate, a, b)
                with self.assertRaisesRegex(
                    RuntimeError, "(condition|predicate).*bool"
                ):
                    compiled(predicate, a, b)

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("scalar_kind", ("real", "complex"))
    @parametrize("scalar_dtype", (torch.float32, torch.float64))
    @parametrize("scalar_first", (False, True))
    def test_generated_complex_where_cpu_scalar(
        self, device, dtype, scalar_kind, scalar_dtype, scalar_first
    ):
        from torch._inductor import ir

        x = torch.tensor([1 + 2j, -1 - 2j, 0j], device=device, dtype=dtype)
        if scalar_kind == "complex":
            scalar_dtype = (
                torch.complex64 if scalar_dtype == torch.float32 else torch.complex128
            )
        scalar = torch.tensor(
            1.1 if scalar_kind == "real" else 1.1 + 2.2j, dtype=scalar_dtype
        )
        mask = torch.tensor([True, False, True], device=device)

        def fn(mask, x, scalar):
            return (
                torch.where(mask, scalar, x)
                if scalar_first
                else torch.where(mask, x, scalar)
            )

        numeric = []
        original = ir.FallbackKernel.codegen

        def record(kernel, wrapper):
            if not isinstance(kernel, ir.ComplexView):
                numeric.append(str(kernel.op_overload))
            return original(kernel, wrapper)

        expected = fn(mask, x, scalar)
        with mock.patch.object(ir.FallbackKernel, "codegen", record):
            actual = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})(
                mask, x, scalar
            )
        self._assert_bitwise_equal(actual, expected)
        self.assertEqual(numeric, [])

    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_complex_where_unsupported_dtype(self, device, numerics):
        components = torch.tensor(
            [[0.0, -0.0], [-0.0, 0.0], [1.0, -1.0]],
            dtype=torch.bfloat16,
            device=device,
        )
        a = torch.empty(3, dtype=torch.bcomplex32, device=device)
        a.view(torch.bfloat16).copy_(components.flatten())
        b = a.clone()
        b.view(torch.bfloat16).neg_()
        mask = torch.tensor([True, False, True], device=device)
        actual, code = run_and_get_code(
            torch.compile(torch.where, fullgraph=True, options={"numerics": numerics}),
            mask,
            a,
            b,
        )
        self._assert_bitwise_equal(
            actual.view(torch.int16), torch.where(mask, a, b).view(torch.int16)
        )
        self.assertIn("torch.ops.aten.where.self", "\n".join(code))

    @dtypes(torch.float32)
    @parametrize(
        "case",
        (
            "masked_fill",
            "value_only",
            "where",
            "func_grad",
            "cudagraph",
            "detached",
            "subclass_input",
            "subclass_mask",
            "custom_autograd",
            "subclass_extracted_mask",
        ),
    )
    def test_masked_fill_backward_provenance(self, device, dtype, case):
        from torch.testing._internal.two_tensor import TwoTensor

        torch.manual_seed(1024)
        mask = torch.arange(528, device=device) % 3 == 0
        wrapped_mask = TwoTensor(mask, mask.clone())
        grad = torch.randn(528, dtype=dtype, device=device)
        x = torch.zeros_like(grad, requires_grad=case != "value_only")
        value = torch.ones((), dtype=dtype, device=device, requires_grad=True)
        subclass = case in ("subclass_input", "subclass_mask")
        if subclass:
            if case == "subclass_input":
                x = TwoTensor(x.detach(), x.detach().clone()).requires_grad_()
            else:
                mask = wrapped_mask
            grad = TwoTensor(grad, grad.clone())

        class CustomMaskedFill(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, fill):
                ctx.save_for_backward(mask)
                return inp.masked_fill(mask, fill)

            @staticmethod
            def backward(ctx, upstream):
                (saved_mask,) = ctx.saved_tensors
                zero = torch.full((), 0.0, device=device, dtype=dtype)
                return (
                    torch.where(saved_mask, zero, upstream),
                    torch.where(saved_mask, upstream, zero).sum(),
                )

        def fn(inp, fill):
            if case == "subclass_extracted_mask":
                return inp.masked_fill(wrapped_mask.clone().a, fill)
            if case == "custom_autograd":
                return CustomMaskedFill.apply(inp, fill)
            if case == "detached":
                return (
                    torch.where(mask, fill, inp),
                    inp.detach().masked_fill(mask, fill.detach()),
                )
            if case == "where":
                return torch.where(mask, fill, inp)
            return inp.masked_fill(mask, fill)

        if case == "func_grad":
            call = torch.func.grad(
                lambda inp, fill: (fn(inp, fill) * grad).sum(), argnums=(0, 1)
            )
            expected = call(x, value)
        else:
            call = fn
            inputs = (value,) if case == "value_only" else (x, value)
            eager = fn(x, value)
            expected = torch.autograd.grad(
                eager[0] if case == "detached" else eager, inputs, grad
            )
        options = {"numerics": "strict"}
        if case == "cudagraph":
            options["triton.cudagraphs"] = True
        compiled = torch.compile(call, fullgraph=True, options=options)
        result = compiled(x, value)
        actual = (
            result
            if case == "func_grad"
            else torch.autograd.grad(
                result[0] if case == "detached" else result, inputs, grad
            )
        )
        if subclass:
            from torch.utils._pytree import tree_map_only

            actual, expected = tree_map_only(
                TwoTensor, lambda t: (t.a, t.b), (actual, expected)
            )
        self._assert_bitwise_equal(actual, expected)

    @dtypes(
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_mm_single_term_edges(self, device, dtype, numerics):
        smallest = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
        left = torch.tensor(
            [-0.0, 0.0, -smallest, smallest, -1.0, 1.0],
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        right = torch.tensor(
            [0.0, -0.0, 0.125, -0.125, 1.0, -1.0], device=device, dtype=dtype
        ).reshape(1, -1)
        if dtype.is_complex:
            components = torch.tensor(
                [
                    [0.0, 0.0],
                    [-0.0, 0.0],
                    [0.0, -0.0],
                    [-0.0, -0.0],
                    [1.0, -1.0],
                    [float("inf"), 0.0],
                    [0.0, float("inf")],
                    [float("nan"), 1.0],
                    [1.0, float("nan")],
                ],
                device=device,
                dtype=left.real.dtype,
            )
            values = torch.view_as_complex(components)
            left, right = values.reshape(-1, 1), values.reshape(1, -1)
        actual = torch.compile(
            torch.mm, fullgraph=True, options={"numerics": numerics}
        )(left, right)
        self._assert_bitwise_equal(actual, torch.mm(left, right))

    @dtypes(torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("op", ("sinc", "silu"))
    def test_complex_transcendental_edges(self, device, dtype, numerics, op):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _min_max_specials(real_dtype, device)
        x = torch.complex(
            values[:, None].expand(-1, values.numel()),
            values[None, :].expand(values.numel(), -1),
        )
        if op == "silu":
            # Neighbors of Thrust's two exponential scaling boundaries, with
            # zero, finite, and nonfinite imaginary parts through the same graph.
            bounds = (
                (0x42B17218, 0x43400074)
                if real_dtype == torch.float32
                else (0x40862E4200000000, 0x4096B8E4FFFFFFFF)
            )
            real = torch.tensor(
                [v + step for v in bounds for step in (-1, 0, 1)],
                dtype=_BIT_VIEW[real_dtype],
                device=device,
            ).view(real_dtype)
            edges = torch.complex(
                -real[:, None].expand(-1, values.numel()),
                values[None, :].expand(real.numel(), -1),
            )
            x = torch.cat((x, edges))
        storage = torch.empty(
            (x.shape[0], 2 * x.shape[1] + 1), dtype=dtype, device=device
        )
        strided = storage[:, 1::2]
        strided.copy_(x)
        inputs = (x, x.t(), strided, strided.t(), strided.conj(), strided._neg_view())

        call = torch.sinc if op == "sinc" else torch.nn.functional.silu

        def panels(*tensors):
            return tuple(call(t) for t in tensors)

        actual, code = run_and_get_code(
            torch.compile(panels, fullgraph=True, options={"numerics": numerics}),
            *inputs,
        )
        self._assert_bitwise_equal(
            tuple(torch.view_as_real(t) for t in actual),
            tuple(torch.view_as_real(t) for t in panels(*inputs)),
        )
        self.assertIn("@triton.jit", "\n".join(code))

    @ops(
        [op for op in op_db if op.name in ("sigmoid", "tanh")],
        allowed_dtypes=(torch.complex64, torch.complex128),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_complex_activation_backward(self, device, dtype, op, numerics):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        int_dtype = _BIT_VIEW[real_dtype]
        width = real_dtype.itemsize * 8
        infinity = 0x7F800000 if width == 32 else 0x7FF0000000000000
        quiet = infinity | (0x00400000 if width == 32 else 0x0008000000000000)
        raw = (
            quiet | 0x17,
            quiet | 0x23,
            infinity | 0x17,
            quiet | (1 << (width - 1)) | 0x23,
        )
        raw = [v if v < 1 << (width - 1) else v - (1 << width) for v in raw]
        values = torch.cat(
            (
                _min_max_specials(real_dtype, device),
                torch.tensor(raw, dtype=int_dtype, device=device).view(real_dtype),
            )
        )
        count = 17 * 61
        g = torch.complex(values, values.roll(1))
        y = torch.complex(values.roll(2), values.roll(3))
        g, y = (t.repeat((count + t.numel() - 1) // t.numel())[:count] for t in (g, y))

        def storage(t, step):
            result = torch.empty(step * count + 1, dtype=dtype, device=device)[1::step]
            result.copy_(t)
            return result

        offset_g, offset_y = storage(g, 1), storage(y, 1)
        transposed_g, transposed_y = (
            t.reshape(17, 61).t() for t in (offset_g, offset_y)
        )
        backward = getattr(torch.ops.aten, f"{op.name}_backward").default
        # Prefixes, full blocks, scalar tails, native output ordering, and fresh
        # storage after MathBits resolution all use the same small edge panel.
        pairs = (
            (g, y),
            (offset_g, offset_y),
            (transposed_g, transposed_y),
            (storage(g, 2).reshape(17, 61).t(), storage(y, 2).reshape(17, 61).t()),
            (transposed_g, y.reshape(61, 17)),
            (transposed_g[:, :1], transposed_y),
            (transposed_g.conj(), transposed_y._neg_view()),
            (offset_g.clone(), offset_y),
        )

        def backward_panels(*tensors):
            return tuple(
                backward(grad, output)
                for grad, output in zip(tensors[::2], tensors[1::2])
            )

        args = tuple(tensor for pair in pairs for tensor in pair)
        actual, code = run_and_get_code(
            torch.compile(
                backward_panels, fullgraph=True, options={"numerics": numerics}
            ),
            *args,
        )
        expected = backward_panels(*args)
        self._assert_bitwise_equal(
            tuple(torch.view_as_real(t.resolve_conj().resolve_neg()) for t in actual),
            tuple(torch.view_as_real(t.resolve_conj().resolve_neg()) for t in expected),
        )
        self.assertIn("@triton.jit", "\n".join(code))

        def selected_backward(grad, output, selector):
            indices = selector.nonzero().flatten()
            return backward(grad[indices], output[indices])

        selector = torch.tensor(
            [True, False, True, True, False, False, True, False], device=device
        )
        grad = torch.ones(8, dtype=dtype, device=device)
        output = torch.full_like(grad, 0.25 + 0.5j)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled = torch.compile(
                selected_backward, fullgraph=True, options={"numerics": numerics}
            )
            for mask in (selector, ~selector, torch.zeros_like(selector)):
                self._assert_bitwise_equal(
                    torch.view_as_real(compiled(grad, output, mask)),
                    torch.view_as_real(selected_backward(grad, output, mask)),
                )

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_complex_scalar_and_storage_views(self, device, dtype, numerics):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        components = torch.tensor(
            [[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0], [-0.0, -0.0], [1.0, -1.0]],
            dtype=real_dtype,
            device=device,
        )
        upstream = torch.view_as_complex(components)

        def fn(grad):
            return grad * complex(0.0, 0.4).conjugate()

        actual, _ = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": numerics}), upstream
        )
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(fn(upstream))
        )

        def storage_views(x):
            components = torch.view_as_real(x)
            return (
                x,
                components,
                components + 1,
                x.real + 1,
                x.imag + 1,
                torch.view_as_real(torch.view_as_complex(components)) + 2,
            )

        values = torch.cat(
            (
                _pointwise_subnormals(real_dtype).to(device),
                _min_max_specials(real_dtype, device),
            )
        )
        x = torch.view_as_complex(torch.stack((values, values.roll(1)), dim=-1))
        compiled = torch.compile(
            storage_views, fullgraph=True, options={"numerics": numerics}
        )
        for view in (x, x._neg_view(), x[::2]._neg_view()):
            expected = storage_views(view)
            actual = compiled(view)
            self._assert_bitwise_equal(actual, expected)
            self.assertEqual(
                [t.is_neg() for t in actual], [t.is_neg() for t in expected]
            )
            self.assertTrue(torch._C._is_alias_of(actual[0], view))
            self.assertTrue(torch._C._is_alias_of(actual[1], view))

        def cloned_outputs(a):
            return a.clone(), a.clone()

        from torch._inductor.utils import fresh_cache

        with (
            fresh_cache(),
            config.patch(fx_graph_cache=True, force_disable_caches=False),
            torch._functorch.config.patch(
                enable_autograd_cache=True, enable_remote_autograd_cache=False
            ),
        ):
            compiled = torch.compile(
                cloned_outputs, fullgraph=True, options={"numerics": numerics}
            )
            backward_compiler = torch.compile(backend="eager", fullgraph=True)
            for cache_hit in (False, True):
                torch._dynamo.reset()
                hits = counters["aot_autograd"]["autograd_cache_hit"]
                base = x.clone().requires_grad_()
                expected_base = base.detach().clone().requires_grad_()
                actual = compiled(base)
                expected = cloned_outputs(expected_base)
                if cache_hit:
                    self.assertGreater(
                        counters["aot_autograd"]["autograd_cache_hit"], hits
                    )
                # Prime the cached backward with tensor gradients before testing None.
                zeros = tuple(torch.zeros_like(output) for output in actual)
                self._assert_bitwise_equal(
                    torch.autograd.grad(actual, base, zeros, retain_graph=True),
                    torch.autograd.grad(
                        expected, expected_base, zeros, retain_graph=True
                    ),
                )
                self._assert_optional_output_gradients(
                    actual,
                    expected,
                    base,
                    expected_base,
                    x,
                    backward_compiler=backward_compiler,
                )

        def mutate_components(x, operation, view):
            components = torch.view_as_real(x)
            original = components.clone()
            if view == "full":
                destination = components
            elif view == "real":
                destination = components[:, 0]
            elif view == "nested":
                destination = components[::2, 0]
            else:
                destination = components.diagonal()
            if operation == "add":
                destination.add_(1)
            elif operation == "mul":
                destination.mul_(2)
                destination.mul_(0.5)
            else:
                destination.copy_(destination + 1)
            return original, components, x, components + 2

        compiled_mutation = torch.compile(
            mutate_components, fullgraph=True, options={"numerics": numerics}
        )
        for operation, view, strided in itertools.product(
            ("add", "mul", "copy"),
            ("full", "real", "nested", "diagonal"),
            (False, True),
        ):
            torch._dynamo.reset()
            if strided:
                base = torch.empty(2 * x.numel() + 1, dtype=dtype, device=device)
                base.fill_(7 + 8j)
                base[1::2].copy_(x)
            else:
                base = x.clone()
            expected_base = base.clone()
            actual_view = (base[1::2] if strided else base)._neg_view()
            expected_view = (
                expected_base[1::2] if strided else expected_base
            )._neg_view()
            expected = mutate_components(expected_view, operation, view)
            actual = compiled_mutation(actual_view, operation, view)
            self._assert_bitwise_equal(actual, expected)
            self._assert_bitwise_equal(
                torch.view_as_real(base), torch.view_as_real(expected_base)
            )
            self.assertEqual(
                [t.is_neg() for t in actual], [t.is_neg() for t in expected]
            )
            self.assertEqual(
                [torch._C._is_alias_of(t, actual_view) for t in actual],
                [torch._C._is_alias_of(t, expected_view) for t in expected],
            )

        input_components = torch.tensor(
            [
                [1.0, 2.0],
                [-2.0, 1.0],
                [-0.0, -0.0],
                [0.0, 0.0],
                [float("nan"), 1.0],
                [1.0, float("nan")],
                [-0.0, 0.0],
                [0.0, -0.0],
            ],
            dtype=real_dtype,
            device=device,
        )
        upstream = torch.tensor(
            [
                [0.0, 0.25],
                [0.5, 0.75],
                [1.0, 1.25],
                [1.0, 1.25],
                [0.5, 0.75],
                [0.5, 0.75],
                [-0.0, 1.0],
                [-0.0, 1.0],
            ],
            dtype=real_dtype,
            device=device,
        )

        def local_mutation(a, tracked, view):
            allocation = a.clone()
            x = allocation._neg_view()
            components = torch.view_as_real(x)
            old = components.clone()
            destination = (
                components
                if view == "full"
                else components[:, 0]
                if view == "real"
                else components.diagonal()
            )
            if tracked:
                destination.add_(1)
            else:
                with torch.no_grad():
                    destination.mul_(2)
            middle = components.clone()
            if tracked:
                destination.add_(2)
            else:
                with torch.no_grad():
                    destination.mul_(3)
            return (
                old * old + middle * middle + components * components,
                components,
                x,
                allocation,
            )

        compiled_local = torch.compile(
            local_mutation, fullgraph=True, options={"numerics": numerics}
        )
        for tracked in (False, True):
            for view in ("full", "real", "diagonal"):
                torch._dynamo.reset()
                base = torch.view_as_complex(input_components).clone().requires_grad_()
                expected_base = base.detach().clone().requires_grad_()
                expected = local_mutation(expected_base, tracked, view)
                actual = compiled_local(base, tracked, view)
                self._assert_bitwise_equal(actual, expected)
                self._assert_bitwise_equal(base, expected_base)
                self.assertEqual(
                    [torch._C._is_alias_of(t, base) for t in actual],
                    [torch._C._is_alias_of(t, expected_base) for t in expected],
                )
                self.assertEqual(
                    [t.is_neg() for t in actual], [t.is_neg() for t in expected]
                )
                for tensor in actual[1:]:
                    self.assertTrue(torch._C._is_alias_of(tensor, actual[3]))
                self._assert_bitwise_equal(
                    torch.autograd.grad(actual[0], base, upstream)[0],
                    torch.autograd.grad(expected[0], expected_base, upstream)[0],
                )
                self._assert_bitwise_equal(base, expected_base)

        def mutated_input(a, linear):
            components = torch.view_as_real(a)
            old = components.clone()
            with torch.no_grad():
                components.mul_(2)
            result = (
                old + components * 3 if linear else old * old + components * components
            )
            return result, components, a

        compiled_input = torch.compile(
            mutated_input, fullgraph=True, options={"numerics": numerics}
        )
        compile_backward = torch.compile(backend="eager", fullgraph=True)

        def input_view(kind):
            values = torch.view_as_complex(input_components)
            if kind == "offset":
                base = torch.full(
                    (values.numel() + 2,), 7 + 8j, dtype=dtype, device=device
                )
                base[1:-1].copy_(values)
                base.requires_grad_()
                return base, base[1:-1]._neg_view()
            base = values.clone().requires_grad_()
            view = base._neg_view()
            if kind == "nonview":
                view = view.detach().requires_grad_()
                return view, view
            return base, view

        for linear in (False, True):
            torch._dynamo.reset()
            for kind in ("nonview", "view", "offset", "cached"):
                if kind == "cached":
                    torch._dynamo.reset()
                base, view = input_view(kind)
                expected_base, expected_view = input_view(kind)
                hits = counters["aot_autograd"]["autograd_cache_hit"]
                expected = mutated_input(expected_view, linear)
                actual = compiled_input(view, linear)
                self._assert_bitwise_equal(actual, expected)
                self.assertEqual(
                    [t.is_neg() for t in actual], [t.is_neg() for t in expected]
                )
                self.assertTrue(torch._C._is_alias_of(actual[1], actual[2]))
                self.assertTrue(torch._C._is_alias_of(actual[2], base))
                backward_context = (
                    compiled_autograd._enable(compile_backward)
                    if kind == "cached"
                    else nullcontext()
                )
                with backward_context:
                    grad = torch.autograd.grad(actual[0], base, upstream)[0]
                self._assert_bitwise_equal(
                    grad, torch.autograd.grad(expected[0], expected_base, upstream)[0]
                )
                if kind == "cached":
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_hit"], hits + 1
                    )
                # A saved alias must not be donated to backward and overwrite input storage.
                self._assert_bitwise_equal(base, expected_base)

        def symbolic_alias(a, sizes, negate_before_slice):
            with torch.no_grad():
                a.add_(1)
            view = a._neg_view() if negate_before_slice else a
            length = sizes.shape[0]
            if negate_before_slice:
                length = int(length * 0.5)
            view = view.reshape(-1)[:length]
            if not negate_before_slice:
                view = view._neg_view()
            return a * 2, view

        grad_values = torch.view_as_complex(upstream)
        grad_storage = torch.stack((grad_values, grad_values), dim=-1)
        for negate_before_slice in (False, True):
            scale = 2 if negate_before_slice else 1
            torch._dynamo.reset()
            compiled = torch.compile(
                symbolic_alias,
                fullgraph=True,
                dynamic=True,
                options={"numerics": numerics},
            )
            base, view = input_view("view")
            warm = compiled(
                view, torch.empty(3 * scale, device=device), negate_before_slice
            )
            torch.autograd.grad(warm, base, tuple(torch.ones_like(t) for t in warm))
            torch._dynamo.reset()
            hits = counters["aot_autograd"]["autograd_cache_hit"]
            for size in (3, 5):
                base, view = input_view("view")
                ref_base, ref_view = input_view("view")
                sizes = torch.empty(size * scale, device=device)
                expected = symbolic_alias(ref_view, sizes, negate_before_slice)
                actual = compiled(view, sizes, negate_before_slice)
                self._assert_bitwise_equal(actual, expected)
                self.assertTrue(torch._C._is_alias_of(actual[1], view))
                self.assertEqual(
                    (
                        actual[1].stride(),
                        actual[1].storage_offset(),
                        actual[1].is_neg(),
                    ),
                    (
                        expected[1].stride(),
                        expected[1].storage_offset(),
                        expected[1].is_neg(),
                    ),
                )
                grad = grad_storage[:size, 0]
                saved_grad = grad.clone()
                self._assert_bitwise_equal(
                    torch.autograd.grad(actual[1], base, grad)[0],
                    torch.autograd.grad(expected[1], ref_base, grad)[0],
                )
                self._assert_bitwise_equal(base, ref_base)
                self._assert_bitwise_equal(grad, saved_grad)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], hits + 1)

        def real_mutation(a, local):
            allocation = a.clone() if local else a
            view = allocation._neg_view() if local else a
            old = view.clone()
            view.add_(1)
            return old, view, allocation

        compiled_real = torch.compile(
            real_mutation, fullgraph=True, options={"numerics": numerics}
        )
        for local, empty in itertools.product((False, True), (False, True)):
            base = values.clone()
            expected_base = base.clone()
            a = base[:0] if empty else base[1::2]
            expected_a = expected_base[:0] if empty else expected_base[1::2]
            if not local:
                a, expected_a = a._neg_view(), expected_a._neg_view()
            actual = compiled_real(a, local)
            expected = real_mutation(expected_a, local)
            self._assert_bitwise_equal(actual, expected)
            self._assert_bitwise_equal(base, expected_base)
            self.assertEqual(
                [t.is_neg() for t in actual], [t.is_neg() for t in expected]
            )
            self.assertTrue(torch._C._is_alias_of(actual[1], actual[2]))
            self.assertEqual(torch._C._is_alias_of(actual[1], a), not local)

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "weight",
        (
            -0.5 - 2**-10,
            -0.5,
            -0.5 + 2**-10,
            0.5 - 2**-10,
            0.5,
            0.5 + 2**-10,
            0.4j,
            1 + 2j,
        ),
    )
    def test_complex_lerp_scalar(self, device, dtype, numerics, weight):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _min_max_specials(real_dtype, device)
        components = torch.stack((values, values.flip(0)), dim=-1)
        values = torch.view_as_complex(components)
        start, end = values[:, None], values[None, :]

        def fn(start, end):
            return torch.lerp(start, end, weight)

        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": numerics}),
            start,
            end,
        )
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(fn(start, end))
        )
        self.assertIn("@triton.jit", "\n".join(code))
        self.assertNotIn("= torch.ops.aten.lerp.Scalar(", "\n".join(code))

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_complex_lerp_tensor_broadcast(self, device, dtype, numerics):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _min_max_specials(real_dtype, device)
        components = torch.stack((values, values.flip(0)), dim=-1)
        values = torch.view_as_complex(components)
        start, end = values[:, None], values[None, :]
        weights = torch.tensor(
            [
                -0.5 - 2**-10,
                -0.5,
                -0.5 + 2**-10,
                0.5 - 2**-10,
                0.5,
                0.5 + 2**-10,
                0.4j,
                1 + 2j,
                complex(-0.0, 0.0),
                0.5j,
            ],
            dtype=dtype,
            device=device,
        )[None, :]

        def fn(start, end, weights):
            return torch.lerp(start, end, weights)

        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": numerics}),
            start,
            end,
            weights,
        )
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(fn(start, end, weights))
        )
        self.assertIn("@triton.jit", "\n".join(code))
        self.assertNotIn("= torch.ops.aten.lerp.Tensor(", "\n".join(code))

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("weight_kind", ("tensor", "small", "large"))
    @parametrize(
        "layout", ("contiguous", "broadcast", "transposed", "offset1", "offset2")
    )
    def test_generated_complex_lerp_rounding(self, device, dtype, weight_kind, layout):
        torch.manual_seed(371)
        start = torch.randn(129, 127, device=device, dtype=dtype)
        end = torch.randn(
            129, 1 if layout == "broadcast" else 127, device=device, dtype=dtype
        )
        weight = torch.randn(
            1 if layout == "broadcast" else 129, 127, device=device, dtype=dtype
        )
        if layout == "transposed":
            start, end, weight = start.t(), end.t(), weight.t()
        elif layout.startswith("offset"):
            offset = int(layout[-1])
            start = torch.cat((start.new_zeros(offset), start.flatten()))[
                offset:
            ].view_as(start)
        if weight_kind != "tensor":
            weight = 0.1 + 0.2j if weight_kind == "small" else 1.1 + 2.2j
        fn = torch.lerp
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": "strict"}),
            start,
            end,
            weight,
        )
        self._assert_bitwise_equal(actual, fn(start, end, weight))
        self.assertIn("@triton.jit", "\n".join(code))
        self.assertNotIn("= torch.ops.aten.lerp.", "\n".join(code))

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("conjugated", (False, True))
    @parametrize("alpha", (None, 3.14))
    def test_complex_add_alpha(self, device, dtype, numerics, conjugated, alpha):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _min_max_specials(real_dtype, device)
        zeros = torch.tensor(
            [[-0.0, -0.0], [-0.0, 0.0], [0.0, -0.0], [0.0, 0.0]],
            dtype=real_dtype,
            device=device,
        )
        components = torch.cat((torch.stack((values, values.flip(0)), dim=-1), zeros))
        if dtype == torch.complex32:
            # Scaling followed by cancellation exposes premature half rounding.
            finite = torch.tensor(
                [[-1.869140625, 0.1026611328125], [0.6533203125, -0.51953125]],
                dtype=real_dtype,
                device=device,
            )
            components = torch.cat((components, finite))
        values = torch.view_as_complex(components)
        x = values.repeat_interleave(values.numel())
        y = values.repeat(values.numel())
        if conjugated:
            x, y = x.conj(), y.conj()

        def fn(x, y):
            return x + y if alpha is None else torch.add(x, y, alpha=alpha)

        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": numerics}), x, y
        )
        code = "\n".join(code)
        self._assert_bitwise_equal(
            torch.view_as_real(actual), torch.view_as_real(fn(x, y))
        )
        self.assertIn("@triton.jit", code)
        self.assertNotIn("= torch.ops.aten.add.", code)

    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_complex_unary_nan_payload(self, device, dtype, numerics, upcast):
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _min_max_specials(real_dtype, device)
        x = torch.view_as_complex(torch.cartesian_prod(values, values))

        def fn(x):
            return torch.angle(x), torch.view_as_real(torch.sgn(x))

        self._check_bitwise(fn, x, numerics=numerics, upcast=upcast)

    @dtypes(torch.complex64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_complex_add_mixed_nan_payload(self, device, dtype, numerics):
        from torch.utils._pytree import tree_map

        bits = torch.tensor([[0, 0x7FFFFFFF]], dtype=torch.int32, device=device)
        x = torch.view_as_complex(bits.view(torch.float32))
        y = torch.tensor(
            [complex(float("inf"), 0)], dtype=torch.complex128, device=device
        )

        def fn(x, y):
            return (
                x + y,
                y + x,
                torch.add(x, y, alpha=3.14),
                torch.add(y, x, alpha=3.14),
            )

        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, options={"numerics": numerics}), x, y
        )
        self._assert_bitwise_equal(
            tree_map(torch.view_as_real, actual), tree_map(torch.view_as_real, fn(x, y))
        )
        code = "\n".join(code)
        self.assertIn("@triton.jit", code)
        self.assertNotIn("= torch.ops.aten.add.", code)

    @ops(
        [op for op in op_db if op.name == "_native_batch_norm_legit"],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    def test_batch_norm_parameter_gradients(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            if len(sample.args) != 7:
                continue

            def fn(inp, *args):
                args = tuple(
                    x.clone() if isinstance(x, torch.Tensor) else x for x in args
                )
                return op.op(inp, *args)[0]

            # Request parameter gradients while keeping the input gradient disabled.
            inp = sample.input.detach()
            parameters = sample.args[:2]
            expected = fn(inp, *sample.args)
            actual = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})(
                inp, *sample.args
            )
            gradient = torch.ones_like(expected)
            self.assertTrue(
                _outputs_equal(
                    torch.autograd.grad(actual, parameters, gradient),
                    torch.autograd.grad(expected, parameters, gradient),
                )
            )

    @ops(
        [op for op in op_db if op.name == "_native_batch_norm_legit"],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("view", ("slice", "expand", "convolution"))
    @config.patch(force_layout_optimization=True, realize_reads_threshold=0)
    def test_batch_norm_computed_view(self, device, dtype, op, view):
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            if sample.input.ndim < 3 or (
                view == "convolution" and sample.input.ndim != 4
            ):
                continue

            def fn(inp, *args):
                inp = inp + 1
                if view == "slice":
                    inp = inp[..., ::2]
                elif view == "expand":
                    inp = inp[:, :, :1].expand_as(inp)
                # Isolate native updates to running statistics between calls.
                args = tuple(
                    x.clone() if isinstance(x, torch.Tensor) else x for x in args
                )
                result = op.op(inp, *args, **sample.kwargs)
                if view == "convolution":
                    # A convolution consumer makes Inductor prefer channels last.
                    weight = inp.new_ones((inp.shape[1], 1, 1, 1))
                    result = (
                        *result,
                        torch.nn.functional.conv2d(inp, weight, groups=inp.shape[1]),
                    )
                return result

            actual = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})(
                sample.input, *sample.args
            )
            self.assertTrue(_outputs_equal(actual, fn(sample.input, *sample.args)))

    def _check_bitwise(self, fn, *args, numerics, upcast=None):
        options = {"numerics": numerics}
        if upcast is not None:
            options["triton.codegen_upcast_to_fp32"] = upcast
        actual = torch.compile(fn, fullgraph=True, options=options)(*args)
        self._assert_bitwise_equal(actual, fn(*args))

    @dtypes(torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("source", ("power", "product", "floor", "difference"))
    def test_whole_input_stride_power(self, device, dtype, numerics, source):
        import operator

        from torch._inductor.fx_passes.pre_grad import _preserve_symbolic_strides
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        env = ShapeEnv()
        u, v, k = [env.create_unbacked_symint() for _ in range(3)]
        torch._check(u >= 3)
        torch._check(v >= 3)
        torch._check(u > v)
        torch._check(k >= 0)
        stride = {
            "power": u**2,
            "product": u * v,
            "floor": u // 2,
            "difference": u - v,
        }[source]
        mode = FakeTensorMode(shape_env=env)
        with mode:
            fake = torch.empty(
                (2 * (stride**2 + k**2) + 3,), device=device, dtype=dtype
            ).as_strided((3, 3), (stride, 1))
        env.pending_fresh_unbacked_symbols.clear()
        graph = torch.fx.Graph()
        tensor = graph.placeholder("x")
        scalar = graph.placeholder("k")
        query = graph.call_method("stride", (tensor, 0))
        square = graph.call_function(operator.mul, (query, query))
        offset = graph.call_function(operator.mul, (scalar, scalar))
        summed = graph.call_function(operator.add, (square, offset))
        view = graph.call_function(
            torch.ops.aten.as_strided.default, (tensor, (3, 3), (summed, 1))
        )
        query = graph.call_method("stride", (view, 0))
        graph.output(query)
        gm = torch.fx.GraphModule({}, graph)
        FakeTensorProp(gm, mode=mode).propagate(fake, k)
        with config.patch(numerics=numerics):
            _preserve_symbolic_strides(gm)
        graph.lint()
        gm.recompile()
        for stride, known in ((9, 0), (10, 3)):
            tensor = torch.empty(
                2 * (stride**2 + known**2) + 3, device=device, dtype=dtype
            ).as_strided((3, 3), (stride, 1))
            self.assertEqual(gm(tensor, known), stride**2 + known**2)

    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("op", ("add", "mul", "neg"))
    @parametrize("prefix", (0, 1, 2))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_pointwise_unbacked_output_strides(
        self, device, dtype, numerics, op, prefix
    ):
        def fn(mask1, mask2):
            m = torch.nonzero(mask1).shape[0]
            n = torch.nonzero(mask2).shape[0]
            size = (*(2,) * prefix, m, n)
            prefix_stride = tuple(m * n * 2**i for i in reversed(range(prefix)))
            stride = (*prefix_stride, 1, m)
            x = torch.empty_strided(size, stride, device=device, dtype=dtype)
            x.fill_(0.5)
            out = torch.neg(x) if op == "neg" else getattr(torch, op)(x, x)
            return out, out + 1, out.stride()

        counter = CompileCounterWithBackend("inductor")
        metrics.reset()
        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
            for m, n in itertools.product((0, 1, 2), repeat=2):
                with self.subTest(m=m, n=n):
                    mask1 = torch.arange(4, device=device) < m
                    mask2 = torch.arange(4, device=device) < n
                    expected = fn(mask1, mask2)
                    actual = compiled(mask1, mask2)
                    self._assert_bitwise_equal(actual[:2], expected[:2])
                    self.assertEqual(actual[0].stride(), expected[0].stride())
                    self.assertEqual(actual[2], expected[2])
                    self.assertEqual(actual[2], actual[0].stride())
        self.assertEqual(counter.frame_count, 1)
        self.assertGreater(metrics.generated_kernel_count, 0)

    @dtypes(torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("two_dimensions", (False, True))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_addr_unbacked_strides(self, device, dtype, numerics, two_dimensions):
        def fn(mask1, mask2):
            m = torch.nonzero(mask1).shape[0]
            n = torch.nonzero(mask2).shape[0] if two_dimensions else 3
            x = torch.empty_strided((m, n), (1, m), device=device, dtype=dtype)
            x.fill_(0.5)
            a = torch.ones(m, device=device, dtype=dtype)
            b = torch.ones(n, device=device, dtype=dtype)
            out = torch.addr(x, a, b, beta=0, alpha=2.3)
            return out, out + 1, out.stride()

        counter = CompileCounterWithBackend("inductor")
        metrics.reset()
        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
            for m in (0, 1, 2):
                for n in (0, 1, 2) if two_dimensions else (3,):
                    mask1 = torch.arange(4, device=device) < m
                    mask2 = torch.arange(4, device=device) < n
                    expected = fn(mask1, mask2)
                    actual = compiled(mask1, mask2)
                    self._assert_bitwise_equal(actual[:2], expected[:2])
                    self.assertEqual(actual[0].stride(), expected[0].stride())
                    self.assertEqual(actual[2], expected[2])
                    self.assertEqual(actual[2], actual[0].stride())
        self.assertEqual(counter.frame_count, 1)
        self.assertGreater(metrics.generated_kernel_count, 0)

        if not two_dimensions:
            from torch._inductor.fx_passes.pre_grad import _preserve_symbolic_strides
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp

            # Input producers must read the incoming geometry before mutations.
            for symbolic_shape in (False, True):
                for mutation, args in (("transpose_", (0, 1)), ("unsqueeze_", (0,))):
                    shape_env = ShapeEnv()
                    u = shape_env.create_unbacked_symint()
                    torch._check(u >= 3)
                    mode = FakeTensorMode(shape_env=shape_env)
                    with mode:
                        fake = torch.empty_strided(
                            (3, u if symbolic_shape else 3),
                            (u, 1),
                            device=device,
                            dtype=dtype,
                        )
                    graph = torch.fx.Graph()
                    x = graph.placeholder("x")
                    mutated = graph.call_method(mutation, (x, *args))
                    query = graph.call_method("stride", (mutated,))
                    graph.output(query)
                    gm = torch.fx.GraphModule({}, graph)
                    FakeTensorProp(gm, mode=mode).propagate(fake)
                    self.assertEqual(x.meta["val"].stride()[1], 1)
                    with config.patch(numerics=numerics):
                        _preserve_symbolic_strides(gm)
                    graph.lint()
                    gm.recompile()
                    nodes = list(graph.nodes)
                    for node in nodes:
                        if node.target in (
                            torch.ops.aten.sym_stride.int,
                            torch.ops.aten.sym_size.int,
                        ):
                            self.assertLess(nodes.index(node), nodes.index(mutated))
                    for count in (3, 5):
                        tensor = torch.empty_strided(
                            (3, count if symbolic_shape else 3),
                            (count, 1),
                            device=device,
                            dtype=dtype,
                        )
                        expected = (
                            (1, count)
                            if mutation == "transpose_"
                            else (3 * count, count, 1)
                        )
                        self.assertEqual(gm(tensor), expected)
                        self.assertEqual(tensor.stride(), expected)

            for mutation, args in (("transpose_", (0, 1)), ("unsqueeze_", (0,))):
                mode = FakeTensorMode(shape_env=ShapeEnv())
                with mode:
                    fake = torch.empty((3, 4), device=device, dtype=dtype)
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                producer = graph.call_function(torch.ops.aten.nonzero.default, (x,))
                mutated = graph.call_method(mutation, (producer, *args))
                query = graph.call_method("stride", (mutated,))
                graph.output(query)
                gm = torch.fx.GraphModule({}, graph)
                FakeTensorProp(gm, mode=mode).propagate(fake)
                self.assertTrue(producer.meta["unbacked_bindings"])
                with config.patch(numerics=numerics):
                    _preserve_symbolic_strides(gm)
                graph.lint()
                gm.recompile()
                nodes = list(graph.nodes)
                for node in nodes:
                    if node.target == torch.ops.aten.sym_size.int:
                        self.assertLess(nodes.index(producer), nodes.index(node))
                        self.assertLess(nodes.index(node), nodes.index(mutated))
                for count in (3, 5, 9):
                    tensor = torch.zeros((3, 4), device=device, dtype=dtype)
                    tensor.view(-1)[:count] = 1
                    expected = getattr(tensor.nonzero(), mutation)(*args).stride()
                    self.assertEqual(gm(tensor), expected)

            csr = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_csr()
            for tensor in (csr, csr.to("meta"), FakeTensorMode().from_tensor(csr)):
                shape_env = ShapeEnv()
                symbol = shape_env.create_unbacked_symint()
                graph = torch.fx.Graph()
                graph.placeholder("s").meta["val"] = symbol
                graph.placeholder("csr").meta["val"] = tensor
                result = graph.materialize_symints((symbol + 1,))
                graph.output(result)
                graph.lint()
                gm = torch.fx.GraphModule({}, graph)
                self.assertEqual(gm(7, csr), [8])

            from itertools import permutations

            combinations = [
                ("shared", (0, 1)),
                ("shared", (1, 0)),
                ("floor", (0, 1)),
                ("floor", (1, 0)),
            ]
            combinations += [
                (kind, order)
                for kind in ("overcomplete", "scaled", "symbolic")
                for order in permutations(range(3))
            ]
            for kind, order in combinations:
                shape_env = ShapeEnv()
                mode = FakeTensorMode(shape_env=shape_env)
                u, v, w, scale, weight = [
                    shape_env.create_unbacked_symint() for _ in range(5)
                ]
                for symbol in (u, v, w):
                    torch._check(symbol >= 3)
                torch._check(u > v)
                torch._check(scale >= 0)
                torch._check(weight >= 0)
                if kind in ("shared", "floor"):
                    producers = (u + v if kind == "shared" else u // 2 + v, u + w)
                elif kind == "symbolic":
                    producers = (
                        scale * u + v,
                        u + scale * v,
                        (scale + 2) * u + (scale + 3) * v,
                    )
                else:
                    producers = (u + v, u - v, u + 2 * v)
                    if kind == "scaled":
                        producers = tuple(scale * producer for producer in producers)
                producers = tuple(producers[i] for i in order)
                values = (
                    sum(producers),
                    producers[0] - producers[1],
                    scale * producers[0] + weight * sum(producers[1:]),
                )
                with mode:
                    fake = torch.empty_strided(
                        (3,) * len(producers), producers, device=device, dtype=dtype
                    )
                graph = torch.fx.Graph()
                graph.placeholder("x").meta["val"] = fake
                graph.placeholder("scale").meta["val"] = scale
                graph.placeholder("weight").meta["val"] = weight
                graph.output(tuple(graph.materialize_symints(values)))
                graph.lint()
                gm = torch.fx.GraphModule({}, graph)
                self.assertTrue(
                    all(
                        node.meta.get("val") is not None
                        for node in graph.nodes
                        if node.op == "call_function"
                    )
                )
                for concrete in ((7, 3, 4, 2, 3), (8, 4, 5, 3, 2), (9, 5, 6, 0, 1)):
                    substitutions = {
                        symbol.node.expr: value
                        for symbol, value in zip((u, v, w, scale, weight), concrete)
                    }
                    strides = tuple(
                        int(producer.node.expr.subs(substitutions))
                        for producer in producers
                    )
                    x = torch.empty_strided(
                        (3,) * len(producers), strides, device=device, dtype=dtype
                    )
                    expected = tuple(
                        int(value.node.expr.subs(substitutions)) for value in values
                    )
                    self.assertEqual(gm(x, concrete[3], concrete[4]), expected)

            # Independently known terms belong in the reconstructed remainder.
            import operator

            for known_term in ("identity", "square", "floor"):
                shape_env = ShapeEnv()
                u, k = [shape_env.create_unbacked_symint() for _ in range(2)]
                torch._check(u >= 3)
                torch._check(k >= 3)
                extra = (
                    k
                    if known_term == "identity"
                    else k * k
                    if known_term == "square"
                    else k // 2
                )
                mode = FakeTensorMode(shape_env=shape_env)
                with mode:
                    fake = torch.empty(
                        (2 * (u + k + extra) + 3,), device=device, dtype=dtype
                    ).as_strided((3, 3), (u + k, 1))
                # These symbols were bound before entering this nested graph.
                shape_env.pending_fresh_unbacked_symbols.clear()
                graph = torch.fx.Graph()
                tensor = graph.placeholder("x")
                scalar = graph.placeholder("k")
                stride = graph.call_method("stride", (tensor, 0))
                term = scalar
                if known_term != "identity":
                    term = graph.call_function(
                        operator.mul if known_term == "square" else operator.floordiv,
                        (scalar, scalar if known_term == "square" else 2),
                    )
                summed = graph.call_function(operator.add, (stride, term))
                view = graph.call_function(
                    torch.ops.aten.as_strided.default, (tensor, (3, 3), (summed, 1))
                )
                query = graph.call_method("stride", (view, 0))
                graph.output(query)
                gm = torch.fx.GraphModule({}, graph)
                FakeTensorProp(gm, mode=mode).propagate(fake, k)
                with config.patch(numerics=numerics):
                    _preserve_symbolic_strides(gm)
                graph.lint()
                gm.recompile()
                nodes = list(graph.nodes)
                for node in nodes:
                    if node.target == torch.ops.aten.sym_stride.int:
                        self.assertLess(nodes.index(tensor), nodes.index(node))
                        self.assertLess(nodes.index(node), nodes.index(view))
                    elif node.target in (
                        operator.add,
                        operator.mul,
                        operator.floordiv,
                        operator.pow,
                    ):
                        self.assertLess(nodes.index(scalar), nodes.index(node))
                for unknown, known in ((16, 4), (7, 3), (9, 5)):
                    tensor = torch.empty(256, device=device, dtype=dtype).as_strided(
                        (3, 3), (unknown + known, 1)
                    )
                    extra = (
                        known
                        if known_term == "identity"
                        else known * known
                        if known_term == "square"
                        else known // 2
                    )
                    self.assertEqual(gm(tensor, known), unknown + known + extra)

            for target, dim, keyword, result_type in (
                ("stride", None, False, tuple),
                ("stride", 0, False, int),
                ("stride", 0, True, int),
                (torch.ops.aten.sym_stride.default, None, False, list),
                (torch.ops.aten.sym_stride.default, None, True, list),
                (torch.ops.aten.sym_stride.int, 0, False, int),
                (torch.ops.aten.sym_stride.int, 0, True, int),
                (torch.ops.aten.sym_stride.int, -2, False, int),
                (torch.ops.aten.sym_stride.int, -1, False, int),
            ):
                shape_env = ShapeEnv()
                mode = FakeTensorMode(shape_env=shape_env)
                u = shape_env.create_unbacked_symint()
                torch._check(u >= 0)
                with mode:
                    fake = torch.empty_strided(
                        (3, 3), (u, 1), device=device, dtype=dtype
                    )
                graph = torch.fx.Graph()
                tensor = graph.placeholder("x")
                args = (tensor,) if dim is None or keyword else (tensor, dim)
                kwargs = {"dim": dim} if dim is not None and keyword else {}
                if isinstance(target, str):
                    query = graph.call_method(target, args, kwargs)
                else:
                    if keyword:
                        args = ()
                        kwargs["self"] = tensor
                    query = graph.call_function(target, args, kwargs)
                graph.output(query)
                gm = torch.fx.GraphModule({}, graph)
                FakeTensorProp(gm, mode=mode).propagate(fake)
                with config.patch(numerics=numerics):
                    _preserve_symbolic_strides(gm)
                gm.graph.lint()
                gm.recompile()
                for count in (0, 1, 5):
                    x = torch.empty_strided(
                        (3, 3), (count, 1), device=device, dtype=dtype
                    )
                    expected = (
                        list(x.stride())
                        if result_type is list
                        else x.stride()
                        if dim is None
                        else x.stride(dim)
                    )
                    actual = gm(x)
                    self.assertIs(type(actual), result_type)
                    self.assertEqual(actual, expected)

            # Branch inputs can expose whole stride expressions without
            # exposing the individual symbols needed by their view strides.
            def cond_fn(mask1, mask2, mask3, predicate, source, view):
                u = torch.nonzero(mask1).shape[0]
                torch._check(u >= 3)
                stride1 = 1
                if source in (
                    "sum",
                    "two_sources",
                    "mixed_sign",
                    "shared_leading",
                    "floor_composite",
                ):
                    v = torch.nonzero(mask2).shape[0]
                    torch._check(v >= 3)
                    stride0 = u + v
                    if source in (
                        "two_sources",
                        "mixed_sign",
                        "shared_leading",
                        "floor_composite",
                    ):
                        w = torch.nonzero(mask3).shape[0]
                        torch._check(w >= 3)
                        if source == "mixed_sign":
                            torch._check(u > v)
                            torch._check(v > w)
                            stride0, stride1 = u - v, v - w
                        elif source in ("shared_leading", "floor_composite"):
                            stride1 = u + w
                            if source == "floor_composite":
                                stride0 = u // 2 + v
                        else:
                            stride1 = w + v
                elif source == "negative":
                    stride0 = u - 1
                elif source == "scaled_positive":
                    stride0 = 2 * u + 1
                elif source == "scaled_negative":
                    stride0 = 3 * u - 2
                elif source == "nonlinear":
                    stride0 = u * u + 1
                else:
                    stride0 = u + (source == "positive")
                x = torch.empty_strided(
                    (3, 3), (stride0, stride1), device=device, dtype=dtype
                )
                x.fill_(0.5)

                def view_stride(t):
                    if view == "transpose":
                        return t.t().stride(1)
                    if view == "slice":
                        return t[::2].stride(0)
                    if view == "diagonal":
                        return t.diagonal().stride(0)
                    if view == "select":
                        return t.select(1, 0).stride(0)
                    if view == "unfold":
                        return t.unfold(0, 1, 2).stride(0)
                    if view == "reshape":
                        return t[::2, 0].reshape(2, 1).stride(0)
                    return t.stride(0)

                def true_branch(t):
                    return torch.ones((view_stride(t),), device=device, dtype=dtype)

                def false_branch(t):
                    return torch.ones((view_stride(t),), device=device, dtype=dtype) + 1

                return torch.cond(predicate, true_branch, false_branch, (x,))

            scenarios = (
                ("bare", "input"),
                ("positive", "input"),
                ("positive", "transpose"),
                ("positive", "slice"),
                ("positive", "diagonal"),
                ("negative", "unfold"),
                ("scaled_positive", "reshape"),
                ("scaled_negative", "slice"),
                ("sum", "slice"),
                ("nonlinear", "unfold"),
                ("two_sources", "diagonal"),
                ("mixed_sign", "diagonal"),
                ("shared_leading", "diagonal"),
                ("floor_composite", "diagonal"),
                ("negative", "select"),
            )
            for source, view in scenarios:
                torch._dynamo.reset()
                counter = CompileCounterWithBackend("inductor")
                with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
                    compiled = torch.compile(
                        cond_fn, backend=counter, fullgraph=True, dynamic=True
                    )
                    for count in (3, 4, 5):
                        counts = (
                            (count + 4, count + 2, count)
                            if source == "mixed_sign"
                            else (count, 8 - count, count)
                        )
                        if source in ("shared_leading", "floor_composite"):
                            counts = (count, 8 - count, (5, 3, 4)[count - 3])
                        masks = tuple(
                            torch.arange(10, device=device) < n for n in counts
                        )
                        for predicate in (False, True):
                            pred = torch.tensor(predicate, device=device)
                            expected = cond_fn(*masks, pred, source, view)
                            actual = compiled(*masks, pred, source, view)
                            self.assertEqual(actual.shape, expected.shape)
                            self._assert_bitwise_equal(actual, expected)
                self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_mv_computed_matrix_layout(self, device, dtype, numerics):
        torch.manual_seed(10)
        a = torch.randn(19, 37, device=device, dtype=dtype)
        b = torch.randn(37, 19, device=device, dtype=dtype).t()
        v = torch.randn(37, device=device, dtype=dtype)

        def fn(a, b, v):
            return torch.mv(b + a, v)

        actual = torch.compile(fn, fullgraph=True, options={"numerics": numerics})(
            a, b, v
        )
        expected = fn(a, b, v)
        if dtype.is_complex:
            actual = torch.view_as_real(actual)
            expected = torch.view_as_real(expected)
        self._assert_bitwise_equal(actual, expected)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_mvlgamma_unbacked_size(self, device, dtype, numerics):
        def fn(x):
            return torch.mvlgamma(x[x > 10], 3)

        x = torch.linspace(-30, 30, 257, device=device, dtype=dtype).requires_grad_()
        expected = fn(x)
        grad = torch.ones_like(expected)
        expected_grad = torch.autograd.grad(expected, x, grad)[0]
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            result, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"numerics": numerics}), x
            )
            self.assertIn("@triton.jit", "\n".join(codes))
            actual_grad = torch.autograd.grad(result, x, grad)[0]
        if numerics == "strict":
            self._assert_bitwise_equal(result, expected)
            self._assert_bitwise_equal(actual_grad, expected_grad)
        else:
            self.assertEqual(result, expected)
            self.assertEqual(actual_grad, expected_grad)

    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("p", (1, 3, 5, 9, 15, 16))
    @parametrize("layout", ("contiguous", "transposed", "strided"))
    def test_mvlgamma_sum_order(self, device, dtype, numerics, p, layout):
        from torch._native.registry import (
            deregister_op_overrides,
            native_decomp_table,
            reenable_op_overrides,
        )

        if torch.ops.aten.sum.dim_IntList not in native_decomp_table(
            overrides_only=True
        ):
            self.skipTest("requires active native inner-tree sum registration")
        if layout == "transposed":
            x = (
                torch.linspace(10, 30, 257 * 3, device=device, dtype=dtype)
                .reshape(3, 257)
                .t()
            )
        elif layout == "strided":
            x = torch.linspace(10, 30, 257 * 2, device=device, dtype=dtype)[::2]
        else:
            x = torch.linspace(10, 30, 257, device=device, dtype=dtype)
        x.requires_grad_()
        grad = torch.linspace(-1.3, 1.7, x.numel(), device=device, dtype=dtype).reshape(
            x.shape
        )

        def fn(x):
            return torch.mvlgamma(x, p)

        with config.patch(force_disable_caches=False, fx_graph_cache=True):
            for index, inner_tree in enumerate((True, False, True)):
                if not inner_tree:
                    deregister_op_overrides(disable_op_symbols="sum.dim_IntList")
                try:
                    expected = fn(x)
                    expected_grad = torch.autograd.grad(expected, x, grad)[0]
                    hits = counters["aot_autograd"]["autograd_cache_hit"]
                    misses = counters["aot_autograd"]["autograd_cache_miss"]
                    result, codes = run_and_get_code(
                        torch.compile(
                            fn, fullgraph=True, options={"numerics": numerics}
                        ),
                        x,
                    )
                    if index == 0:
                        self.assertIn("@triton.jit", "\n".join(codes))
                    grads, codes = run_and_get_code(
                        torch.autograd.grad, result, x, grad
                    )
                    if index == 0:
                        self.assertIn("@triton.jit", "\n".join(codes))
                    self._assert_bitwise_equal(result, expected)
                    self._assert_bitwise_equal(grads[0], expected_grad)
                    if index < 2:
                        self.assertGreater(
                            counters["aot_autograd"]["autograd_cache_miss"], misses
                        )
                    else:
                        self.assertGreater(
                            counters["aot_autograd"]["autograd_cache_hit"], hits
                        )
                finally:
                    if not inner_tree:
                        reenable_op_overrides(enable_op_symbols="sum.dim_IntList")
                    torch._dynamo.reset()

    @dtypes(torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "op_name",
        (
            "erfcx",
            "i0",
            "i1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
        ),
    )
    def test_special_function_boundaries(self, device, dtype, numerics, op_name):
        if op_name == "erfcx":
            centers = (
                400.0 / torch.arange(1, 101, dtype=torch.float64, device=device) - 4
            ).to(dtype)
            below = torch.nextafter(centers, torch.full_like(centers, -float("inf")))
            above = torch.nextafter(centers, torch.full_like(centers, float("inf")))
            values = torch.cat((centers, below, above, -centers))
        else:
            centers = torch.tensor(
                [-8.0, -5.0, -0.0, 0.0, 1e-5, 5.0, 8.0, 88.0],
                device=device,
                dtype=dtype,
            ).view(torch.int32)
            bits = centers[:, None] + torch.arange(
                -8, 9, device=device, dtype=torch.int32
            )
            values = bits.flatten().view(dtype)
        x = torch.cat((values, _min_max_specials(dtype, device)))
        fn = getattr(torch.special, op_name)
        self._check_bitwise(fn, x, numerics=numerics)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("lambd", (-0.5, 0.0, 0.3, float("inf"), float("nan")))
    def test_softshrink_backward_threshold(self, device, dtype, numerics, lambd):
        x = _min_max_specials(dtype, device)
        grad = x.flip(0)

        def fn(grad, x):
            return torch.ops.aten.softshrink_backward.default(grad, x, lambd)

        self._check_bitwise(fn, grad, x, numerics=numerics)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_softshrink_backward_overflow(self, device, dtype):
        x = torch.ones(4, device=device, dtype=dtype)
        lambd = torch.finfo(dtype).max * 2

        def fn(x):
            return torch.ops.aten.softshrink_backward.default(x, x, lambd)

        with self.assertRaisesRegex(RuntimeError, "overflow"):
            fn(x)
        compiled = torch.compile(
            fn, fullgraph=True, options={"numerics": "strict_pointwise"}
        )
        with self.assertRaisesRegex(Exception, "overflow"):
            compiled(x)

    @dtypes(torch.float32, torch.float64, torch.int16)
    @parametrize("op_name", ("bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1"))
    def test_generated_bessel_denominator_order(self, device, dtype, op_name):
        x = torch.tensor(
            [-100, -20, -10, -6, -5, -4, -2, -1, 0, 1, 2, 4, 5, 6, 10, 20, 100],
            device=device,
            dtype=dtype,
        )
        fn = getattr(torch.special, op_name)
        metrics.reset()
        actual = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})(x)
        self._assert_bitwise_equal(actual, fn(x))
        self.assertEqual(metrics.generated_fallback_kernel_count, 0)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("op_name", ("i0", "i1"))
    @parametrize("upcast", (False, True))
    def test_bessel_opmath(self, device, dtype, numerics, op_name, upcast):
        x = _exhaustive_16bit(dtype, device)
        fn = getattr(torch.special, op_name)
        self._check_bitwise(fn, x, numerics=numerics, upcast=upcast)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "eps",
        (None, -1.0, -1e-100, -(2.0**-149), -(2.0**-150), 2.0**-150, 0.0, 0.1, 0.7),
    )
    def test_logit_backward_bounds(self, device, dtype, numerics, eps):
        x = torch.tensor(
            [
                -float("inf"),
                -1.0,
                -0.0,
                0.0,
                0.1,
                0.5,
                0.7,
                1.0,
                float("inf"),
                float("nan"),
            ],
            device=device,
            dtype=dtype,
        )
        grad = x.flip(0)

        def fn(grad, x):
            return torch.ops.aten.logit_backward.default(grad, x, eps)

        self._check_bitwise(fn, grad, x, numerics=numerics)

    @dtypes(torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_xlogy_scalar_log(self, device, dtype, numerics):
        x = torch.linspace(-10, 10, 1024, dtype=dtype, device=device)

        def fn(x):
            return torch.xlogy(x, 2.7296853065490723)

        self._check_bitwise(fn, x, numerics=numerics)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("scalar", (False, True))
    def test_copysign_storage_bits(self, device, dtype, numerics, scalar):
        x = _exhaustive_16bit(dtype, device)
        y = -float("nan") if scalar else x.flip(0)
        self._check_bitwise(torch.copysign, x, y, numerics=numerics)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("left", (False, True))
    def test_scalar_nan_product(self, device, dtype, numerics, left):
        x = torch.tensor(
            [[0.0, -0.0, 1.0], [float("inf"), -float("inf"), float("nan")]],
            dtype=dtype,
            device=device,
        ).t()

        def fn(x):
            result = float("nan") * x if left else x * float("nan")
            return result.t().view(-1)

        self._check_bitwise(fn, x, numerics=numerics)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    @parametrize(
        "operation",
        (
            "binary_cross_entropy",
            "binary_cross_entropy_forward",
            "soft_margin",
            "smooth_l1",
            "mse",
        ),
    )
    def test_loss_intermediate_rounding(
        self, device, dtype, numerics, upcast, operation
    ):
        if dtype in (torch.float16, torch.bfloat16):
            x = _exhaustive_16bit(dtype, device)
        else:
            if operation in ("binary_cross_entropy", "binary_cross_entropy_forward"):
                values = [0.0, 1e-12, 0.1, 0.5, 1.0]
            elif operation == "soft_margin":
                values = [-104.0, -50.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 50.0, 104.0]
            else:
                values = [-10.0, -5.0, -1.0, -0.0, 0.0, 1.0, 5.0, 10.0]
            bounds = torch.tensor(values, device=device, dtype=dtype)
            x = torch.cat(
                (
                    _min_max_specials(dtype, device).repeat(128),
                    bounds,
                    torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
                    torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
                    torch.linspace(
                        values[0], values[-1], 257, device=device, dtype=dtype
                    ),
                )
            )
        if operation in ("binary_cross_entropy", "binary_cross_entropy_forward"):
            x = x[(x >= 0) & (x <= 1)]
            grad, target = x.roll(1) - 0.5, x.flip(0)
        elif operation == "soft_margin":
            grad, target = x.roll(1), torch.ones_like(x)
        elif operation == "mse":
            grad, target = x.roll(1), x.flip(0)
        else:
            grad, target = x.roll(1), torch.zeros_like(x)

        def fn(grad, x, target):
            if operation == "mse":
                result = torch.ops.aten.mse_loss_backward.default(grad, x, target, 0)
                return result, result.double()
            if operation == "smooth_l1":
                result = torch.ops.aten.smooth_l1_loss_backward.default(
                    grad, x, target, 0, 5.0
                )
                large = torch.ops.aten.smooth_l1_loss_backward.default(
                    grad, x, target, 0, 1e100
                )
                return result, result.double(), large, large.double()
            if operation == "binary_cross_entropy_forward":
                result = torch.ops.aten.binary_cross_entropy.default(x, target, None, 0)
                weighted = torch.ops.aten.binary_cross_entropy.default(
                    x, target, grad, 0
                )
                return result, result.double(), weighted, weighted.double()
            if operation == "soft_margin":
                result = torch.ops.aten.soft_margin_loss_backward.default(
                    grad, x, target, 0
                )
                return result, result.double()
            result = torch.ops.aten.binary_cross_entropy_backward.default(
                grad, x, target, None, 0
            )
            weighted = torch.ops.aten.binary_cross_entropy_backward.default(
                grad, x, target, target, 0
            )
            return result, result.double(), weighted, weighted.double()

        self._check_bitwise(fn, grad, x, target, numerics=numerics, upcast=upcast)
        if operation in ("smooth_l1", "mse") and dtype == torch.float64:
            nans = _min_max_specials(dtype, device)[-4:].repeat(384)
            offset_grad = torch.cat((nans[:1], nans.roll(1)))[1:]

            def cloned_fn(grad, x, target):
                return fn(grad.clone(), x, target)

            for target in (torch.zeros_like(nans), nans.roll(2)):
                strided_target = torch.stack((target, target), dim=1)[:, 0]
                for grad, current_target in (
                    (nans.roll(1), target),
                    (offset_grad, target),
                    (nans.roll(1), strided_target),
                ):
                    self._check_bitwise(
                        fn, grad, nans, current_target, numerics=numerics, upcast=upcast
                    )
                self._check_bitwise(
                    cloned_fn,
                    offset_grad,
                    nans,
                    target,
                    numerics=numerics,
                    upcast=upcast,
                )
        if operation == "soft_margin" and dtype == torch.float64:
            nans = _min_max_specials(dtype, device)[-4:].repeat(256)

            def cloned_fn(grad, x, target):
                return torch.ops.aten.soft_margin_loss_backward.default(
                    grad.clone(), x, target, 0
                )

            offset_grad = torch.cat((nans[:1], nans.roll(1)))[1:]
            self._check_bitwise(
                cloned_fn,
                offset_grad,
                nans,
                torch.ones_like(nans),
                numerics=numerics,
                upcast=upcast,
            )

            def reduced_fn(grad, x, target):
                return tuple(
                    torch.ops.aten.soft_margin_loss_backward.default(grad, x, target, r)
                    for r in (1, 2)
                )

            self._check_bitwise(
                reduced_fn,
                nans.roll(1),
                nans,
                torch.ones_like(nans),
                numerics=numerics,
                upcast=upcast,
            )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    @parametrize(
        "operation",
        (
            "cast",
            "cube",
            "mse",
            "soft_margin",
            "smooth_l1",
            "huber",
            "huber_backward",
            "binary_cross_entropy_with_logits",
        ),
    )
    def test_narrow_intermediate_rounding(
        self, device, dtype, numerics, upcast, operation
    ):
        if operation == "cast":
            mantissa_bits = 10 if dtype == torch.float16 else 7
            midpoint = 1.0 + 2.0 ** (-mantissa_bits - 1)
            values = [midpoint - 2**-40, midpoint, midpoint + 2**-40]
            x = torch.tensor(
                values + [-v for v in values], device=device, dtype=torch.float64
            )
        else:
            x = _exhaustive_16bit(dtype, device)

        def fn(x):
            if operation == "cast":
                result = x.to(dtype)
            elif operation == "cube":
                result = x.pow(3)
            elif operation == "mse":
                result = torch.ops.aten.mse_loss.default(x, x.flip(0), 0)
            elif operation == "soft_margin":
                result = torch.ops.aten.soft_margin_loss.default(
                    x, torch.ones_like(x), 0
                )
            elif operation == "binary_cross_entropy_with_logits":
                target, weight, pos_weight = x.flip(0), x.roll(1), x.roll(2)
                results = tuple(
                    torch.ops.aten.binary_cross_entropy_with_logits.default(
                        x, target, w, p, 0
                    )
                    for w, p in (
                        (None, None),
                        (weight, None),
                        (None, pos_weight),
                        (weight, pos_weight),
                    )
                )
                return results, tuple(result.double() for result in results)
            elif operation == "smooth_l1":
                tiny = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                result = torch.ops.aten.smooth_l1_loss.default(x, x.flip(0), 0, 0.7)
                small = torch.ops.aten.smooth_l1_loss.default(x, x.flip(0), 0, tiny)
                large = torch.ops.aten.smooth_l1_loss.default(x, x.flip(0), 0, 1e100)
                return (
                    result,
                    result.double(),
                    small,
                    small.double(),
                    large,
                    large.double(),
                )
            else:
                tiny = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                if operation == "huber":
                    result = torch.ops.aten.huber_loss.default(x, x.flip(0), 0, 0.7)
                    small = torch.ops.aten.huber_loss.default(x, x.flip(0), 0, tiny)
                    large = torch.ops.aten.huber_loss.default(x, x.flip(0), 0, 1e100)
                else:
                    result = torch.ops.aten.huber_loss_backward.default(
                        x.roll(1), x, x.flip(0), 0, 0.7
                    )
                    small = torch.ops.aten.huber_loss_backward.default(
                        x.roll(1), x, x.flip(0), 0, tiny
                    )
                    large = torch.ops.aten.huber_loss_backward.default(
                        x.roll(1), x, x.flip(0), 0, 1e100
                    )
                return (
                    result,
                    result.double(),
                    small,
                    small.double(),
                    large,
                    large.double(),
                )
            return result, result.double()

        self._check_bitwise(fn, x, numerics=numerics, upcast=upcast)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("op_name", ("floor_divide", "remainder", "true_divide"))
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("scalar", (False, True))
    @parametrize("upcast", (False, True))
    def test_quotient_specials(self, device, dtype, op_name, numerics, scalar, upcast):
        values = _min_max_specials(dtype, device)
        a = values.repeat_interleave(values.numel())
        b = 0.7 if scalar else values.repeat(values.numel())

        def fn(a, b):
            result = getattr(torch, op_name)(a, b)
            return result, result.double()

        self._check_bitwise(fn, a, b, numerics=numerics, upcast=upcast)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("op_name", ("sub", "rsub", "logaddexp2", "ldexp", "mul"))
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_arithmetic_opmath(self, device, dtype, op_name, numerics, upcast):
        x = torch.linspace(-3, 3, 128, device=device, dtype=dtype)
        y = torch.linspace(2, -2, 128, device=device, dtype=dtype)

        def fn(x, y):
            if op_name in ("sub", "rsub"):
                result = getattr(torch, op_name)(x, y, alpha=0.7)
            elif op_name == "mul":
                result = x * 0.6931471805599453
            else:
                result = getattr(torch, op_name)(x, y)
            return result, result.double()

        actual, code = run_and_get_code(
            torch.compile(
                fn,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
            y,
        )
        self.assertIn("@triton.jit", "\n".join(code))
        self._assert_bitwise_equal(actual, fn(x, y))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("op_name", ("addcmul", "addcdiv"))
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_addc_options_gradients(self, device, dtype, op_name, numerics):
        inputs = [
            torch.randn(512, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        grad = torch.randn_like(inputs[0])
        op = getattr(torch, op_name)

        def fn(a, b, c, value):
            if isinstance(value, torch.Tensor):
                value = value.item()
            return op(a, b, c, value=value)

        compiled = torch.compile(fn, fullgraph=True, options={"numerics": numerics})
        for value in (0.7, 0.6, -0.7, 1.0):
            expected = fn(*inputs, value)
            actual, codes = run_and_get_code(compiled, *inputs, value)
            if value == 0.7:
                self.assertIn("@triton.jit", "\n".join(codes))
            expected_grad = torch.autograd.grad(expected, inputs, grad)
            actual_grad = torch.autograd.grad(actual, inputs, grad)
            self._assert_bitwise_equal((actual, actual_grad), (expected, expected_grad))

        if dtype == torch.float64:
            values = _min_max_specials(dtype, device)
            triples = torch.cartesian_prod(values, values, values).t().contiguous()
            # Put every triple in both full 1024-element CUDA blocks and the tail.
            aligned = tuple(torch.cat((t.repeat(3)[:2048], t)) for t in triples)
            misaligned = tuple(
                torch.empty(t.numel() + 1, device=device, dtype=dtype)[1:]
                for t in aligned
            )
            for out, inp in zip(misaligned, aligned):
                out.copy_(inp)
            layouts = (
                aligned,
                tuple(t.reshape(3, -1).t() for t in aligned),
                misaligned,
                tuple(torch.stack((t, t), dim=1)[:, 0] for t in aligned),
            )
            scalars = torch.cat(
                (
                    torch.tensor(
                        [1.0, 0.0, 0.25, float("inf"), -float("inf")], dtype=dtype
                    ),
                    values[-4:].cpu(),
                )
            )
            for inputs in layouts:
                torch._dynamo.reset()
                compiled = torch.compile(
                    fn, fullgraph=True, options={"numerics": numerics}
                )
                for sample, scalar in enumerate(scalars):
                    actual, codes = run_and_get_code(compiled, *inputs, scalar)
                    if sample == 0:
                        self.assertIn("@triton.jit", "\n".join(codes))
                    self._assert_bitwise_equal(actual, fn(*inputs, scalar))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_gaussian_nll_variance_validation(self, device, dtype):
        x = torch.tensor([0.125, 0.25, 0.5], dtype=dtype, device=device)
        target = torch.full_like(x, 0.25)
        var = torch.tensor(
            [-0.0, torch.finfo(dtype).tiny / 2, 1.0],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        def fn(x, target, var):
            return torch.nn.functional.gaussian_nll_loss(
                x, target, var, full=True, eps=0.125, reduction="none"
            )

        def unused_loss(x, target, var):
            fn(x, target, var)
            return x

        original_var = var.detach().clone()
        expected = fn(x, target, var)
        expected_grad = torch.autograd.grad(expected, var, torch.ones_like(expected))[0]
        bad_var = torch.full_like(var, -1, requires_grad=True)
        for partition in (False, True):
            options = {
                "numerics": "strict_pointwise",
                "triton.cudagraphs": True,
                "graph_partition": partition,
            }
            compiled = torch.compile(fn, fullgraph=True, options=options)
            actual, code = run_and_get_code(compiled, x, target, var)
            self._assert_bitwise_equal(actual, expected)
            self._assert_bitwise_equal(
                torch.autograd.grad(actual, var, torch.ones_like(actual))[0],
                expected_grad,
            )
            self._assert_bitwise_equal(var.detach(), original_var)
            with self.assertRaisesRegex(ValueError, "var has negative entry/entries"):
                compiled(x, target, bad_var)
            self.assertIn("@triton.jit", "\n".join(code))

            compiled_unused = torch.compile(
                unused_loss, fullgraph=True, options=options
            )
            self._assert_bitwise_equal(compiled_unused(x, target, var), x)
            with self.assertRaisesRegex(ValueError, "var has negative entry/entries"):
                compiled_unused(x, target, bad_var)

        def summed_loss(var):
            return fn(x, target, var).sum()

        grad = torch.func.grad(summed_loss)
        expected_func_grad = grad(var)
        compositions = (
            torch.compile(grad, backend="eager", fullgraph=True),
            torch.func.grad(
                torch.compile(summed_loss, backend="eager", fullgraph=True)
            ),
        )
        for compiled_grad in compositions:
            self._assert_bitwise_equal(compiled_grad(var), expected_func_grad)
            with self.assertRaisesRegex(ValueError, "var has negative entry/entries"):
                compiled_grad(bad_var)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_cuda_scalar_opmath(self, device, dtype, numerics):
        x = torch.cat(
            (
                torch.tensor(
                    [1.0078125, 2.599609375, 1.458984375], dtype=dtype, device=device
                ),
                _min_max_specials(dtype, device),
            )
        )
        scalars = (
            0.9189385332046727,
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            -float("nan"),
        )

        def fn(x):
            arithmetic = tuple(
                op(a, b, alpha=alpha)
                for scalar in scalars
                for op in (torch.add, torch.sub)
                for a, b in ((x, scalar), (scalar, x))
                for alpha in (1.0, 0.25)
            )
            products = tuple(
                a * b for scalar in scalars for a, b in ((x, scalar), (scalar, x))
            )
            return *arithmetic, *products

        for upcast in (False, True):
            actual, code = run_and_get_code(
                torch.compile(
                    fn,
                    fullgraph=True,
                    options={
                        "numerics": numerics,
                        "triton.codegen_upcast_to_fp32": upcast,
                    },
                ),
                x,
            )
            expected = fn(x)
            self.assertEqual(tuple(t.dtype for t in actual), (dtype,) * len(expected))
            self._assert_bitwise_equal(actual, expected)
            self.assertIn("@triton.jit", "\n".join(code))

    @dtypes(
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_negative_input_storage(self, device, dtype, numerics):
        """Check negative storage and queries throughout kernel construction."""
        from unittest.mock import patch

        def fn(x, y):
            old = x.resolve_neg()
            padded = tuple(
                torch.nn.functional.pad(x, (1, 1), value=value)
                for value in (2, 0.0, -0.0)
            )
            x.add_(1)
            return old, x, x + y, x.view_as(x), *padded

        values = _pointwise_edge_values(dtype).to(device)
        values = torch.cat((values, values[:1])).repeat(32)
        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={"numerics": numerics, "triton.cudagraphs": True},
        )
        for call_index, offset in enumerate((0, 1, 0)):
            base = torch.empty(values.numel() + offset, device=device, dtype=dtype)[
                offset:
            ]
            base.copy_(values)
            expected_base = base.clone()
            y = torch.full_like(base, 2)
            x = base._neg_view()
            expected = fn(expected_base._neg_view(), y)
            if call_index == 0:
                with patch(
                    "torch.accelerator.max_memory_allocated",
                    side_effect=torch.accelerator.memory_allocated,
                ):
                    actual, code = run_and_get_code(compiled, x, y)
                self.assertIn("@triton.jit", "\n".join(code))
            else:
                actual = compiled(x, y)
            self._assert_bitwise_equal(actual, expected)
            self._assert_bitwise_equal(base, expected_base)
            self.assertIs(actual[1], x)
            self.assertTrue(torch._C._is_alias_of(actual[3], x))
            self.assertEqual(
                [v.is_neg() for v in actual], [v.is_neg() for v in expected]
            )

        if dtype.is_floating_point:

            def read(x):
                return x.clone() + x.resolve_neg() + x * 2

            base = torch.randn(512, device=device, dtype=dtype, requires_grad=True)
            x = base._neg_view()
            grad = torch.randn_like(x)
            expected = read(x)
            actual = torch.compile(
                read, fullgraph=True, options={"numerics": numerics}
            )(x)
            self._assert_bitwise_equal(actual, expected)
            self._assert_bitwise_equal(
                torch.autograd.grad(actual, x, grad),
                torch.autograd.grad(expected, x, grad),
            )

            def local_mutation(a):
                allocation = a.clone()
                view = allocation._neg_view()
                old = view.clone()
                with torch.no_grad():
                    view.mul_(2)
                return old * old + view * view, allocation

            base = torch.tensor(
                [-0.0, 0.0, 1.0, -1.0], dtype=dtype, device=device, requires_grad=True
            )
            expected_base = base.detach().clone().requires_grad_()
            upstream = torch.full_like(base, 2.25)
            actual = torch.compile(
                local_mutation, fullgraph=True, options={"numerics": numerics}
            )(base)
            expected = local_mutation(expected_base)
            self._assert_optional_output_gradients(
                actual, expected, base, expected_base, upstream
            )
            allocation_grad = base.detach().clone()
            original_grad = allocation_grad.clone()
            self._assert_bitwise_equal(
                torch.autograd.grad(actual[1], base, allocation_grad)[0],
                torch.autograd.grad(expected[1], expected_base, allocation_grad)[0],
            )
            self._assert_bitwise_equal(base, expected_base)
            self._assert_bitwise_equal(allocation_grad, original_grad)

            base = base.detach().clone()
            base[2] = float("nan")
            base.requires_grad_()
            expected_base = base.detach().clone().requires_grad_()

            def independent_outputs(a, b, unbind):
                if unbind:
                    parts = b.unbind()
                    return (
                        a.clone()._neg_view(),
                        a.sigmoid() + parts[0].clone(),
                        parts[1].clone(),
                    )
                view = a.clone()._neg_view()
                # GELU backward includes registered prims in the cached recipe.
                # Repeated sin factors need distinct native owners. Power-zero
                # derivatives may be saved forward and still need absence pruning.
                if dtype == torch.float64:
                    return view**0, view**0, torch.nn.functional.gelu(b)
                call = torch.sin if dtype == torch.float32 else torch.clone
                return call(view), call(view), torch.nn.functional.gelu(b)

            from torch._inductor.utils import fresh_cache

            with (
                fresh_cache(),
                config.patch(fx_graph_cache=True, force_disable_caches=False),
                torch._functorch.config.patch(
                    enable_autograd_cache=True, enable_remote_autograd_cache=False
                ),
            ):
                backward_compiler = torch.compile(backend="eager", fullgraph=True)
                optional_upstream = torch.stack((upstream * 0, upstream * 0), dim=-1)[
                    :, 0
                ]._neg_view()
                for unbind, cache_hit in itertools.product(
                    (False, True), (False, True)
                ):
                    torch._dynamo.reset()
                    hits = counters["aot_autograd"]["autograd_cache_hit"]
                    b = base.detach().clone()
                    if unbind:
                        b = b.repeat(2, 1)
                    b.requires_grad_()
                    expected_b = b.detach().clone().requires_grad_()
                    actual = torch.compile(
                        independent_outputs,
                        fullgraph=True,
                        options={"numerics": numerics},
                    )(base, b, unbind)
                    expected = independent_outputs(expected_base, expected_b, unbind)
                    self._assert_bitwise_equal(actual, expected)
                    weights = tuple(torch.full_like(t, 2.25) for t in actual)
                    self._assert_bitwise_equal(
                        torch.autograd.grad(
                            actual, (base, b), weights, retain_graph=True
                        ),
                        torch.autograd.grad(
                            expected,
                            (expected_base, expected_b),
                            weights,
                            retain_graph=True,
                        ),
                    )
                    # The active unbind needs a local sibling zero while the
                    # shared sigmoid and independent clone keep absent gradients.
                    self._assert_optional_output_gradients(
                        actual,
                        expected,
                        (base, b),
                        (expected_base, expected_b),
                        optional_upstream,
                        selected=2,
                        backward_compiler=backward_compiler if cache_hit else None,
                    )
                    # Selecting only the clone leaves the whole multi-output
                    # unbind group inactive, so its input gradient must be None.
                    self._assert_optional_output_gradients(
                        actual,
                        expected,
                        (base, b),
                        (expected_base, expected_b),
                        optional_upstream,
                        backward_compiler=backward_compiler if cache_hit else None,
                    )
                    if cache_hit:
                        self.assertGreater(
                            counters["aot_autograd"]["autograd_cache_hit"], hits
                        )
            self._assert_bitwise_equal((base, b), (expected_base, expected_b))
            self._assert_bitwise_equal(upstream, torch.full_like(base, 2.25))

        if dtype == torch.float16:
            # Templates query negative inputs before the scheduler is initialized.
            a = torch.ones((128, 128), device=device, dtype=dtype)
            compiled = torch.compile(
                torch.mm,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                },
            )
            actual, codes = run_and_get_code(compiled, a, a.clone())
            self.assertIn("tl.dot(", "\n".join(codes))
            self._assert_bitwise_equal(actual, a @ a)

        if dtype == torch.float32:
            from torch._inductor.compile_fx import compile_fx
            from torch._inductor.lowering import fallback_handler, lowerings

            # Fresh fallback storage must not be reused with its lazy neg flag.
            with torch.library._scoped_library(
                "strict_numerics_test", "FRAGMENT"
            ) as lib:
                lib.define("fresh_neg(Tensor x) -> Tensor")
                lib.impl(
                    "fresh_neg",
                    lambda x: x.clone()._neg_view(),
                    "CompositeExplicitAutograd",
                )
                torch.library.register_fake(
                    "strict_numerics_test::fresh_neg",
                    lambda x: torch.empty_like(x)._neg_view(),
                    lib=lib,
                )

                def fresh_fallback(x):
                    return -torch.ops.strict_numerics_test.fresh_neg(x)

                x = torch.arange(1, 513, device=device, dtype=dtype)
                actual = torch.compile(
                    fresh_fallback, fullgraph=True, options={"numerics": numerics}
                )(x)
                expected = fresh_fallback(x)
                self._assert_bitwise_equal(actual, expected)
                self.assertEqual(actual.is_neg(), expected.is_neg())

            def fallback_views(x):
                a, b = torch.ops.aten.split.Tensor(x._neg_view(), 512)
                return a, b, a + 1, b + 2

            def backend(gm, args):
                return compile_fx(gm, args, decompositions={})

            forced = {
                op: fallback_handler(op, add_to_fallback_set=False)
                for op in (
                    torch.ops.aten._neg_view.default,
                    torch.ops.aten.split.Tensor,
                )
            }
            with (
                patch.dict(lowerings, forced),
                config.patch(
                    {
                        "numerics": numerics,
                        "fx_graph_cache": False,
                        "triton.cudagraphs": True,
                    }
                ),
            ):
                compiled = torch.compile(
                    fallback_views, backend=backend, fullgraph=True
                )
                for offset in (0, 1, 0):
                    x = torch.arange(1024 + offset, device=device, dtype=dtype)[offset:]
                    expected = fallback_views(x)
                    actual = compiled(x)
                    self._assert_bitwise_equal(actual, expected)
                    self.assertEqual(
                        [t.is_neg() for t in actual], [t.is_neg() for t in expected]
                    )
                    for tensor in actual[:2]:
                        self.assertTrue(torch._C._is_alias_of(tensor, x))

    @dtypes(torch.float32, torch.float64)
    @parametrize("axis", (None, 1))
    def test_sum_fallback_preserves_eager_input_layout(self, device, dtype, axis):
        a = torch.randn(301, 64, device=device, dtype=dtype).t()
        b = torch.randn(64, 301, device=device, dtype=dtype)

        def fn(a, b):
            return (a + b).sum(axis)

        compiled = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})
        actual, codes = run_and_get_code(compiled, a, b)
        self.assertIn("@triton.jit", "\n".join(codes))
        self.assertTrue(_outputs_equal(actual, fn(a, b)))

    @dtypes(torch.float32, torch.float64)
    def test_empty_reshape_preserves_dtype_view_strides(self, device, dtype):
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        def fn(x):
            return (x + 1).reshape(0, 10, 6).view(complex_dtype)

        x = torch.empty(0, 10, 3, 2, dtype=dtype, device=device)
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected, exact_stride=True)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("cudagraphs", (False, True))
    @parametrize("input_dtype", (torch.float64, torch.int64, torch.float32))
    def test_mul_input_provenance(
        self, device, dtype, numerics, cudagraphs, input_dtype
    ):
        values = _min_max_specials(dtype, device)
        shared = torch.empty(2052, device=device, dtype=dtype)
        shared[:1026] = values[4]
        shared[1026:] = values[1]
        a, b = shared[:1025].view(input_dtype), shared[1026:2051].view(input_dtype)

        def fn(a, b):
            return a.view(dtype) * b.view(dtype)

        counter = CompileCounterWithBackend("inductor")
        with config.patch({"numerics": numerics, "triton.cudagraphs": cudagraphs}):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            actual, codes = run_and_get_code(compiled, a, b)
            self.assertIn("@triton.jit", "\n".join(codes))
            self.assertTrue(_outputs_equal(actual, fn(a, b)))
            del actual
            # Read-only input aliasing and pointer alignment can change without guards.
            for left, right in ((0, 1), (0, 0), (1, 1), (0, 0), (1, 1)):
                a = values[4:5].repeat(1025 + left)[left:].view(input_dtype)
                b = values[1:2].repeat(1025 + right)[right:].view(input_dtype)
                torch.compiler.cudagraph_mark_step_begin()
                actual = compiled(a, b)
                self.assertTrue(_outputs_equal(actual, fn(a, b)))
                del actual
        self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float32)
    @parametrize("numerics", ("default", "strict_pointwise", "strict"))
    def test_dead_local_mutation(self, device, dtype, numerics):
        from torch._inductor.compile_fx import compile_fx_inner
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x):
            y = x @ x
            y.add_(1)
            return (x * 2,)

        x = torch.randn(8, 8, device=device, dtype=dtype)
        gm = make_fx(fn)(x)
        with config.patch(numerics=numerics):
            compiled, codes = run_and_get_code(compile_fx_inner, gm, [x])
        self._assert_bitwise_equal(compiled([x]), fn(x))
        self.assertNotIn("extern_kernels.mm(", "\n".join(codes))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "hop", ("cond", "nested_cond", "invoke", "while_loop", "aliased_clone")
    )
    @parametrize("partition", (False, True))
    def test_mul_subgraph_input_provenance(
        self, device, dtype, numerics, hop, partition
    ):
        values = _min_max_specials(dtype, device)
        skips_before = counters["inductor"]["cudagraph_skips"]

        @torch.compiler.nested_compile_region
        def region(a, b):
            return a * b

        def run_loop(a, b):
            _, result, _ = torch.while_loop(
                lambda i, a, b: i < 1,
                lambda i, a, b: (i + 1, a * b, b + 0),
                (torch.zeros((), device=a.device, dtype=torch.int64), a, b),
            )
            return result

        def fn(p, a, b):
            if hop == "invoke":
                return region(a, b), region(b, a)
            if hop == "aliased_clone":
                computed = a + 0
                return (
                    run_loop(a.clone()[:1025], a.clone()[1025:]),
                    run_loop(computed.clone()[:1025], computed.clone()[1025:]),
                )
            if hop == "while_loop":
                return run_loop(a, b)
            if hop == "nested_cond":

                def true_fn(a, b):
                    return torch.cond(
                        p, lambda a, b: a * b, lambda a, b: a * b + 2, (a, b)
                    )
            else:

                def true_fn(a, b):
                    return a * b

            return torch.cond(p, true_fn, lambda a, b: a * b + 1, (a, b))

        counter = CompileCounterWithBackend("inductor")
        p = torch.tensor(True, device=device)
        with config.patch(
            {
                "numerics": numerics,
                "triton.cudagraphs": True,
                "graph_partition": partition,
            }
        ):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            for offset in (0, 1, 0):
                a = values[4:5].repeat(1025 + offset)[offset:]
                b = values[1:2].repeat(1025)
                if hop == "aliased_clone":
                    a = torch.cat((a, b))
                torch.compiler.cudagraph_mark_step_begin()
                actual = compiled(p, a, b)
                if hop == "aliased_clone":
                    expected = fn(p, a, b)
                else:
                    expected = (a * b, b * a) if hop == "invoke" else a * b
                self.assertTrue(_outputs_equal(actual, expected))
                del actual
        self.assertEqual(counter.frame_count, 1)

        if hop in ("while_loop", "aliased_clone") and not partition:
            self.assertGreater(counters["inductor"]["cudagraph_skips"], skips_before)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "producer",
        (
            "inplace",
            "chain_view",
            "dtype_view",
            "fresh",
            "copied_fresh",
            "byte_view",
            "float32_inplace",
            "complex64_inplace",
            "out",
        ),
    )
    def test_mul_nan_payload_mutation_provenance(
        self, device, dtype, numerics, producer
    ):
        values = _min_max_specials(dtype, device)

        def fn(a, b, dst=None):
            if producer == "out":
                if typed_root:
                    dst.add_(0)
                    a, dst = a.view(dtype), dst.view(dtype)
                torch.add(a, 0, out=dst)
                # The overwritten result may disappear; the destination view
                # must still refer to the original native storage.
                torch.add(a.reshape(205, -1), 0, out=dst.reshape(205, -1))
                a = dst.view(dtype)
            elif producer == "fresh":
                a = a + 0
            elif producer == "byte_view":
                a = a[3:].view(dtype)
            elif producer == "copied_fresh":
                result = a + 0
                a.copy_(result)
                a = result
            elif producer in ("float32_inplace", "complex64_inplace"):
                a.add_(0)
                a = a.view(dtype)
            else:
                a.view(dtype).add_(0)
                if producer == "inplace":
                    if partial_write:
                        a[:1025].add_(0)
                    else:
                        a.add_(0)
                elif producer == "chain_view":
                    a, b = a.reshape(205, 5), b.reshape(205, 5)
                    a.sub_(0)
                elif producer == "dtype_view":
                    a = a.view(torch.int64).view(dtype)
                    a.add_(0)
            return (a[:1025] if partial_write else a) * b

        def operands(offset):
            if producer == "out":
                dst = torch.zeros(1025 + offset, device=device, dtype=dtype)[
                    offset:
                ].view(out_dtype)
                a = values[7:8].repeat(1025).view(out_dtype)
                if writer_dtype != out_dtype:
                    a = (
                        torch.full(
                            (1025,), 0x7FC00011, dtype=torch.int32, device=device
                        ).view(torch.float32)
                        if writer_dtype == torch.float32
                        else a.to(writer_dtype)
                    )
                return (
                    a,
                    values[6:7].repeat(1025),
                    dst,
                )
            if producer == "byte_view":
                a = (
                    values[7:8]
                    .repeat(1027 + offset)
                    .view(torch.float32)[1 + 2 * offset :]
                )
                return a, values[6:7].repeat(1025)
            count = 2050 if partial_write else 1025
            storage = values[7:8].repeat(count * input_stride + offset)
            if input_stride == 2:
                storage[offset + 1 :: 2] = 1
            a = storage[offset::input_stride]
            if producer == "float32_inplace":
                a = a.view(torch.float32)
            elif producer == "complex64_inplace":
                a = a.view(torch.complex64)
            b = values[6:7].repeat(1025).requires_grad_(partial_write)
            return a, b

        for out_dtype, writer_dtype, typed_root, input_stride, partial_write in (
            (
                (dtype, dtype, False, 1, False),
                (torch.int64, torch.int64, False, 1, False),
                (torch.float32, torch.float32, True, 1, False),
                (dtype, torch.float32, False, 1, False),
                (torch.float32, dtype, False, 1, False),
            )
            if producer == "out"
            else tuple(
                (dtype, dtype, False, stride, partial)
                for stride in (
                    (1, 2)
                    if producer in ("inplace", "chain_view", "dtype_view")
                    else (1,)
                )
                for partial in ((False, True) if producer == "inplace" else (False,))
            )
        ):
            counter = CompileCounterWithBackend("inductor")
            with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
                compiled = torch.compile(fn, fullgraph=True, backend=counter)
                for i, offset in enumerate((0, 1, 0, 1)):
                    eager_args, actual_args = operands(offset), operands(offset)
                    expected = fn(*eager_args)
                    if i == 0:
                        actual, codes = run_and_get_code(compiled, *actual_args)
                        self.assertIn("@triton.jit", "\n".join(codes))
                    else:
                        actual = compiled(*actual_args)
                    self._assert_bitwise_equal(actual, expected)
                    if partial_write:
                        grad = values[6:7].repeat(1025)
                        expected_grad = torch.autograd.grad(
                            expected, eager_args[1], grad
                        )[0]
                        actual_grad = torch.autograd.grad(actual, actual_args[1], grad)[
                            0
                        ]
                        self._assert_bitwise_equal(
                            actual_grad,
                            expected_grad,
                        )
                        self._assert_bitwise_equal(
                            actual_args[0],
                            eager_args[0],
                        )
                    if input_stride == 2:
                        self._assert_bitwise_equal(
                            actual_args[0]._base,
                            eager_args[0]._base,
                        )
                    if producer == "out":
                        for index in (0, 2):
                            self._assert_bitwise_equal(
                                actual_args[index],
                                eager_args[index],
                            )
                if producer == "out" and out_dtype == writer_dtype == dtype:

                    def prior_reader(a, dst):
                        old = dst + 1
                        torch.add(a, 1, out=dst)
                        return old * dst

                    a = torch.ones(1025, device=device, dtype=dtype)
                    dst = torch.ones_like(a)
                    actual = torch.compile(prior_reader, fullgraph=True)(a, dst)
                    self.assertEqual(actual, torch.full_like(actual, 4))
                    self.assertEqual(dst, torch.full_like(dst, 2))
                    self.assertEqual(a, torch.ones_like(a))
            self.assertEqual(counter.frame_count, 1)

        if producer == "inplace":

            def fresh_and_mutated(a, b, c):
                fresh = a + 0
                a.add_(0)
                result = a * b
                a.add_(1)
                return fresh * b, result, c.sin()

            counter = CompileCounterWithBackend("inductor")
            with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
                compiled = torch.compile(
                    fresh_and_mutated, fullgraph=True, backend=counter
                )
                for offset in (0, 1, 0, 1):

                    def inputs():
                        a = values[7:8].repeat(1025 + offset)[offset:]
                        b = values[6:7].repeat(1025)
                        c = torch.tensor(
                            [0.25, -0.5, 0.75],
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                        )
                        return a, b, c

                    eager_args, actual_args = inputs(), inputs()
                    expected = fresh_and_mutated(*eager_args)
                    actual = compiled(*actual_args)
                    for expected_value, actual_value in zip(expected, actual):
                        self._assert_bitwise_equal(
                            actual_value,
                            expected_value,
                        )
                    expected_grad = torch.autograd.grad(
                        expected[2], eager_args[2], torch.ones_like(eager_args[2])
                    )[0]
                    actual_grad = torch.autograd.grad(
                        actual[2], actual_args[2], torch.ones_like(actual_args[2])
                    )[0]
                    self._assert_bitwise_equal(actual_grad, expected_grad)
                    self._assert_bitwise_equal(
                        actual_args[0]._base,
                        eager_args[0]._base,
                    )
            self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float64)
    @parametrize("option", ("strict_pointwise", "numerics"))
    @parametrize("mutation_first", (False, True))
    def test_mul_nan_payload_mutation_cache(
        self, device, dtype, option, mutation_first
    ):
        def copied_fresh(a, b):
            result = a + 0
            a.copy_(result)
            return result * b

        def mutate(a, b):
            a.add_(0)
            return a * b

        def local(a, b):
            z = a + 0
            z.add_(0)
            return z * b, a

        def caller(a, b):
            z = a + 0
            a.add_(0)
            return a * b, z

        values = _min_max_specials(dtype, device)
        functions = (mutate, copied_fresh) if mutation_first else (copied_fresh, mutate)
        options = {
            option: True if option == "strict_pointwise" else "strict_pointwise",
            "triton.cudagraphs": False,
        }
        with config.patch(
            force_disable_caches=False,
            fx_graph_cache=True,
            fx_graph_remote_cache=False,
        ):
            for index, fn in enumerate((*functions, functions[0])):
                torch._dynamo.reset()
                a = values[7:8].repeat(1026)[1:]
                b = values[6:7].repeat(1025)
                eager_a = values[7:8].repeat(1026)[1:]
                expected = fn(eager_a, b)
                hits = counters["inductor"]["fxgraph_cache_hit"]
                misses = counters["inductor"]["fxgraph_cache_miss"]
                actual = torch.compile(fn, fullgraph=True, options=options)(a, b)
                self._assert_bitwise_equal(actual, expected)
                self._assert_bitwise_equal(a, eager_a)
                self.assertEqual(
                    counters["inductor"]["fxgraph_cache_miss"], misses + (index < 2)
                )
                self.assertEqual(
                    counters["inductor"]["fxgraph_cache_hit"], hits + (index == 2)
                )
            functions = (caller, local) if mutation_first else (local, caller)
            for fn in functions:
                torch._dynamo.reset()
                a = torch.arange(1, 1026, device=device, dtype=dtype)
                b = torch.full_like(a, 2).requires_grad_()
                eager_a = a.clone()
                eager_b = b.detach().clone().requires_grad_()
                expected = fn(eager_a, eager_b)
                misses = counters["inductor"]["fxgraph_cache_miss"]
                actual = torch.compile(fn, fullgraph=True, options=options)(a, b)
                self.assertEqual(actual, expected)
                grad = torch.full_like(a, 3)
                expected_grad = torch.autograd.grad(expected[0], eager_b, grad)[0]
                actual_grad = torch.autograd.grad(actual[0], b, grad)[0]
                self.assertEqual(actual_grad, expected_grad)
                self.assertEqual(a, eager_a)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], misses + 2)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("producer", ("clone", "mutate_clone", "slice"))
    def test_mul_clone_mutation_backward(self, device, dtype, numerics, producer):
        from torch._inductor.graph import GraphLowering

        values = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)

        def fn(a, b):
            if producer == "slice":
                return a[1:].mul_(b[1:])
            if producer == "mutate_clone":
                a.add_(0)
            return a.clone().mul_(b)

        def operands(offset):
            length = 1026 if producer == "slice" else 1025
            return (
                values[0:1].repeat(length + offset)[offset:],
                values[1:2].repeat(length).requires_grad_(),
            )

        counter = CompileCounterWithBackend("inductor")
        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            for iteration, offset in enumerate((0, 1, 0, 1)):
                eager_args, args = operands(offset), operands(offset)
                storage = args[0].untyped_storage()
                eager = fn(*eager_args)
                if iteration == 0:
                    actual, codes = run_and_get_code(compiled, *args)
                    self.assertIn("@triton.jit", "\n".join(codes))
                else:
                    actual = compiled(*args)
                self._assert_bitwise_equal(actual, eager)
                self._assert_bitwise_equal(args[0], eager_args[0])
                self.assertEqual(args[0].untyped_storage(), storage)
                self.assertEqual(
                    actual.untyped_storage() == storage, producer == "slice"
                )
                before_backward = args[0].view(torch.int64).clone()
                grad_output = values[2:3].repeat(1025)
                expected_grad = torch.autograd.grad(eager, eager_args[1], grad_output)[
                    0
                ]
                codes = []
                with mock.patch.object(GraphLowering, "save_output_code", codes.append):
                    grads = torch.autograd.grad(actual, args[1], grad_output)
                if iteration == 0:
                    self.assertIn("@triton.jit", "\n".join(codes))
                self._assert_bitwise_equal(grads[0], expected_grad)
                self.assertEqual(args[0].view(torch.int64), before_backward)
        self.assertEqual(counter.frame_count, 1)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("cpu_scalar", (False, True))
    @parametrize("producer", ("inplace", "out"))
    def test_mul_nan_payload_mutation_backward(
        self, device, dtype, numerics, cpu_scalar, producer
    ):
        values = torch.tensor(
            [
                0x7FF8000000000011,
                0x7FF8000000000001,
                0x7FF8000000000033,
                0x7FF8000000000022,
            ],
            dtype=torch.int64,
            device=device,
        ).view(dtype)

        def fn(a, b, dst=None, c=None):
            if producer == "out":
                if typed_root:
                    dst.add_(0)
                    dst = dst.view(dtype)
                torch.add(a, 0, out=dst)
                torch.add(c.reshape(205, -1), 0, out=dst.reshape(205, -1))
                a = dst.view(dtype)
            else:
                if native_mul:
                    if native_mul == "before":
                        a.add_(0)
                    result = a.mul_(b)
                    if native_mul == "after":
                        a.add_(0)
                    return result
                a.add_(0)
            return a * b

        def operands(offset):
            a = values[0:1].repeat(1025 + offset)[offset:]
            b = (
                torch.tensor(1.0, dtype=dtype, device="cpu", requires_grad=True)
                if cpu_scalar
                else values[1:2].repeat(1025).requires_grad_()
            )
            if producer == "out":

                def source_value(value, payload):
                    if writer_dtype == torch.float32:
                        return torch.full(
                            (1025,), payload, dtype=torch.int32, device=device
                        ).view(torch.float32)
                    result = value.repeat(1025)
                    return (
                        result
                        if backing_dtype == dtype or typed_root
                        else result.view(backing_dtype).to(writer_dtype)
                    )

                return (
                    source_value(values[0:1], 0x7FC00011),
                    b,
                    torch.zeros(1025 + offset, dtype=dtype, device=device)[
                        offset:
                    ].view(backing_dtype),
                    source_value(values[3:4], 0x7FC00022),
                )
            return a, b

        grad_output = values[2:3].repeat(1025)
        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            for backing_dtype, writer_dtype, typed_root, native_mul in (
                (
                    (dtype, dtype, False, False),
                    (torch.float32, dtype, True, False),
                    (dtype, torch.float32, False, False),
                    (torch.float32, dtype, False, False),
                )
                if producer == "out"
                else tuple(
                    (dtype, dtype, False, native)
                    for native in (
                        (False, True)
                        if cpu_scalar
                        else (False, True, "before", "after")
                    )
                )
            ):
                counter = CompileCounterWithBackend("inductor")
                compiled = torch.compile(fn, fullgraph=True, backend=counter)
                frames = None
                for offset in (0, 1, 0, 1):
                    eager_args = operands(offset)
                    eager = fn(*eager_args)
                    expected = torch.autograd.grad(eager, eager_args[1], grad_output)[0]
                    args = operands(offset)
                    output = compiled(*args)
                    self._assert_bitwise_equal(output, eager)
                    actual = torch.autograd.grad(output, args[1], grad_output)[0]
                    self._assert_bitwise_equal(actual, expected)
                    for index in (0, 2) if producer == "out" else (0,):
                        self._assert_bitwise_equal(
                            args[index],
                            eager_args[index],
                            msg=f"{backing_dtype=}, {writer_dtype=}, {typed_root=}, {offset=}, {index=}",
                        )
                    if frames is None:
                        frames = counter.frame_count
                    self.assertEqual(counter.frame_count, frames)
                if not cpu_scalar:
                    self.assertEqual(counter.frame_count, 1)

            def prior_reader(a, b, dst):
                if producer == "out":
                    old = dst + 1
                    torch.add(a, 1, out=dst)
                    return old * dst * b
                old = a + 1
                a.add_(1)
                return old * a * b

            if not cpu_scalar:
                compiled = torch.compile(prior_reader, fullgraph=True)
                a = torch.ones(1025, device=device, dtype=dtype)
                b = torch.ones_like(a, requires_grad=True)
                dst = torch.ones_like(a)
                actual = compiled(a, b, dst)
                self.assertEqual(actual, torch.full_like(actual, 4))
                self.assertEqual(
                    torch.autograd.grad(actual, b, torch.ones_like(actual))[0],
                    torch.full_like(b, 4),
                )

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("index_propagation", (False, True))
    def test_mul_arange(self, device, dtype, numerics, index_propagation):
        def fn(x):
            a = torch.arange(x.numel(), device=x.device, dtype=dtype)
            b = torch.arange(x.numel(), device=x.device, dtype=dtype)
            return a * b

        x = torch.empty(1025, device=device)
        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": numerics,
                "constant_and_index_propagation": index_propagation,
            },
        )
        actual, codes = run_and_get_code(compiled, x)
        self.assertIn("@triton.jit", "\n".join(codes))
        self.assertEqual(actual.view(torch.int64), fn(x).view(torch.int64))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("scalar_left", (False, True))
    @parametrize("scalar_dtype", (torch.float32, torch.float64))
    @parametrize("foreach", (False, True))
    def test_mul_cpu_scalar_nan_payload(
        self, device, dtype, numerics, scalar_left, scalar_dtype, foreach
    ):
        x = _min_max_specials(dtype, device)
        scalars = _min_max_specials(scalar_dtype, "cpu")

        def fn(x, scalar):
            if foreach:
                return (
                    foreach_map(torch.mul, scalar, [x])[0]
                    if scalar_left
                    else foreach_map(torch.mul, [x], scalar)[0]
                )
            return scalar * x if scalar_left else x * scalar

        compiled = torch.compile(fn, fullgraph=True, options={"numerics": numerics})
        for scalar in scalars:
            self._assert_bitwise_equal(compiled(x, scalar), fn(x, scalar))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_acos_nan_gradients(self, device, dtype, numerics, upcast):
        endpoints = torch.tensor([-1, 0, 1], dtype=dtype, device=device)
        values = torch.cat(
            (
                _min_max_specials(dtype, device),
                endpoints,
                torch.nextafter(endpoints, torch.full_like(endpoints, float("inf"))),
                torch.nextafter(endpoints, torch.full_like(endpoints, -float("inf"))),
            )
        )
        upstream = _min_max_specials(dtype, device)
        x = values.repeat_interleave(upstream.numel()).requires_grad_()
        grad = upstream.repeat(values.numel())
        compiled = torch.compile(
            torch.acos,
            fullgraph=True,
            options={"numerics": numerics, "triton.codegen_upcast_to_fp32": upcast},
        )
        expected = torch.autograd.grad(torch.acos(x), x, grad)[0]
        actual = torch.autograd.grad(compiled(x), x, grad)[0]
        self._assert_bitwise_equal(actual, expected)

    @ops(
        [op for op in op_db if op.name in ("abs", "neg", "angle", "frexp", "signbit")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    @parametrize("shared", (False, True))
    def test_unary_nan_payload(self, device, dtype, op, numerics, upcast, shared):
        if dtype in (torch.float16, torch.bfloat16):
            x = _exhaustive_16bit(dtype, device)
        else:
            x = _min_max_specials(dtype, device)
            if dtype == torch.float32:
                x = torch.cat((x, _sampled_fp32(NUM_BITPATTERN_SAMPLES, device)))
            else:
                finfo = torch.finfo(dtype)
                edges = torch.tensor(
                    [finfo.tiny * finfo.eps, finfo.tiny, finfo.max],
                    dtype=dtype,
                    device=device,
                )
                x = torch.cat((x, edges, -edges))

        def fn(x):
            result = op.op(x)
            outputs = result if op.name == "frexp" else (result,)
            if shared:
                return (*outputs, outputs[0].double(), x + 1)
            return outputs

        result, codes = run_and_get_code(
            torch.compile(
                fn,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
        )
        self.assertIn("@triton.jit", "\n".join(codes))
        self._assert_bitwise_equal(result, fn(x))

    @ops(
        [
            op
            for op in op_db
            if op.name
            in ("abs", "neg", "angle", "nn.functional.hardtanh", "special.log_ndtr")
        ],
        allowed_dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_unary_integer(self, device, dtype, op, numerics):
        limits = torch.iinfo(dtype)
        x = torch.tensor([limits.min, 0, 1, limits.max], dtype=dtype, device=device)
        if op.name == "nn.functional.hardtanh":
            if dtype in (torch.uint8, torch.int8):
                x = torch.arange(limits.min, limits.max + 1, device=device).to(dtype)

            def fn(x):
                return (
                    op.op(x),
                    op.op(x, -0.7, 0.7),
                    op.op(
                        x,
                        max(limits.min - 1, torch.iinfo(torch.int64).min),
                        min(limits.max + 1, torch.iinfo(torch.int64).max),
                    ),
                )
        else:
            fn = op.op
            if op.name == "special.log_ndtr":
                x = _pointwise_edge_values(dtype).to(device)
        compiled = torch.compile(fn, fullgraph=True, options={"numerics": numerics})
        self._assert_bitwise_equal(compiled(x), fn(x))
        if op.name == "special.log_ndtr":
            from torch.testing._internal.common_utils import set_default_dtype

            with set_default_dtype(torch.float64):
                self._assert_bitwise_equal(compiled(x), fn(x))

    @ops(
        [
            op
            for op in foreach_binary_op_db
            if op.name in ("_foreach_minimum", "_foreach_maximum")
        ],
        allowed_dtypes=(torch.float16, torch.bfloat16),
    )
    def test_foreach_min_max_opmath(self, device, dtype, op):
        x = _min_max_specials(dtype, device)
        y = torch.ones_like(x)
        compiled = torch.compile(
            op.op, fullgraph=True, options={"numerics": "strict_pointwise"}
        )
        self._assert_bitwise_equal(compiled([x], [y])[0], op.op([x], [y])[0])

    @ops(
        [op for op in op_db if op.name in ("minimum", "maximum", "fmin", "fmax")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_min_max_nan_payload(self, device, dtype, op, numerics, upcast):
        x = _min_max_specials(dtype, device)
        a, b = x[:, None], x[None, :]
        compiled = torch.compile(
            op.op,
            fullgraph=True,
            options={"numerics": numerics, "triton.codegen_upcast_to_fp32": upcast},
        )
        result, codes = run_and_get_code(compiled, a, b)
        self.assertIn("@triton.jit", "\n".join(codes))
        self._assert_bitwise_equal(result, op.op(a, b))

    @ops(
        [op for op in op_db if op.name in ("minimum", "maximum", "fmin", "fmax")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("upcast", (False, True))
    @parametrize("mixed", (False, True))
    @parametrize("scalar_input", (False, True))
    def test_min_max_cpu_scalar(self, device, dtype, op, upcast, mixed, scalar_input):
        scalar_dtype = (
            {
                torch.float16: torch.float32,
                torch.bfloat16: torch.float32,
                torch.float32: torch.float16,
                torch.float64: torch.float32,
            }[dtype]
            if mixed
            else dtype
        )
        x = _min_max_specials(dtype, device)
        if scalar_input:
            x = x[-4]  # Signaling NaN.

        def fn(x, bound):
            return op.op(x, bound), op.op(bound, x)

        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": "strict_pointwise",
                "triton.codegen_upcast_to_fp32": upcast,
            },
        )
        bounds = _min_max_specials(scalar_dtype, "cpu")
        _, codes = run_and_get_code(compiled, x, bounds[0])
        self.assertTrue(codes)
        self.assertNotIn("aten.view.dtype(", "\n".join(codes))
        for bound in bounds:
            actual, expected = compiled(x, bound), fn(x, bound)
            self._assert_bitwise_equal(actual, expected)

    @ops(
        [op for op in op_db if op.name in ("fmin", "fmax")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32),
    )
    @parametrize("upcast", (False, True))
    @parametrize("scalar_input", (False, True))
    def test_min_max_mixed_dtype(self, device, dtype, op, upcast, scalar_input):
        a = _min_max_specials(dtype, device)[:, None]
        b = _min_max_specials(torch.float64, device)
        b = b[-4] if scalar_input else b[None, :]

        def fn(a, b):
            return op.op(a, b), op.op(b, a)

        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": "strict_pointwise",
                "triton.codegen_upcast_to_fp32": upcast,
            },
        )
        self._assert_bitwise_equal(compiled(a, b), fn(a, b))

    @ops(
        [op for op in op_db if op.name in ("minimum", "maximum", "fmin", "fmax")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32),
    )
    @parametrize("compute_dtype", (torch.float32, torch.float64))
    @parametrize("upcast", (False, True))
    @parametrize("shared", (False, True))
    def test_min_max_casts(self, device, dtype, op, compute_dtype, upcast, shared):
        x = _min_max_specials(dtype, device)

        def fn(a, b):
            aa, bb = a.to(compute_dtype), b.to(compute_dtype)
            high = op.op(aa, bb)
            result = high.to(dtype)
            if shared:
                return result, op.op(a, b), high, aa, bb, aa + 1
            return (result,)

        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": "strict_pointwise",
                "triton.codegen_upcast_to_fp32": upcast,
            },
        )
        if shared:
            a, b = x.repeat_interleave(x.numel()), x.repeat(x.numel())
        else:
            a, b = x[:, None], x[None, :]
        result, codes = run_and_get_code(compiled, a, b)
        if shared:
            self.assertEqual(sum(code.count("@triton.jit") for code in codes), 1)
        self._assert_bitwise_equal(result, fn(a, b))

    @ops(
        [op for op in op_db if op.name in ("byte", "char", "short")],
        allowed_dtypes=(
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int64,
        ),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_narrow_integer_cast(self, device, dtype, op, numerics, upcast):
        if dtype in (torch.float16, torch.bfloat16):
            x = _exhaustive_16bit(dtype, device)
        elif dtype == torch.float32:
            x = _sampled_fp32(NUM_BITPATTERN_SAMPLES, device)
        else:
            values: list[int | float] = [
                sign * 2**power + delta
                for power in (7, 8, 15, 16, 31, 32)
                for sign in (-1, 1)
                for delta in (-1, 0, 1)
            ]
            values += [torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max]
            if dtype == torch.float64:
                limit = torch.finfo(dtype).max
                values += [
                    sign * value
                    for value in (0.5, 257.999999999, 2.0**63, 2.0**64, limit)
                    for sign in (-1, 1)
                ]
                values += [float("nan"), float("inf"), -float("inf")]
            x = torch.tensor(values, device=device, dtype=dtype)
        result, codes = run_and_get_code(
            torch.compile(
                op.op,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
        )
        self.assertEqual(result, op.op(x))
        intermediate = "int64" if op.name == "byte" else "int32"
        target = {"byte": "uint8", "char": "int8", "short": "int16"}[op.name]
        marker = f".to(tl.{intermediate}).to(tl.{target})"
        self.assertEqual(marker in "\n".join(codes), dtype.is_floating_point)

    @ops(
        [op for op in op_db if op.name == "sigmoid"],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_sigmoid(self, device, dtype, op, numerics, upcast):
        x = (
            _sampled_fp32(NUM_BITPATTERN_SAMPLES, device)
            if dtype == torch.float32
            else _exhaustive_16bit(dtype, device)
        )
        result, codes = run_and_get_code(
            torch.compile(
                op.op,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
        )
        code = "\n".join(codes)
        self._assert_bitwise_equal(result, op.op(x))
        self.assertIn("libdevice.exp", code)
        self.assertIn("libdevice.rcp_rn", code)
        self.assertNotIn("tl.sigmoid", code)

    @ops(
        [
            op
            for op in op_db
            if op.name
            in (
                "nn.functional.gelu",
                "nn.functional.glu",
                "nn.functional.hardswish",
                "nn.functional.mish",
                "nn.functional.silu",
                "nn.functional.softplus",
                "nn.functional.tanhshrink",
                "sigmoid",
                "tanh",
            )
            and not op.variant_test_name
        ],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_activation_backward(self, device, dtype, op, numerics, upcast):
        bounds = torch.tensor(
            [-3.0, -0.5, -0.0, 0.0, 0.2, 0.5, 3.0, 20.0, 104.0],
            device=device,
            dtype=dtype,
        )
        x = torch.cat(
            (
                bounds,
                torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
                torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
                torch.linspace(-20, 20, 257, device=device, dtype=dtype),
            )
        ).requires_grad_()
        grad = torch.linspace(-1.3, 1.7, x.numel(), device=device, dtype=dtype)
        grad[:2] = torch.tensor([0.0, -0.0], device=device, dtype=dtype)
        if op.name == "nn.functional.glu":
            grad = grad[: x.numel() // 2]
        kwargs = {}
        if op.name == "nn.functional.gelu":
            kwargs = {"approximate": "tanh"}
        elif op.name == "nn.functional.softplus":
            kwargs = {"beta": 3, "threshold": 0.2}

        def fn(x):
            # Share opmath without unused-output tangents changing signed zeros.
            result = (op.op(x, **kwargs), (x.float() * x.float()).detach())
            if op.name == "nn.functional.softplus":
                return (*result, op.op(x, beta=-1).detach())
            return result

        expected = fn(x)
        expected_grad = torch.autograd.grad(expected[0], x, grad)[0]
        result, codes = run_and_get_code(
            torch.compile(
                fn,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
        )
        self.assertIn("@triton.jit", "\n".join(codes))
        result_grads, codes = run_and_get_code(torch.autograd.grad, result[0], x, grad)
        self.assertIn("@triton.jit", "\n".join(codes))
        self._assert_bitwise_equal(result, expected)
        self._assert_bitwise_equal(result_grads[0], expected_grad)

        if dtype == torch.float64 and op.name in (
            "nn.functional.gelu",
            "nn.functional.mish",
        ):
            values = _min_max_specials(dtype, device)
            x = (
                values.repeat((1536 + values.numel() - 1) // values.numel())[:1536]
                .reshape(2, 768)
                .t()
            )
            if op.name == "nn.functional.gelu":
                self._check_bitwise(op.op, x, numerics=numerics, upcast=upcast)
            bits = torch.full(
                (1537,), 0x7FF8000000000002, dtype=torch.int64, device=device
            )
            g = bits[:1536].view(dtype).reshape(2, 768).t()
            offset = bits[1:].view(dtype).reshape(2, 768).t()

            def backward(g, x):
                if op.name == "nn.functional.gelu":
                    return (
                        torch.ops.aten.gelu_backward.default(g, x, approximate="none"),
                        torch.ops.aten.gelu_backward.default(g, x, approximate="tanh"),
                    )
                return (
                    torch.ops.aten.mish_backward.default(g, x),
                    torch.ops.aten.log_sigmoid_backward.default(
                        g, x, torch.empty(0, dtype=dtype, device=device)
                    ),
                )

            for gradient, value in (
                (g, x),
                (offset, x),
                (g.contiguous(), x.contiguous()),
                (g[0, 0].expand_as(x), x),
            ):
                self._check_bitwise(
                    backward, gradient, value, numerics=numerics, upcast=upcast
                )

            def cloned_backward(g, x):
                return backward(g.clone(), x)

            self._check_bitwise(
                cloned_backward, offset, x, numerics=numerics, upcast=upcast
            )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("upcast", (False, True))
    @parametrize("scalar_grad", (False, True))
    def test_tanh_backward_scalar_cast(self, device, dtype, upcast, scalar_grad):
        scalar = torch.tensor(
            1 + torch.finfo(dtype).eps / 2 + 2**-45,
            device=device,
            dtype=torch.float64,
        )
        vector = torch.full((17,), 0 if scalar_grad else 1, device=device, dtype=dtype)
        grad, output = (scalar, vector) if scalar_grad else (vector, scalar)
        fn = torch.ops.aten.tanh_backward.default
        self._check_bitwise(
            fn, grad, output, numerics="strict_pointwise", upcast=upcast
        )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("upcast", (False, True))
    @parametrize("opname", ("gelu", "hardswish", "mish", "silu"))
    def test_activation_backward_mixed_dtype(self, device, dtype, upcast, opname):
        x = torch.tensor(
            [-3.703125, -3.671875, -1.5390625, 0.0, 0.5, 1.5],
            device=device,
            dtype=dtype,
        )
        grad = torch.tensor(-2.7281209403305, device=device, dtype=torch.float64)
        if opname == "mish":
            grad = grad.expand_as(x).clone()
        fn = getattr(torch.ops.aten, f"{opname}_backward").default
        kwargs = {"approximate": "tanh"} if opname == "gelu" else {}
        expected = fn(grad, x, **kwargs)
        result = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": "strict_pointwise",
                "triton.codegen_upcast_to_fp32": upcast,
            },
        )(grad, x, **kwargs)
        self.assertEqual(result.dtype, expected.dtype)
        self._assert_bitwise_equal(result, expected)

    @ops(
        [
            op
            for op in op_db
            if op.name
            in (
                "clamp",
                "heaviside",
                "nn.functional.relu",
                "nn.functional.relu6",
                "nn.functional.hardtanh",
                "nn.functional.threshold",
            )
        ],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_clamp_shared_predicate(self, device, dtype, op, numerics, upcast):
        if (
            op.name == "nn.functional.relu"
            and IS_FBCODE
            and upcast
            and dtype in (torch.float16, torch.bfloat16)
        ):
            # The generated where(x < 0, 0, x) is correct, but fbcode's Triton
            # 3.8 returns +0.0 for relu(-0.0) after the fp32 upcast; OSS Triton
            # 3.9 does not. Remove once fbcode moves to Triton 3.9.
            self.skipTest("fbcode Triton 3.8 drops the sign of relu(-0.0) when upcast")
        if dtype in (torch.float16, torch.bfloat16):
            x = _exhaustive_16bit(dtype, device)
        elif dtype == torch.float32:
            x = _sampled_fp32(NUM_BITPATTERN_SAMPLES, device)
        else:
            bits = torch.tensor(
                [
                    0,
                    1,
                    0x0010000000000000,
                    0x3FF0000000000000,
                    0x7FEFFFFFFFFFFFFF,
                    0x7FF0000000000000,
                    0x7FF0000000000001,
                    0x7FF8123456789ABC,
                    0x7FFFFFFFFFFFFFFF,
                ],
                dtype=torch.int64,
                device=device,
            )
            x = torch.cat((bits, bits | torch.iinfo(torch.int64).min)).view(dtype)

        values = [-1.0, 0.0, 1.0, 6.0]
        if op.name == "nn.functional.threshold":
            values.append(0.7)
        bounds = torch.tensor(values, device=device, dtype=dtype)
        x = torch.cat(
            (
                x,
                bounds,
                torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
                torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
            )
        )

        x.requires_grad_(op.supports_autograd)
        lower = -1 if op.name in ("clamp", "nn.functional.hardtanh") else 0

        def fn(x):
            if op.name == "heaviside":
                return op.op(x, x.flip(0)), x == 0
            if op.name == "nn.functional.threshold":
                return op.op(x, 0.7, -0.0), x <= 0.7
            result = op.op(x, -1, 1) if op.name == "clamp" else op.op(x)
            return result, x < lower

        (result, below_lower), codes = run_and_get_code(
            torch.compile(
                fn,
                fullgraph=True,
                options={
                    "numerics": numerics,
                    "triton.codegen_upcast_to_fp32": upcast,
                },
            ),
            x,
        )
        expected, expected_below_lower = fn(x)
        self._assert_bitwise_equal(result, expected)
        self.assertEqual(below_lower, expected_below_lower)
        self.assertIn("tl.where", "\n".join(codes))
        if op.supports_autograd:
            expected_grad = torch.autograd.grad(expected, x, torch.ones_like(x))[0]
            result_grad = torch.autograd.grad(result, x, torch.ones_like(x))[0]
            self._assert_bitwise_equal(result_grad, expected_grad)

        if op.name == "heaviside":
            self._check_bitwise(
                op.op, torch.zeros_like(x), x, numerics=numerics, upcast=upcast
            )

        if op.name == "nn.functional.threshold":

            def nan_fill(x):
                return op.op(x, 0.7, float("nan")), op.op(x, float("nan"), -0.0)

            self._check_bitwise(nan_fill, x, numerics=numerics, upcast=upcast)

    @config.patch(
        numerics="strict_pointwise", max_fusion_size=1, realize_opcount_threshold=0
    )
    @ops(
        [
            op
            for op in op_db
            if op.name in ("clamp", "nn.functional.hardtanh", "nn.functional.relu6")
        ],
        allowed_dtypes=(torch.float16, torch.bfloat16),
    )
    def test_clamp_materialized_opmath(self, device, dtype, op):
        def fn(x):
            return op.op(x, -1, 1) if op.name == "clamp" else op.op(x)

        x = _exhaustive_16bit(dtype, device)
        metrics.reset()
        result = torch.compile(fn, fullgraph=True)(x)
        self._assert_bitwise_equal(result, fn(x))
        self.assertGreater(metrics.generated_kernel_count, 1)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("bound_kind", ("min", "max", "both", "clamp_min", "clamp_max"))
    def test_clamp_tensor_bounds(self, device, dtype, numerics, bound_kind):
        values = torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")],
            device=device,
            dtype=dtype,
        )
        int_dtype = _BIT_VIEW[dtype]
        nan_bits = values[-1:].view(int_dtype) | 1
        nan_bits = torch.cat((nan_bits, nan_bits + 1))
        nan_bits = torch.cat((nan_bits, nan_bits | torch.iinfo(int_dtype).min))
        values = torch.cat((values, nan_bits.view(dtype)))
        x = values[:, None, None].clone().requires_grad_()
        lo = values[None, :, None].clone().requires_grad_()
        hi = values[None, None, :].clone().requires_grad_()

        def fn(x, lo, hi):
            if bound_kind == "clamp_min":
                result = torch.clamp_min(x, lo)
            elif bound_kind == "clamp_max":
                result = torch.clamp_max(x, hi)
            else:
                result = torch.clamp(
                    x,
                    min=None if bound_kind == "max" else lo,
                    max=None if bound_kind == "min" else hi,
                )
            return result, x < lo, hi < x

        expected, *expected_masks = fn(x, lo, hi)
        result, *masks = torch.compile(
            fn, fullgraph=True, options={"numerics": numerics}
        )(x, lo, hi)
        self._assert_bitwise_equal(result, expected)
        self.assertEqual(masks, expected_masks)
        expected_grads = torch.autograd.grad(
            expected, (x, lo, hi), torch.ones_like(expected), allow_unused=True
        )
        result_grads = torch.autograd.grad(
            result, (x, lo, hi), torch.ones_like(result), allow_unused=True
        )
        for actual, wanted in zip(result_grads, expected_grads):
            if wanted is None:
                self.assertIsNone(actual)
            else:
                self._assert_bitwise_equal(actual, wanted)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("zero_dim", (False, True))
    @parametrize("op", ("clamp", "clamp_min", "clamp_max"))
    def test_clamp_mixed_tensor_dtypes(self, device, dtype, zero_dim, op):
        def fn(x, lo, hi):
            if op == "clamp":
                return torch.clamp(x, lo, hi)
            return getattr(torch, op)(x, hi)

        x = _exhaustive_16bit(dtype, device)
        lo = torch.tensor(-1.0, dtype=torch.float32, device=device)
        midpoint = torch.tensor(
            1.0 + torch.finfo(dtype).eps / 2, dtype=torch.float64, device=device
        )
        above_mid = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
        nan = torch.tensor(0x7FF0000000000001, dtype=torch.int64, device=device)
        negative_nan = nan | torch.iinfo(torch.int64).min
        bounds = (
            above_mid,
            nan.view(torch.float64),
            negative_nan.view(torch.float64),
        )
        if not zero_dim:
            lo = lo[None]
        compiled = torch.compile(
            fn, fullgraph=True, options={"numerics": "strict_pointwise"}
        )
        for hi in bounds:
            if not zero_dim:
                hi = hi[None]
            expected = fn(x, lo, hi)
            result = compiled(x, lo, hi)
            self._assert_bitwise_equal(result, expected)

    @dtypes(torch.int32, torch.int64, torch.uint32, torch.uint64)
    @parametrize("upcast", (False, True))
    def test_clamp_integer_to_bfloat16(self, device, dtype, upcast):
        limits = torch.iinfo(dtype)
        values = [0, 1, limits.max]
        for exponent in range(24, limits.bits - int(dtype.is_signed)):
            for offset in (1, 3):
                midpoint = 2**exponent + offset * 2 ** (exponent - 8)
                values.extend(midpoint + delta for delta in (-1, 0, 1))
        if dtype.is_signed:
            values = [*values, *(-v for v in values), limits.min]
        x = torch.tensor(values, dtype=dtype, device=device)
        lo = torch.tensor([-float("inf")], dtype=torch.bfloat16, device=device)
        hi = -lo

        def fn(x, lo, hi):
            return (
                torch.clamp(x, lo, hi),
                torch.clamp(lo, x, hi),
                torch.clamp(hi, lo, x),
                torch.clamp_min(x, lo),
                torch.clamp_min(lo, x),
                torch.clamp_max(x, hi),
                torch.clamp_max(hi, x),
                x.to(torch.bfloat16),
            )

        actual = torch.compile(
            fn,
            fullgraph=True,
            options={
                "numerics": "strict_pointwise",
                "triton.codegen_upcast_to_fp32": upcast,
            },
        )(x, lo, hi)
        expected = fn(x, lo, hi)
        self._assert_bitwise_equal(actual, expected)

    @dtypes(torch.float32)
    @parametrize("op", ("argmax", "argmin"))
    @parametrize("upcast", (False, True))
    def test_computed_index_to_bfloat16(self, device, dtype, op, upcast):
        def fn(x):
            return getattr(x, op)(-1).to(torch.bfloat16)

        x = torch.randn((16, 16), device=device, dtype=dtype)
        self._check_bitwise(fn, x, numerics="strict_pointwise", upcast=upcast)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("bound_kind", ("min", "max"))
    @parametrize("standalone", (False, True))
    def test_clamp_scalar_single_bound(self, device, dtype, bound_kind, standalone):
        op = getattr(torch, f"clamp_{bound_kind}") if standalone else torch.clamp

        def fn(x, bound):
            return op(x, **{bound_kind: bound})

        x = torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
            device=device,
            dtype=dtype,
        )
        compiled = torch.compile(
            fn, fullgraph=True, options={"numerics": "strict_pointwise"}
        )
        for bound in (-0.0, 0.7, 1e-42, float("nan")):
            self._assert_bitwise_equal(compiled(x, bound), fn(x, bound))

    @torch._dynamo.config.patch(capture_scalar_outputs=True, specialize_float=False)
    @config.patch(
        numerics="strict_pointwise",
        _use_fp64_for_unbacked_floats=False,
        force_disable_caches=False,
    )
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("op", ("clamp_min", "clamp_max"))
    @parametrize("bound_source", ("python", "cpu", "cuda"))
    def test_clamp_single_nan_bound(self, device, dtype, op, bound_source):
        def fn(x, bound):
            return getattr(torch, op)(
                x, bound if bound_source == "python" else bound.item()
            )

        x = torch.tensor(
            [-1.0, -0.0, 0.0, 1.0, float("nan")], device=device, dtype=dtype
        )
        bits = torch.tensor(
            [
                0x3FE6666666666666,
                0x7FF0000000000001,
                0x7FF8000000000000,
                0x7FFABCDE12345678,
            ],
            dtype=torch.int64,
            device=device if bound_source == "cuda" else "cpu",
        )
        bits = torch.cat((bits, bits | torch.iinfo(torch.int64).min))
        bounds = bits.view(torch.float64)
        compiled = torch.compile(fn, fullgraph=True)
        for bound in bounds:
            if bound_source == "python":
                bound = bound.item()
            self._assert_bitwise_equal(compiled(x, bound), fn(x, bound))

        if bound_source == "python":
            hits = counters["aot_autograd"]["autograd_cache_hit"]
            torch._dynamo.reset()
            bound = bounds[1].item()
            self._assert_bitwise_equal(compiled(x, bound), fn(x, bound))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], hits + 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @config.patch(numerics="strict_pointwise", _use_fp64_for_unbacked_floats=False)
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("op", ("clamp_min", "clamp_max"))
    @parametrize("layout", ("transpose", "channels_last"))
    def test_clamp_nan_fill_layout(self, device, dtype, op, layout):
        if layout == "transpose":
            x = torch.arange(6, device=device, dtype=dtype).reshape(2, 3).t()
            permutation = (1, 0)
        else:
            x = (
                torch.arange(120, device=device, dtype=dtype)
                .reshape(2, 3, 4, 5)
                .contiguous(memory_format=torch.channels_last)
            )
            permutation = (0, 2, 3, 1)

        def fn(x, bound):
            value = float("nan") if bound is None else bound.item()
            result = getattr(torch, op)(x, value)
            return result, result.permute(permutation).view(-1)

        nan = torch.tensor(0x7FFABCDE12345678, dtype=torch.int64).view(torch.float64)
        compiled = torch.compile(fn, fullgraph=True)
        for bound in (None, torch.tensor(0.7, dtype=torch.float64), nan):
            expected = fn(x, bound)
            actual = compiled(x, bound)
            self.assertEqual(actual[0].stride(), expected[0].stride())
            self._assert_bitwise_equal(actual, expected)

    @ops(
        [op for op in op_db if op.name in ("clamp", "nn.functional.hardtanh")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "bounds",
        (
            (-0.7, 0.7),
            (-0.0, 0.0),
            (0.0, -0.0),
            (1e-42, 1.0),
            (-1.0, -1e-42),
        ),
    )
    def test_clamp_bounds(self, device, dtype, op, numerics, bounds):
        def fn(x):
            return op.op(x, *bounds)

        x = torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan"), *bounds],
            device=device,
            dtype=dtype,
        )
        x = torch.cat(
            (
                x,
                torch.nextafter(x[-2:], torch.full_like(x[-2:], -float("inf"))),
                torch.nextafter(x[-2:], torch.full_like(x[-2:], float("inf"))),
            )
        ).requires_grad_()
        result = torch.compile(
            fn,
            fullgraph=True,
            options={"numerics": numerics},
        )(x)
        expected = fn(x)
        self._assert_bitwise_equal(result, expected)
        expected_grad = torch.autograd.grad(expected, x, torch.ones_like(x))[0]
        result_grad = torch.autograd.grad(result, x, torch.ones_like(x))[0]
        self._assert_bitwise_equal(result_grad, expected_grad)

    @torch._dynamo.config.patch(capture_scalar_outputs=True, specialize_float=False)
    @config.patch(numerics="strict_pointwise", _use_fp64_for_unbacked_floats=False)
    @ops(
        [op for op in op_db if op.name in ("clamp", "nn.functional.hardtanh")],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    @parametrize("bound_source", ("python_float", "python_int", "tensor"))
    @parametrize("bound_kind", ("lower", "upper"))
    def test_clamp_symbolic_bounds(self, device, dtype, op, bound_source, bound_kind):
        from_tensor = bound_source == "tensor"
        # The functional wrapper validates bounds in Python, which needs concrete values.
        fn_op = op.op
        if from_tensor and op.name == "nn.functional.hardtanh":
            fn_op = torch.ops.aten.hardtanh.default

        def fn(x, bound):
            if from_tensor:
                bound = bound.item()
            if bound_kind == "lower":
                return fn_op(x, bound, 6)
            return fn_op(x, 0, bound)

        x = torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, 5.0, 6.0, 7.0, float("inf")],
            device=device,
            dtype=dtype,
        )
        int_dtype = _BIT_VIEW[dtype]
        nan_bits = x[-1:].view(int_dtype) | 1
        nan_bits = torch.cat((nan_bits, nan_bits | torch.iinfo(int_dtype).min))
        x = torch.cat((x, nan_bits.view(dtype)))
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        if bound_source == "python_int":
            bounds = (1, 2, 257, 2049, 16777217, 9007199254740993, 1)
            if bound_kind == "lower":
                bounds = tuple(-bound for bound in bounds)
        else:
            bounds = (-0.0, -1.0) if bound_kind == "lower" else (6.0, 5.0)
            bounds = (*bounds, 0.7, 0.1, 1e-42, bounds[0])
        if from_tensor:
            bounds = (*bounds, float("nan"), bounds[0])
        for bound in bounds:
            with self.subTest(bound=bound):
                if from_tensor:
                    bound = torch.tensor(bound, dtype=torch.float64)
                expected = fn(x, bound)
                result = compiled(x, bound)
                self._assert_bitwise_equal(result, expected)
        # Python bounds generalize once; tensor bounds recompile once for NaN.
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @ops(
        [op for op in op_db if op.name in ("clamp", "nn.functional.hardtanh")],
        allowed_dtypes=(torch.float32,),
    )
    @parametrize("from_tensor", (False, True))
    @parametrize("bounds", ((float("nan"), 1e100), (-1e100, float("nan"))))
    def test_clamp_nan_bound_before_conversion(
        self, device, dtype, op, from_tensor, bounds
    ):
        fn_op = op.op
        if from_tensor and op.name == "nn.functional.hardtanh":
            fn_op = torch.ops.aten.hardtanh.default

        def fn(x, lo, hi):
            if from_tensor:
                return fn_op(x, lo.item(), hi.item())
            return fn_op(x, lo, hi)

        x = torch.tensor(
            [-1.0, -0.0, 0.0, 1.0, float("nan")], device=device, dtype=dtype
        )
        if from_tensor:
            bounds = tuple(torch.tensor(b, dtype=torch.float64) for b in bounds)
        self._check_bitwise(fn, x, *bounds, numerics="strict_pointwise")

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_softshrink_strict_symbolic_lambd(self, device, dtype, numerics):
        from torch._dynamo.testing import CompileCounterWithBackend

        def fn(x, lambd):
            return torch.nn.functional.softshrink(x, lambd)

        counter = CompileCounterWithBackend("inductor")
        with config.patch({"numerics": numerics, "force_disable_caches": True}):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            # The first value is specialized; later values reuse one graph (f32).
            for lambd in (0.5, 0.1, 0.3):
                # lambd is not representable, so its rounding is observable.
                bound = torch.tensor([lambd, -lambd], dtype=dtype, device=device)
                x = torch.cat(
                    (
                        bound,
                        torch.nextafter(bound, bound * 2),
                        torch.nextafter(bound, bound * 0),
                        torch.tensor(
                            [0.0, -0.0, 0.05, -0.05, float("nan")],
                            dtype=dtype,
                            device=device,
                        ),
                    )
                ).requires_grad_(True)
                grad_output = torch.full_like(x, -0.0)
                grad_output[::2] = 1.5
                expected = fn(x, lambd)
                expected_grad = torch.autograd.grad(expected, x, grad_output)[0]
                result = compiled(x, lambd)
                result_grad = torch.autograd.grad(result, x, grad_output)[0]
                self._assert_bitwise_equal(result, expected)
                self._assert_bitwise_equal(result_grad, expected_grad)
        if dtype == torch.float32:
            # Half-precision inputs round lambd with scalar_tensor, which
            # tensorify cannot handle, so lambd is specialized there.
            self.assertEqual(counter.frame_count, 2)

    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_compile_options_enable_eager_division(self, device, numerics):
        x = torch.full((1024,), 11.0, device=device)
        y = torch.full((1024,), 7.0, device=device)
        tiny = torch.finfo(x.dtype).tiny
        subnormal = tiny * torch.finfo(x.dtype).eps
        x[:6] = torch.tensor(
            [1.0, tiny, subnormal, subnormal, -0.0, -subnormal], device=device
        )
        y[:6] = torch.tensor([tiny, subnormal, 1.0, 2.0, 1.0, 1.0], device=device)

        result, codes = run_and_get_code(
            torch.compile(
                lambda a, b: a / b,
                fullgraph=True,
                options={"numerics": numerics},
            ),
            x,
            y,
        )

        self._assert_bitwise_equal(result, x / y)
        self.assertIn("div_rn", "\n".join(codes))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("requires_grad", (False, True))
    def test_mul_nan_payload_local_mutation(
        self, device, dtype, numerics, requires_grad
    ):
        def fn(z, x, y):
            z.add_(3)
            a = x.clone()
            a.mul_(y)
            return a

        def operands(offset):
            x = torch.arange(2048 + offset, device=device, dtype=dtype)[offset:]
            x[::4].view(torch.int64).fill_(0x7FF8000000000011)
            y = torch.full((2048,), 0.5, device=device, dtype=dtype)
            y[::4].view(torch.int64).fill_(0x7FF8000000000022)
            return (
                torch.arange(8, device=device, dtype=dtype),
                x,
                y.requires_grad_(requires_grad),
            )

        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, fullgraph=True)
            for offset in (0, 1):
                with self.subTest(offset=offset):
                    eager_z, eager_x, eager_y = operands(offset)
                    z, x, y = operands(offset)
                    expected = fn(eager_z, eager_x, eager_y)
                    actual = compiled(z, x, y)
                    self._assert_bitwise_equal(
                        actual.detach(),
                        expected.detach(),
                    )
                    self._assert_bitwise_equal(x, eager_x)
                    self._assert_bitwise_equal(z, eager_z)
                    if requires_grad:
                        gradient = torch.arange(2048, device=device, dtype=dtype) / 8
                        expected_grad = torch.autograd.grad(
                            expected, eager_y, gradient
                        )[0]
                        actual_grad = torch.autograd.grad(actual, y, gradient)[0]
                        self._assert_bitwise_equal(
                            actual_grad,
                            expected_grad,
                        )

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("index", (-1, 1))
    def test_mul_nan_payload_view_mutation_backward(
        self, device, dtype, numerics, index
    ):
        def fn(z, x, y):
            z.add_(3)
            x.add_(1)
            selected = x[index]
            selected.mul_(y)
            return selected

        def operands(offset):
            x = torch.arange(
                2 * 4097 + 2048 + offset, device=device, dtype=dtype
            ).as_strided((3, 2048), (4097, 1), offset)
            x[:, ::4].view(torch.int64).fill_(0x7FF8000000000011)
            y = torch.full((2048,), 0.5, device=device, dtype=dtype)
            y[::4].view(torch.int64).fill_(0x7FF8000000000022)
            # An earlier unrelated mutation must not supply x's provenance.
            return torch.arange(8, device=device, dtype=dtype), x, y.requires_grad_()

        with config.patch({"numerics": numerics, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, fullgraph=True)
            for offset in (1, 0):
                with self.subTest(offset=offset):
                    eager_z, eager_x, eager_y = operands(offset)
                    z, x, y = operands(offset)
                    expected = fn(eager_z, eager_x, eager_y)
                    actual = compiled(z, x, y)
                    self._assert_bitwise_equal(
                        actual.detach(),
                        expected.detach(),
                    )
                    self._assert_bitwise_equal(x, eager_x)
                    self._assert_bitwise_equal(z, eager_z)
                    gradient = torch.arange(2048, device=device, dtype=dtype) / 8
                    expected_grad = torch.autograd.grad(expected, eager_y, gradient)[0]
                    actual_grad = torch.autograd.grad(actual, y, gradient)[0]
                    self._assert_bitwise_equal(
                        actual_grad,
                        expected_grad,
                    )

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("unbacked", (False, True))
    @parametrize("layout", ((2048, 1), (4096, 2), (4097, 1)))
    @parametrize("view", ("select", "slice"))
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_mul_nan_payload_unbacked_mutation_provenance(
        self, device, dtype, numerics, unbacked, layout, view
    ):
        def fn(x, idx, y):
            x.add_(1)
            index_value = idx.item() if unbacked else index
            selected = x[index_value] if view == "select" else x[index_value:]
            selected.add_(2)
            return (selected if view == "select" else x) * y

        def inputs(offset):
            x = torch.arange(
                2 * layout[0] + 2047 * layout[1] + 1 + offset,
                dtype=dtype,
                device=device,
            ).as_strided((3, 2048), layout, offset)
            x[:, ::4].view(torch.int64).fill_(0x7FF8000000000011)
            y = torch.full((2048,), 0.5, dtype=dtype, device=device)
            y[::4].view(torch.int64).fill_(0x7FF8000000000022)
            return x, torch.tensor(index), y

        with config.patch(numerics=numerics):
            compiled = torch.compile(fn, fullgraph=True)
            for index in (-1, 1) if view == "select" else (-1, 1, -4, 3):
                for offset in (1, 0):
                    with self.subTest(index=index, offset=offset):
                        expected_inputs = inputs(offset)
                        actual_inputs = inputs(offset)
                        expected = fn(*expected_inputs)
                        actual = compiled(*actual_inputs)
                        self._assert_bitwise_equal(actual, expected)
                        self._assert_bitwise_equal(
                            actual_inputs[0],
                            expected_inputs[0],
                        )


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering()
    and torch.ops.aten.sum.dim_IntList in native_decomp_table(overrides_only=True),
    "requires CUDA, tl.ReductionOrdering, and the eager inner-tree implementation",
)
class StrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _run(self, fn, *args, **cfg):
        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        return result, "\n".join(codes)

    def _assert_bitwise_equal(self, eager, result):
        self.assertEqual(
            _pointwise_comparison_bits(eager),
            _pointwise_comparison_bits(result),
        )

    def _check_sum(self, device, shape, dim, dtype, keepdim=False, **cfg):
        x = torch.randn(*shape, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, dim, keepdim=keepdim)

        eager = fn(x)
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)
        return code

    @parametrize("case", SUM_CASES, name_fn=lambda c: c[0])
    def test_sum_bitwise(self, device, case):
        name, shape, dim, dtype = case
        code = self._check_sum(device, shape, dim, dtype)
        if name.startswith("persistent"):
            self.assertIn("@triton_heuristics.persistent_reduction(", code)
        elif name.startswith("looped"):
            self.assertIn("for r0_offset in", code)
            self.assertEqual(code.count(INNER_TREE_CALL), 1)
        else:
            self.assertEqual(code.count(INNER_TREE_CALL), 2)

    def _make_prod_input(self, shape, dtype, device):
        # Perturb each element by O(1)/n so the length-n product stays finite
        # (no over/underflow) while remaining order-sensitive; prod shares the
        # sum inner-tree order, only the combiner (*) and identity (1) differ.
        m, n = shape
        cols = torch.arange(n, device=device, dtype=torch.float32).reshape(1, n)
        pattern = (cols % 2) * 2 - 1
        if m > 1:
            rows = torch.arange(m, device=device, dtype=torch.float32).reshape(m, 1)
            pattern = pattern + ((rows % 5) - 2)
        return (1.0 + pattern / n).to(dtype)

    @parametrize("case", PROD_CASES, name_fn=lambda c: c[0])
    def test_prod_bitwise(self, device, case):
        name, shape, dim, dtype = case
        x = self._make_prod_input(shape, dtype, device)

        def fn(z):
            return torch.prod(z, dim)

        eager = fn(x)
        result, code = self._run(fn, x)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)
        if name.startswith("split"):
            self.assertEqual(code.count(INNER_TREE_CALL), 2)

    def test_prod_out_of_scope_uses_default_order(self, device):
        # A dtype-casting prod is out of scope -> falls back to the default order.
        x = self._make_prod_input((64, 300), torch.float32, device)
        _, code = self._run(lambda z: torch.prod(z, 1, dtype=torch.float64), x)
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", SUM_VARIANTS, name_fn=lambda c: c[0])
    def test_sum_variants(self, device, case):
        _, shape, dim, dtype, keepdim, cfg = case
        self._check_sum(device, shape, dim, dtype, keepdim, **cfg)

    @parametrize(
        "input_dtype",
        (torch.float32, torch.float64),
        name_fn=lambda dtype: str(dtype).removeprefix("torch."),
    )
    def test_special_values_match_eager(self, device, input_dtype):
        x = torch.zeros(6, 300, device=device, dtype=input_dtype)
        x[0, 0] = torch.nan
        x[1, 0] = torch.inf
        x[2, 0] = -torch.inf
        x[3, :2] = torch.tensor(
            [torch.inf, -torch.inf], device=device, dtype=input_dtype
        )
        x[4] = -0.0
        x[5] = torch.tensor(
            [1e20, 1.0, -1e20, 1.0], device=device, dtype=input_dtype
        ).repeat(75)

        def fn(z):
            return z.sum(1)

        result, code = self._run(fn, x)
        self._assert_bitwise_equal(fn(x), result)
        self.assertIn(INNER_TREE_CALL, code)

    @skipIfNoCuteDSL
    @parametrize("case", DYNAMIC_CASES, name_fn=lambda c: c[0])
    def test_dynamic_sum(self, device, case):
        _, sizes, cfg = case

        def fn(z):
            return torch.sum(z, 1)

        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for n in sizes:
                x = torch.randn(8, n, device=device)
                self._assert_bitwise_equal(fn(x), compiled(x))

    @parametrize("case", OUT_OF_SCOPE_CASES, name_fn=lambda c: c[0])
    def test_out_of_scope_uses_default_order(self, device, case):
        _, fn = case
        _, code = self._run(fn, torch.randn(64, 300, device=device))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", LAYOUT_CASES, name_fn=lambda c: c[0])
    def test_layout_eligibility(self, device, case):
        _, make_input, eligible = case
        x, dim = make_input(device)

        def fn(z):
            return torch.sum(z, dim)

        result, code = self._run(fn, x)
        self.assertEqual(INNER_TREE_CALL in code, eligible)
        if eligible:
            self._assert_bitwise_equal(fn(x), result)

    def test_unbacked_reduction_size_uses_default_order(self, device):
        def fn(z):
            return z[z > 0].sum(0)

        x = torch.randn(1024, device=device)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            result, code = self._run(fn, x)
        self.assertEqual(result, fn(x))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", SIGNED_ZERO_CASES, name_fn=lambda c: c[0])
    def test_signed_zero(self, device, case):
        _, rows, n, multirow, dtype = case
        x = torch.full((rows, n), -0.0, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, 1)

        cfg = {"triton.persistent_reductions": False} if multirow else {}
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(fn(x), result)
        vector_size = vec_size(x.element_size())
        if multirow:
            num_loads = (n + vector_size - 1) // vector_size
            rblock = (1 << (num_loads - 1).bit_length()) * vector_size
        else:
            rblock = compute_inner_tree_params(n, 1, vector_size).batch_total_elements
        self.assertIn(f"R0_BLOCK: tl.constexpr = {rblock}", code)

    def _make_fusion_case(self, kind, device):
        cfg = {}
        expected_metrics = {}
        kernel_count = None
        result_index = None

        if kind == "nested":
            batch, width, group = 8, 4096, 16
            args = (torch.randn(batch, width, device=device),)
            cfg = {"triton.nested_reduction": True}
            expected_metrics = {"codegen_nested_reduction": 0}

            def fn(x):
                outer = x.amax(-1, keepdim=True)
                y = torch.ops._inductor_test.realize(x + outer)
                return y.reshape(batch, width // group, group).sum(-1)

        elif kind == "mix":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {
                "triton.mix_order_reduction": True,
                "triton.mix_order_reduction_non_strict_mode": True,
            }
            result_index = 0
            expected_metrics = {"codegen_mix_order_reduction": 0}

            def fn(x):
                return x.sum(-1), x.prod(0)

        elif kind == "multi_kernel":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {"triton.multi_kernel": True}

            def fn(x):
                return x.sum(1)

        else:
            args = (torch.randn(8, 300, device=device),)
            cfg = {"online_softmax": True}
            result_index = 1
            kernel_count = 2

            def fn(x):
                return torch.softmax(x, -1), x.sum(-1)

        return fn, args, result_index, cfg, expected_metrics, kernel_count

    @parametrize("kind", FUSION_CASES)
    def test_fusion_preserves_strict_reduction(self, device, kind):
        fn, args, index, cfg, expected_metrics, kernel_count = self._make_fusion_case(
            kind, device
        )
        eager = fn(*args)
        metrics.reset()
        result, code = self._run(fn, *args, **cfg)
        expected = eager if index is None else eager[index]
        actual = result if index is None else result[index]
        self._assert_bitwise_equal(expected, actual)
        self.assertEqual(code.count(INNER_TREE_CALL), 1)
        for metric, expected_value in expected_metrics.items():
            self.assertEqual(getattr(metrics, metric), expected_value)
        if kernel_count is not None:
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
        if kind == "multi_kernel":
            self.assertNotIn("async_compile.multi_kernel(", code)
            self.assertIn("for r0_offset in", code)
        elif kind == "multi_output":
            self.assertEqual(eager[0], result[0])

    def test_combo_kernel_preserves_strict_reduction_blocks(self, device):
        args = (
            torch.randn(8, 12000, device=device),
            torch.randn(8, 12000, device=device),
        )

        def fn(a, b):
            return a.sum(1), b.sum(1)

        eager = fn(*args)
        result, code = self._run(
            fn,
            *args,
            combo_kernels=True,
            combo_kernels_autotune=0,
            combo_kernel_peak_memory_pct_threshold=None,
        )
        for expected, actual in zip(eager, result, strict=True):
            self._assert_bitwise_equal(expected, actual)
        self.assertIn(INNER_TREE_CALL, code)
        self.assertNotIn("combo_grid_meta", code)

    @unittest.skipIf(not SM90OrLater, "requires TMA support")
    @parametrize("kind", ("multirow", "split"))
    def test_tma_preserves_strict_reduction(self, device, kind):
        if kind == "multirow":
            x = torch.randn(64, 5, device=device)
        else:
            x = torch.zeros(1, 65536, device=device)
            params = compute_inner_tree_params(
                x.shape[1], x.shape[0], vec_size(x.element_size())
            )
            for batch, value in enumerate((1e20, 1, -1e20, 1)):
                x[0, batch * params.batch_total_elements] = value

        def fn(z):
            return torch.sum(z, 1)

        eager = fn(x)
        if kind == "split":
            self._assert_bitwise_equal(eager, torch.ones_like(eager))
        result, code = self._run(
            fn,
            x,
            assume_aligned_inputs=True,
            **{"triton.use_tensor_descriptor": True},
        )
        self._assert_bitwise_equal(eager, result)
        if kind == "split":
            self.assertEqual(code.count(INNER_TREE_CALL), 2)
        self.assertIn("tensor_descriptor" if kind == "split" else "tl.store", code)


instantiate_device_type_tests(StrictNumericsCompileTest, globals(), only_for="cuda")
instantiate_device_type_tests(
    StrictNumericsFallbackTest, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(StrictNumericsTest, globals(), only_for="cuda")


# Compare eager and compiled pointwise OpInfos on reference and raw-bit inputs.


# Pointwise ops not represented by UnaryUfuncInfo or BinaryUfuncInfo.
POINTWISE_EXTRA = frozenset(
    {
        "addcmul",
        "addcdiv",
        "clamp",
        "lerp",
        "where",
        "masked_fill",
        "logaddexp2",
        "masked.logaddexp",
        "nn.functional.gelu",
        "nn.functional.hardswish",
        "nn.functional.leaky_relu",
        "native_dropout_backward",
        "alias_copy",
        "clone",
        "contiguous",
        "resolve_conj",
        "resolve_neg",
        "to",
        "view_as_complex",
        "view_as_real",
        "addr",
        "outer",
        "kron",
        "nn.functional.glu",
        "tril",
        "triu",
        "zero_",
        "unravel_index",
        "jiterator_2inputs_2outputs",
        "jiterator_4inputs_with_extra_args",
    }
)

POINTWISE_LOSSES = frozenset(
    {
        "nn.functional.binary_cross_entropy",
        "nn.functional.binary_cross_entropy_with_logits",
        "nn.functional.gaussian_nll_loss",
        "nn.functional.hinge_embedding_loss",
        "nn.functional.huber_loss",
        "nn.functional.kl_div",
        "nn.functional.l1_loss",
        "nn.functional.margin_ranking_loss",
        "nn.functional.mse_loss",
        "nn.functional.poisson_nll_loss",
        "nn.functional.smooth_l1_loss",
        "nn.functional.soft_margin_loss",
    }
)
POINTWISE_FIXED_STATS = frozenset(
    {
        "nn.functional.batch_norm",
        "native_batch_norm",
        "_native_batch_norm_legit",
        "nn.functional.instance_norm",
    }
)
POINTWISE_DROPOUT = frozenset(
    {
        "nn.functional.alpha_dropout",
        "nn.functional.dropout",
        "nn.functional.dropout2d",
        "nn.functional.dropout3d",
        "nn.functional.feature_alpha_dropout",
    }
)
POINTWISE_CONDITIONAL = (
    POINTWISE_LOSSES
    | POINTWISE_FIXED_STATS
    | POINTWISE_DROPOUT
    | {"nn.functional.rrelu", "nn.functional.one_hot", "einsum", "tensordot"}
)
POINTWISE_REPRESENTATIONS = frozenset(
    {
        "alias_copy",
        "clone",
        "contiguous",
        "conj",
        "resolve_conj",
        "resolve_neg",
        "view_as_complex",
        "view_as_real",
        "to",
    }
)
POINTWISE_GEOMETRY = frozenset(
    {
        "addr",
        "outer",
        "kron",
        "nn.functional.glu",
        "view_as_complex",
        "view_as_real",
        "tril",
        "triu",
        "unravel_index",
        "nn.functional.prelu",
    }
)


def _pointwise_op(op):
    # Evaluation needs no seed wrapper, which Dynamo cannot trace.
    if op.name in POINTWISE_DROPOUT | {"nn.functional.rrelu"}:
        return getattr(torch.nn.functional, op.name.rsplit(".", 1)[1])
    return op.op


def _op_id(op):
    name = f"{op.name}_{op.variant_test_name}" if op.variant_test_name else op.name
    return name.replace(".", "_")


def _pointwise_ops():
    seen = set()
    pointwise = []
    for op in op_db:
        if not (
            isinstance(op, (UnaryUfuncInfo, BinaryUfuncInfo))
            or op.name in POINTWISE_EXTRA | POINTWISE_CONDITIONAL
        ):
            continue
        oid = _op_id(op)
        if oid in seen:
            continue
        seen.add(oid)
        pointwise.append(op)
    return pointwise


POINTWISE_OPS = _pointwise_ops()
POINTWISE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _dtype_label(dtype):
    return str(dtype).split(".")[-1]


# Exhaust all 16-bit encodings and sample float32 bit patterns.
NUM_BITPATTERN_SAMPLES = 65536
# Opt in to complete OpInfo samples, reference inputs and raw-bit sweeps.
ALL_SAMPLES = os.getenv("PYTORCH_ALL_SAMPLES", "0") == "1"

_FINFO32 = torch.finfo(torch.float32)
# Include exact values that random bit sampling is unlikely to hit.
_SPECIALS = torch.tensor(
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        float("inf"),
        -float("inf"),
        float("nan"),
        _FINFO32.smallest_normal,
        -_FINFO32.smallest_normal,
        _FINFO32.max,
        _FINFO32.min,
        _FINFO32.eps,
        1.0 + _FINFO32.eps,
        1.0 - _FINFO32.eps / 2,
    ]
)


def _exhaustive_16bit(dtype, device):
    # Narrowing wraps the upper half into negative int16 encodings.
    return (
        torch.arange(0, 65536, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(dtype)
    )


def _substitute(t, y, n, parity=0):
    r"""Substitute data vectors and resize masks, preserving scalar operands."""
    if not isinstance(t, torch.Tensor):
        return t
    if t.dtype == torch.bool:
        # Alternate parity so masks select every 16-bit encoding across two calls.
        mask = torch.zeros(n, dtype=torch.bool, device=y.device)
        mask[parity::2] = True
        return mask
    if (t.is_floating_point() or t.is_complex()) and t.numel() != 1:
        return y.to(t.dtype)
    return t


def _sampled_fp32(n, device, seed=0):
    # Draw the sign separately because randint cannot span 2**32 in int32.
    gen = torch.Generator(device=device).manual_seed(seed)
    bits = torch.randint(
        0, 2**31, (n,), dtype=torch.int32, device=device, generator=gen
    )
    signs = torch.randint(0, 2, (n,), dtype=torch.int32, device=device, generator=gen)
    return torch.cat([(bits | (signs << 31)).view(torch.float32), _SPECIALS.to(device)])


_BIT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def _pointwise_comparison_bits(t):
    """Compare NaN positions and all non-NaN bits, including signed zeros."""
    t = t.resolve_conj().resolve_neg()
    if t.is_complex():
        t = torch.view_as_real(t)
    if t.is_floating_point():
        t = torch.where(torch.isnan(t), float("nan"), t)
    return t.contiguous().reshape(-1).view(torch.uint8)


def _min_max_specials(dtype, device):
    x = torch.tensor(
        [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")],
        dtype=dtype,
        device=device,
    )
    int_dtype = _BIT_VIEW[dtype]
    bits = torch.tensor([float("inf"), float("nan")], dtype=dtype, device=device)
    bits = bits.view(int_dtype) | torch.tensor([1, 17], dtype=int_dtype, device=device)
    bits = torch.cat((bits, bits | torch.iinfo(int_dtype).min))
    return torch.cat((x, bits.view(dtype)))


# Native CUDA domain endpoints and piecewise formula switches.
POINTWISE_BOUNDARIES = {
    "log": (0, 1),
    "log2": (0, 1),
    "log10": (0, 1),
    "log1p": (-1, 0),
    "sqrt": (0,),
    "rsqrt": (0,),
    "asin": (-1, 0, 1),
    "acos": (-1, 0, 1),
    "atanh": (-1, 0, 1),
    "acosh": (1,),
    "sinc": (0,),
    "nn.functional.relu": (0,),
    "nn.functional.relu6": (0, 6),
    "nn.functional.hardsigmoid": (-3, 3),
    "nn.functional.hardshrink": (-0.5, 0.5),
    "nn.functional.leaky_relu": (0,),
    "nn.functional.elu": (0,),
    "nn.functional.selu": (0,),
    "nn.functional.celu": (0,),
    "erfinv": (-1, 1),
    "nn.functional.hardswish": (-3, 3),
    "nn.functional.hardtanh": (-1, 1),
    "nn.functional.softshrink": (-0.5, 0.5),
    "special.log_ndtr": (-1,),
    "special.erfcx": (-26.7, -6.1, 0, 50, 5e7),
    "special.bessel_j0": (-5, -1e-5, 0, 1e-5, 5),
    "special.bessel_j1": (-5, 0, 5),
    "special.bessel_y0": (0, 5),
    "special.bessel_y1": (0, 5),
    "special.modified_bessel_i0": (-8, 0, 8),
    "special.modified_bessel_i1": (-8, 0, 8),
    "special.spherical_bessel_j0": (-0.5, 0, 0.5),
}
POINTWISE_EDGE_NAMES = frozenset(POINTWISE_BOUNDARIES) | {
    "add",
    "sub",
    "mul",
    "clamp",
    "lerp",
    "logit",
    "nn.functional.softplus",
}
POINTWISE_EDGE_DTYPES = (*POINTWISE_DTYPES, torch.float64)


def _pointwise_subnormals(dtype):
    # Construct on CPU so input generation is independent of CUDA flushing.
    zero = torch.tensor(0.0, dtype=dtype)
    tiny = torch.nextafter(zero, torch.tensor(float("inf"), dtype=dtype))
    normal = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype)
    positive = torch.stack(
        (
            zero,
            tiny,
            tiny * 2,
            torch.nextafter(normal, zero),
            normal,
            torch.nextafter(normal, torch.tensor(float("inf"), dtype=dtype)),
        )
    )
    return torch.cat((positive, -positive))


def _pointwise_grad_values(dtype, device):
    if dtype.is_complex:
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _pointwise_grad_values(real_dtype, device)
        zeros = torch.zeros_like(values)
        return torch.cat(
            (
                torch.stack((values, zeros), -1).view(dtype).flatten(),
                torch.stack((zeros, values), -1).view(dtype).flatten(),
            )
        )
    return torch.cat(
        (_pointwise_subnormals(dtype).to(device), _min_max_specials(dtype, device))
    )


def _pointwise_neighbors(boundaries, dtype):
    centers = torch.tensor(boundaries, dtype=dtype)
    centers = centers[torch.isfinite(centers)]
    return torch.cat(
        (
            torch.nextafter(centers, torch.full_like(centers, -float("inf"))),
            centers,
            torch.nextafter(centers, torch.full_like(centers, float("inf"))),
        )
    )


def _pointwise_boundary_values(boundaries, dtype, device):
    return torch.cat(
        (
            _pointwise_neighbors(boundaries, dtype).to(device),
            _pointwise_subnormals(dtype).to(device),
            _min_max_specials(dtype, device),
        )
    )


def _pointwise_edge_calls(op, device, dtype, *, backward=False):
    calls = []

    def add(label, inp, args=(), kwargs=None):
        calls.append((f"edge_{label}", inp, args, kwargs or {}))

    if op.name in POINTWISE_BOUNDARIES:
        x = _pointwise_boundary_values(POINTWISE_BOUNDARIES[op.name], dtype, device)
        add("boundary", x)
    elif op.name in ("add", "sub", "mul") and dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        x = _pointwise_boundary_values((), dtype, device)
        for scalar in (
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            -float("nan"),
        ):
            label = repr(scalar)
            for side, inp, args in (("right", x, (scalar,)), ("left", scalar, (x,))):
                add(f"scalar_{side}_{label}", inp, args)
                if op.name != "mul":
                    add(f"scalar_{side}_{label}_alpha", inp, args, {"alpha": 0.25})
    elif op.name == "lerp":
        weight = _pointwise_boundary_values((-0.5, 0, 0.5, 1), dtype, device)
        start = torch.tensor([1, -1, 0.0, -0.0], dtype=dtype, device=device)
        end = torch.tensor([-1, 1, -0.0, 0.0], dtype=dtype, device=device)
        add(
            "tensor_weight",
            start.repeat_interleave(weight.numel()),
            (end.repeat_interleave(weight.numel()), weight.repeat(start.numel())),
        )
        for value in _pointwise_neighbors((-0.5, 0, 0.5, 1), dtype).tolist():
            add(f"scalar_weight_{value}", start, (end, value))
    elif op.name == "logaddexp" and dtype.is_complex:
        values = _pointwise_edge_values(dtype).to(device)
        a = torch.cat(
            (values, torch.tensor([-1 - 1j, -1 - 100j], device=device, dtype=dtype))
        )
        b = torch.cat(
            (values.flip(0), torch.tensor([0j, 100j], device=device, dtype=dtype))
        )
        count = (1025 + a.numel() - 1) // a.numel()
        a, b = a.repeat(count)[:1025], b.repeat(count)[:1025]
        add("complex_full_block_tail", a, (b,))
        add("complex_offset", a[1:], (b[1:],))
        add("complex_strided", a[::2], (b[::2],))
        add("complex_broadcast", a, (b[:1],))
        add("complex_flags", a.conj(), (b._neg_view(),))
        if dtype == torch.complex32:
            for other_dtype in (torch.complex64, torch.complex128):
                scalar = torch.tensor(70000 + 1j, dtype=other_dtype, device=device)
                add(f"complex_wider_scalar_right_{other_dtype}", a, (scalar,))
                add(f"complex_wider_scalar_left_{other_dtype}", scalar, (a,))
        if dtype == torch.complex64:
            others = (b.to(torch.complex32),)
            if not backward:
                # Eager backward cannot return a complex gradient to a real operand.
                others += (b.real,)
            for other in others:
                add(f"complex_mixed_right_{other.dtype}", a, (other,))
                add(f"complex_mixed_left_{other.dtype}", other, (a,))
    elif op.name == "where" and dtype.is_complex:
        values = _pointwise_edge_values(dtype).to(device)
        count = (1025 + values.numel() - 1) // values.numel()
        a = values.repeat(count)[:1025]
        b = a.flip(0)
        mask = torch.arange(a.numel(), device=device) % 3 == 0
        add("complex_selection_bits", a, (mask, b))
        add("complex_selection_offset", a[1:], (mask[1:], b[1:]))
        for predicate in (False, True):
            add(f"complex_cpu_predicate_{predicate}", a, (torch.tensor(predicate), b))
        if dtype == torch.complex32:
            scalar = torch.tensor([0x7F800001, -2147483648], dtype=torch.int32)
            scalar = scalar.view(torch.float32).view(torch.complex64).reshape(())
            add("complex_cpu_promotion_left", scalar, (mask, a))
            add("complex_cpu_promotion_right", a, (mask, scalar))
        if dtype == torch.complex64:
            other = b.to(torch.complex32)
            add("complex_selection_mixed_right", a, (mask, other))
            add("complex_selection_mixed_left", other, (mask, a))

    elif op.name == "logit":
        acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        tiny = _pointwise_subnormals(dtype)[1].item()
        for eps in (None, -1.0, -1e-50, 0.0, tiny, 0.2, 0.6, float("nan")):
            boundaries = [0, 1]
            if eps is not None:
                lo = torch.tensor(eps, dtype=acc_dtype)
                boundaries += [lo.item(), (1 - lo).item()]
            x = _pointwise_boundary_values(boundaries, dtype, device)
            add(f"eps_{eps}", x, kwargs={"eps": eps})
    elif op.name == "nn.functional.softplus":
        for beta, threshold in ((1, 20), (2, 1), (-2, 1)):
            x = _pointwise_boundary_values((0, threshold / beta), dtype, device)
            add(
                f"beta_{beta}_threshold_{threshold}",
                x,
                kwargs={"beta": beta, "threshold": threshold},
            )
    elif op.name == "clamp":
        x = _pointwise_boundary_values((-1, 1), dtype, device)
        for lo, hi in ((-1.0, 1.0), (-1.0, None), (None, 1.0), (1.0, -1.0)):
            add(f"scalar_bounds_{lo}_{hi}", x, (lo, hi))
        bounds = _min_max_specials(dtype, device)
        x = x.repeat_interleave(bounds.numel())
        lo = bounds.repeat(x.numel() // bounds.numel())
        add("tensor_min", x, (lo, None))
        add("tensor_max", x, (None, lo))
        add("tensor_bounds", x, (lo, lo.flip(0)))
    return calls


def _pointwise_edge_values(dtype):
    if dtype == torch.bool:
        return torch.tensor([False, True])
    if dtype.is_complex:
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        values = _pointwise_edge_values(real_dtype)
        zeros = torch.zeros_like(values)
        return torch.cat(
            (
                torch.stack((values, zeros), -1).view(dtype).flatten(),
                torch.stack((zeros, values), -1).view(dtype).flatten(),
            )
        )
    if dtype.is_floating_point:
        limits = torch.finfo(dtype)
        return torch.cat(
            (
                _pointwise_boundary_values((-2, -1, -0.5, 0, 0.5, 1, 2), dtype, "cpu"),
                torch.tensor([limits.min, limits.max], dtype=dtype),
            )
        )
    limits = torch.iinfo(dtype)
    values = {limits.min, limits.min + 1, 0, 1, 2, limits.max - 1, limits.max}
    if limits.min < 0:
        values.add(-1)
    return torch.tensor(sorted(values), dtype=dtype)


def _diff_kind(a, b):
    r"""Classify mismatches as shape, value, or signed zero."""
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return "shape"
        kinds = set()
        for x, y in zip(a, b):
            if x is None or y is None:
                if x is not y:
                    return "shape"
                continue
            if not _outputs_equal(x, y):
                kinds.add(_diff_kind(x, y))
        return "value" if "value" in kinds or not kinds else "+".join(sorted(kinds))
    if a.dtype != b.dtype or a.shape != b.shape:
        return "shape"
    a, b = a.resolve_conj().resolve_neg(), b.resolve_conj().resolve_neg()
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    int_dtype = _BIT_VIEW.get(a.dtype)
    if int_dtype is None:
        return "value"
    differ = (a.contiguous().view(int_dtype) != b.contiguous().view(int_dtype)) & ~(
        torch.isnan(a) & torch.isnan(b)
    )
    zero = differ & (a == 0) & (b == 0)
    if not bool((differ == zero).all()):
        return "value"
    return "signed-zero" if bool(zero.any()) else "value"


def _diff_outputs(out):
    return [
        t
        for t in tree_leaves(out)
        if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())
    ]


def _outputs_equal(a, b):
    r"""Compare NaN positions and exact bits for all other values."""
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x is None or y is None:
                if x is not y:
                    return False
                continue
            if not _outputs_equal(x, y):
                return False
        return True
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(_pointwise_comparison_bits(a), _pointwise_comparison_bits(b))


def _requires_pointwise_kernel(op, dtype):
    return not (
        op.name in POINTWISE_REPRESENTATIONS | POINTWISE_DROPOUT | {"positive"}
        or op.name in {"real", "conj_physical"}
        and not dtype.is_complex
    )


def _scalar_tensor_signature(t):
    bits = (
        t.detach()
        .resolve_conj()
        .resolve_neg()
        .cpu()
        .reshape(-1)
        .clone(memory_format=torch.contiguous_format)
        .view(torch.uint8)
        .tolist()
    )
    return t.dtype, t.device, bytes(bits)


def _representative_samples(samples):
    def signature(t):
        if not isinstance(t, torch.Tensor):
            return t
        return (
            t.dtype,
            t.device,
            t.layout,
            tuple(min(size, 2) for size in t.shape),
            t.is_contiguous(),
            tuple(sorted(range(t.ndim), key=lambda dim: t.stride(dim))),
            tuple(stride == 0 for stride in t.stride()),
            t.is_conj(),
            t.is_neg(),
            t.requires_grad,
        )

    seen = set()
    for sample in samples:
        scalar_parameters = [
            _scalar_tensor_signature(t)
            for t in tree_leaves((sample.args, sample.kwargs))
            if isinstance(t, torch.Tensor) and t.numel() == 1
        ]
        key = repr(
            (
                tree_map(signature, (sample.input, sample.args, sample.kwargs)),
                scalar_parameters,
            )
        )
        if key not in seen:
            seen.add(key)
            yield sample


BACKWARD_OPS = [op for op in POINTWISE_OPS if op.supports_autograd]

# (op_id, dtype_label) pairs that must still differ from eager.
POINTWISE_XFAIL: frozenset[tuple[str, str]] = frozenset()

BACKWARD_XFAIL: frozenset[tuple[str, str]] = frozenset()

NONFLOAT_XFAIL: frozenset[tuple[str, str]] = frozenset()


POINTWISE_STRICT_CFG = {
    "numerics": "strict",
    "triton.autotune_pointwise": False,
}

# These pairs need expanded reference inputs to exercise existing mismatches.
REFERENCE_INPUTS = frozenset(
    {
        ("addcdiv", "bfloat16"),
        ("addcdiv", "float16"),
        ("addcdiv", "float32"),
        ("double", "float16"),
        ("hypot", "float16"),
        ("nn_functional_mish", "bfloat16"),
        ("nn_functional_silu", "bfloat16"),
        ("nn_functional_silu_complex", "complex128"),
    }
)


# Compare complete calls to these deterministic native kernels for every degree.
NATIVE_POLYNOMIALS = {
    "special.laguerre_polynomial_l": torch.ops.aten.special_laguerre_polynomial_l,
    "special.legendre_polynomial_p": torch.ops.aten.special_legendre_polynomial_p,
}


class _NativeCallProbe(TorchDispatchMode):
    """Verify a shared native call without evaluating its value-dependent series."""

    def __init__(self, packet, *, compiling=False):
        super().__init__()
        self.overloads = tuple(packet.op_overloads())
        self.schema_names = {
            op.default._schema.name for op in NATIVE_POLYNOMIALS.values()
        }
        self.compiling = compiling
        self.frames = []
        self.output = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

        kwargs = kwargs or {}
        if func._schema.name in self.schema_names and func not in self.overloads:
            raise AssertionError("Unexpected native polynomial operator")
        if func in self.overloads and not any(
            is_fake(value) for value in tree_leaves((args, kwargs))
        ):
            if self.compiling:
                raise AssertionError("Native polynomial execution during compilation")
            if func._schema.is_mutable or any(
                value.alias_info is not None
                for value in (*func._schema.arguments, *func._schema.returns)
            ):
                raise AssertionError("Expected a pure native polynomial schema")
            if torch.Tag.pointwise not in func.tags or any(
                tag in func.tags
                for tag in (
                    torch.Tag.nondeterministic_bitwise,
                    torch.Tag.nondeterministic_seeded,
                )
            ):
                raise AssertionError("Expected a deterministic pointwise kernel")

            def encode(value):
                if isinstance(value, torch.Tensor):
                    flat = value.detach().resolve_conj().resolve_neg().cpu().reshape(-1)
                    packed = torch.empty((flat.numel(),), dtype=flat.dtype)
                    packed.copy_(flat)
                    digest = hashlib.sha256(
                        packed.view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    return (
                        value.dtype,
                        value.device,
                        tuple(value.shape),
                        value.stride(),
                        value.storage_offset(),
                        value.data_ptr(),
                        value.untyped_storage().data_ptr(),
                        value.requires_grad,
                        value.is_neg(),
                        value.is_conj(),
                        digest,
                    )
                if isinstance(value, float):
                    return (float, struct.pack("!d", value))
                return (type(value), value)

            self.frames.append(
                (
                    func,
                    tree_map(encode, args),
                    tree_map(encode, kwargs),
                    torch.get_default_dtype(),
                    torch.cuda.current_device(),
                    torch.cuda.current_stream().cuda_stream,
                    torch.is_autocast_enabled("cuda"),
                    torch._C._dispatch_dump_table(func._schema.name),
                )
            )
            fake = FakeTensorMode()

            def convert(value):
                return (
                    fake.from_tensor(value)
                    if isinstance(value, torch.Tensor)
                    else value
                )

            with fake:
                meta = func(*tree_map(convert, args), **tree_map(convert, kwargs))
            if meta.dtype not in (torch.float32, torch.float64):
                raise AssertionError(
                    "Native polynomials dispatch only float32 and float64"
                )
            # This tensor is a return-identity probe, never a numerical result.
            self.output = torch.empty_strided(
                meta.shape, meta.stride(), dtype=meta.dtype, device=meta.device
            ).zero_()
            return self.output
        return func(*args, **kwargs)

    def matches(self, other, eager_result, compiled_result):
        return (
            len(self.frames) == len(other.frames) == 1
            and self.frames == other.frames
            and eager_result is self.output
            and compiled_result is other.output
        )


def _native_polynomial_probe(op):
    packet = NATIVE_POLYNOMIALS.get(op.name)
    return _NativeCallProbe(packet) if packet is not None else nullcontext()


def _native_call_wrapper_is_direct(code, operator):
    """Allow a native call and metadata checks, but no tensor computations."""
    runtime = next(
        (
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.FunctionDef) and node.name == "call"
        ),
        None,
    )
    if runtime is None:
        return False
    output = None
    returned = False
    inputs = set()
    metadata_checks = {
        "assert_size_stride",
        "assert_size_stride_grouped",
        "assert_tensor_metadata",
        "assert_alignment",
    }

    def simple(node):
        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        if isinstance(node, ast.Attribute):
            return simple(node.value)
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(simple(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(
                node.operand, ast.Constant
            )
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and not node.keywords
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("nan", "inf", "-inf")
        )

    def simple_args(call):
        return all(simple(item) for item in call.args) and all(
            simple(item.value) for item in call.keywords
        )

    def check(statements):
        nonlocal output, returned
        for node in statements:
            if isinstance(node, ast.With):
                if (
                    len(node.items) != 1
                    or not isinstance(node.items[0].context_expr, ast.Call)
                    or ast.unparse(node.items[0].context_expr.func)
                    != "torch.cuda._DeviceGuard"
                    or not simple_args(node.items[0].context_expr)
                    or not check(node.body)
                ):
                    return False
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, (ast.Tuple, ast.List))
                    and all(isinstance(item, ast.Name) for item in target.elts)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "args"
                ):
                    inputs.update(item.id for item in target.elts)
                elif isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    if not simple_args(node.value):
                        return False
                    call = ast.unparse(node.value.func)
                    if (
                        call == "copy_if_misaligned"
                        and output is None
                        and target.id in inputs
                    ):
                        if (
                            len(node.value.args) != 1
                            or ast.unparse(node.value.args[0]) != target.id
                        ):
                            return False
                    elif call == f"torch.ops.{operator}" and output is None:
                        output = target.id
                    else:
                        return False
                else:
                    return False
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = ast.unparse(node.value.func)
                if call not in metadata_checks | {
                    "args.clear",
                    "torch.cuda.set_device",
                } or not simple_args(node.value):
                    return False
            elif isinstance(node, ast.Delete):
                if not all(
                    isinstance(target, ast.Name) and target.id in inputs
                    for target in node.targets
                ):
                    return False
            elif isinstance(node, ast.Return):
                if (
                    output is None
                    or returned
                    or not isinstance(node.value, ast.Tuple)
                    or len(node.value.elts) != 1
                    or not isinstance(node.value.elts[0], ast.Name)
                    or node.value.elts[0].id != output
                ):
                    return False
                returned = True
            else:
                return False
        return True

    return check(runtime.body) and returned


class _RngOpDetector(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.has_rng = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if torch.Tag.nondeterministic_seeded in getattr(func, "tags", ()):
            self.has_rng = True
        return func(*args, **(kwargs or {}))


# The ledger was measured on sm_89 (CUDA 13.0/13.2), sm_90, and sm_100.
_LEDGER_ARCH = LazyVal(lambda: bool(IS_SM89 or IS_SM90 or IS_SM100))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None and _LEDGER_ARCH,
    "requires CUDA and Triton on sm_89, sm_90 or sm_100 (the ledger's measured arches)",
)
class PointwiseStrictNumericsTest(TestCase):
    @dtypes(torch.float32, torch.complex64)
    @parametrize("other_kind", ("tensor", "scalar"))
    def test_scalar_input_reference_samples(self, device, dtype, other_kind):
        self._coverage = CoverageRecord({"test": self._testMethodName})
        condition = torch.tensor([True, False], device=device)
        value = complex(2, 1) if dtype.is_complex else 2.0
        other = complex(3, 2) if dtype.is_complex else 3.0
        if other_kind == "tensor":
            other = torch.full(
                (2,), other, dtype=dtype, device=device, requires_grad=True
            )
        sample = SampleInput(value, args=(condition, other))
        calls = self._build_calls([sample])
        self.assertEqual(len(calls), 1)
        op = next(op for op in op_db if op.name == "where")
        mismatches = self._sweep_backward(
            device, op, dtype, POINTWISE_STRICT_CFG, calls=[("ref", *calls[0])]
        )
        self.assertEqual(mismatches, [])
        self.assertGreater(self._coverage.compared, 0)
        self.assertEqual(self._coverage.filters["no_input_gradients"], 0)

    @dtypes(torch.complex32)
    @parametrize("activation", ("sigmoid", "tanh"))
    @parametrize("shape", ((128, 128), (1024, 1024)))
    def test_complex_half_transposed_activation_backward(
        self, device, dtype, activation, shape
    ):
        torch.manual_seed(0)
        x = (torch.randn(shape, device=device, dtype=torch.complex64) * 5).to(dtype).t()
        grad = torch.randn(x.shape, device=device, dtype=dtype)
        x[0, 0] = complex(2.55859375, 8.640625)
        grad[0, 0] = complex(-0.31005859375, 0.250244140625)
        output = getattr(torch, activation)(x)
        fn = getattr(torch.ops.aten, activation + "_backward").default
        expected = fn(grad, output)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(grad, output)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @parametrize("offset", (0, 1))
    @parametrize("shape", ((), (8,), (1024,)))
    def test_jiterator_input_alignment(self, device, dtype, offset, shape):
        value = (
            complex(2.6924262046813965, 6.4225850105285645)
            if dtype.is_complex
            else 2.6924262046813965
        )
        length = math.prod(shape)
        base = torch.full((length + 1,), value, device=device, dtype=dtype)
        x = base.narrow(0, offset, length).reshape(shape)
        fn = torch.cuda.jiterator._create_jit_fn(
            "template <typename T> T unary(T x) { return x * x + x; }"
        )
        with config.patch(POINTWISE_STRICT_CFG), config.patch(fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True)
            for current_offset in (offset, 1 - offset):
                x = base.narrow(0, current_offset, length).reshape(shape)
                expected = fn(x)
                actual = compiled(x)
                self.assertEqual(
                    _pointwise_comparison_bits(actual),
                    _pointwise_comparison_bits(expected),
                )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_scalar_power_rounds_before_widening(self, device, dtype):
        x = torch.tensor([3.0, 9.0], device=device, dtype=dtype)

        def fn(x):
            return torch.pow(x, -2).double()

        expected = fn(x)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("exponent", (4, 7, -2, -3, 31))
    def test_scalar_power_native_specializations(self, device, dtype, exponent):
        x = torch.tensor([-8, -2.3, -0.3, 0, 0.1, 2.3, 8], device=device, dtype=dtype)

        def fn(x):
            return torch.pow(x, exponent)

        expected = fn(x)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    @dtypes(
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float64
    )
    @parametrize("exponent", (4, 7, -2, -3, 31))
    def test_float_power_native_specializations(self, device, dtype, exponent):
        x = torch.tensor([0, 1, 2, 3, 8], device=device, dtype=dtype)

        def fn(x):
            return torch.float_power(x, exponent)

        expected = fn(x)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    @dtypes(torch.complex32)
    @parametrize("activation", ("sigmoid", "tanh"))
    @parametrize("length", (1, 8, 128, 129, 1024, 1029 * 917))
    def test_complex_half_activation_backward(self, device, dtype, activation, length):
        torch.manual_seed(0)
        x = (torch.randn(length, device=device, dtype=torch.complex64) * 5).to(dtype)
        grad = torch.randn_like(x)
        x[0] = complex(2.55859375, 8.640625)
        grad[0] = complex(-0.31005859375, 0.250244140625)
        output = getattr(torch, activation)(x)
        fn = getattr(torch.ops.aten, activation + "_backward").default
        expected = fn(grad, output)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(grad, output)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    @dtypes(torch.float64)
    @parametrize("op_name", ("modified_bessel_i0", "modified_bessel_i1"))
    @parametrize("length", (1, 128, 1024))
    def test_modified_bessel_recurrence(self, device, dtype, op_name, length):
        torch.manual_seed(0)
        x = torch.empty(length, device=device, dtype=dtype).uniform_(-10, 10)
        x[0] = 8.905643855488954
        fn = getattr(torch.special, op_name)
        expected = fn(x)
        with config.patch(POINTWISE_STRICT_CFG):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(
            _pointwise_comparison_bits(actual), _pointwise_comparison_bits(expected)
        )

    _coverage = None

    def run(self, result=None):
        directory = os.getenv("PYTORCH_POINTWISE_COVERAGE_DIR")
        case = coverage_case(self._testMethodName)
        if not directory or case is None:
            return super().run(result)
        ledgers = {
            "test_pointwise_bitwise": POINTWISE_XFAIL | NONFLOAT_XFAIL,
            "test_pointwise_backward": BACKWARD_XFAIL,
        }
        self._coverage = CoverageRecord(
            case,
            (case["op"], case["dtype"]) in ledgers.get(case["family"], ()),
            required_sample_kinds=("opinfo", "edges"),
        )
        self._coverage.kernel_required = case["op"] not in {
            _op_id(op)
            for op in POINTWISE_OPS
            if not _requires_pointwise_kernel(op, getattr(torch, case["dtype"]))
        }
        if result is None:
            result = self.defaultTestResult()
        try:
            with record_result(self._coverage, result, self):
                return super().run(result)
        except BaseException as error:
            self._coverage.status, self._coverage.reason = "failed", str(error)
            raise
        finally:
            write_event(directory, self._coverage.event())

    def _coverage_filter(self, reason, count=1):
        if self._coverage is not None and count:
            self._coverage.filters[reason] += count

    def _coverage_comparison(self, equal, kind="", tag="sample"):
        if self._coverage is not None:
            self._coverage.compared += 1
            self._coverage.compared_by_sample_kind[
                "edges" if tag.startswith("edge_") else "opinfo"
            ] += 1
            self._coverage.matched += int(equal)
            self._coverage.generated_kernels = metrics.generated_kernel_count
            self._coverage.generated_fallbacks = metrics.generated_fallback_kernel_count
            if not equal:
                self._coverage.mismatches[kind] += 1

    @dtypes(torch.float64)
    @parametrize(
        "change", ("none", "dtype", "stride", "value", "result", "duplicate", "inplace")
    )
    def test_native_call_probe_detects_changed_execution(self, device, dtype, change):
        from torch._dynamo.backends.registry import lookup_backend
        from torch._inductor.graph import GraphLowering

        packet = torch.ops.aten.special_laguerre_polynomial_l
        x = torch.tensor([[0.25, 0.5], [0.75, 1]], device=device, dtype=dtype)
        degree = torch.full_like(x, 1e20)
        eager = _NativeCallProbe(packet)
        with eager:
            expected = packet(x, degree)
        observed = []
        wrappers = []

        def fn(x, degree):
            if change == "dtype":
                x, degree = x.float(), degree.float()
            elif change == "stride":
                x, degree = x.t(), degree.t()
            elif change == "value":
                x = x + 1
            result = packet(x, degree)
            if change == "result":
                result = result + 1
            elif change == "duplicate":
                result = packet(x, degree) + result
            elif change == "inplace":
                result.add_(1)
            return result

        def backend(gm, inputs):
            with (
                _NativeCallProbe(packet, compiling=True),
                mock.patch.object(GraphLowering, "save_output_code", wrappers.append),
            ):
                compiled = lookup_backend("inductor")(gm, inputs)

            def execute(*args):
                probe = _NativeCallProbe(packet)
                with probe:
                    result = compiled(*args)
                observed.append(probe)
                return result

            return execute

        torch._dynamo.reset()
        with config.patch(fx_graph_cache=False, fx_graph_remote_cache=False):
            actual = torch.compile(fn, backend=backend, fullgraph=True)(x, degree)
        self.assertEqual(len(observed), 1)
        self.assertEqual(len(wrappers), 1)
        direct = _native_call_wrapper_is_direct(wrappers[0], observed[0].frames[0][0])
        matched = eager.matches(observed[0], expected, actual)
        if change == "inplace":
            self.assertTrue(matched)
            self.assertEqual(actual, torch.ones_like(expected), atol=0, rtol=0)
            self.assertFalse(direct)
        self.assertEqual(direct and matched, change == "none")

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        # Release large reference-input allocations between concurrent tests.
        torch.cuda.empty_cache()

    @dtypes(torch.float32)
    @parametrize("parameter_kind", ("python", "tensor"))
    def test_representative_samples_keep_parameters_and_layouts(
        self, device, dtype, parameter_kind
    ):
        x = torch.randn((2, 3), device=device, dtype=dtype)
        parameters = (0.0, -0.0) if parameter_kind == "tensor" else (0.1, 0.2)
        if parameter_kind == "tensor":
            parameters = tuple(
                torch.tensor([p, 1.0], device=device)[::2] for p in parameters
            )
        samples = [
            SampleInput(x, args=(parameters[0],)),
            SampleInput(torch.ones((4, 5), device=device), args=(parameters[0],)),
            SampleInput(x.t(), args=(parameters[0],)),
            SampleInput(x[:, ::2], args=(parameters[0],)),
            SampleInput(x.expand(2, 2, 3), args=(parameters[0],)),
            SampleInput(x, args=(parameters[1],)),
        ]
        selected = list(_representative_samples(samples))
        self.assertEqual(len(selected), 5)
        self.assertIs(selected[0], samples[0])
        self.assertIs(selected[-1], samples[-1])

        if parameter_kind == "tensor":
            parameters = (
                torch.tensor(1.0, dtype=dtype),
                torch.tensor(1.0, dtype=dtype, device=device),
                torch.tensor(2.0, dtype=dtype, device=device),
                torch.tensor(1.0, dtype=torch.float64, device=device),
                *parameters,
                *torch.tensor([0x7FC00011, 0x7FC00012], device=device)
                .to(torch.int32)
                .view(dtype)
                .unbind(),
            )
        op = next(op for op in POINTWISE_OPS if op.name == "mul")
        for keyword in (False, True):
            samples = [
                SampleInput(x, kwargs={"other": p})
                if keyword
                else SampleInput(x, args=(p,))
                for p in parameters
            ]
            self.assertEqual(len(list(_representative_samples(samples))), len(samples))
            with mock.patch.object(self, "_sample_inputs", return_value=samples):
                calls = self._bitpattern_call(op, dtype, device)
            self.assertEqual(len(calls), len(samples))
            for (_, args, kwargs), parameter in zip(calls, parameters, strict=True):
                self.assertIs(kwargs["other"] if keyword else args[0], parameter)

    @dtypes(torch.float32)
    @parametrize("full", (False, True))
    def test_combined_samples_preserve_seeded_edges(self, device, dtype, full):
        op = next(op for op in POINTWISE_OPS if op.name == "nn.functional.relu")
        torch.manual_seed(0)
        expected = self._edge_calls(op, device, dtype)
        with mock.patch(f"{__name__}.ALL_SAMPLES", full):
            calls = self._combined_calls(device, op, dtype)
        actual = [call for call in calls if call[0].startswith("edge_")]
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            self.assertEqual(result[0], reference[0])
            self.assertTrue(_outputs_equal(result[1], reference[1]))
            self.assertEqual(result[2:], reference[2:])

        if not full:
            inp = next(call[1] for call in calls if call[0] == "bits")
            actual_bits = inp.view(torch.int32).tolist()
            expected_bits = _sampled_fp32(256, device).view(torch.int32).tolist()
            self.assertEqual(set(actual_bits), set(expected_bits))
            self.assertEqual(len(actual_bits), len(set(expected_bits)))

    @dtypes(torch.float16, torch.bfloat16)
    def test_keyword_input_grads(self, device, dtype):
        inp = torch.tensor([-2.0], device=device, dtype=dtype)
        weight = torch.tensor([0.25], device=device, dtype=dtype)

        def fn(inp, args, kwargs):
            return torch.nn.functional.prelu(inp, *args, **kwargs)

        _, grads = self._input_grads(
            fn, inp, (), {"weight": weight}, torch.ones_like(inp)
        )
        self.assertEqual(grads, (weight, inp))

    @dtypes(torch.complex64)
    def test_nested_multi_output_input_grads(self, device, dtype):
        x = torch.tensor([1 + 2j, -3 + 4j], device=device, dtype=dtype).requires_grad_()
        y = torch.tensor(2 - 1j, device=device, dtype=dtype).requires_grad_()
        constant = torch.tensor(3 + 2j, device=device, dtype=dtype)
        g1, g2 = torch.ones_like(x), torch.full_like(x, 2 + 1j)

        def fn(inputs, args, kwargs):
            a, b = inputs
            return a * b, a * kwargs["constant"]

        _, grads = self._input_grads(
            fn, (x, y), (), {"constant": constant}, (g1, g2), respect_requires_grad=True
        )
        expected = torch.autograd.grad(
            (x * y, x * constant), (x, y), grad_outputs=(g1, g2)
        )
        self.assertTrue(_outputs_equal(grads, expected))
        self.assertEqual(len(grads), 2)
        # The scalar's gradient must include the broadcast sum.
        self.assertEqual(grads[1], (g1 * x.conj()).sum())

    @dtypes(torch.float32)
    def test_swapped_output_gradients_are_detected(self, device, dtype):
        class Swapped(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return 2 * x, 3 * x

            @staticmethod
            def backward(ctx, g1, g2):
                return 3 * g1 + 2 * g2

        class Op:
            name, variant_test_name = "clone", ""

            @staticmethod
            def op(x):
                return 2 * x, 3 * x

        x = torch.ones(4, device=device, dtype=dtype).requires_grad_()
        with mock.patch(
            "torch.compile", return_value=lambda inp, args, kwargs: Swapped.apply(inp)
        ):
            mismatches = self._sweep_backward(
                device, Op(), dtype, POINTWISE_STRICT_CFG, calls=[("sample", x, (), {})]
            )
        self.assertTrue(mismatches)
        self.assertEqual(mismatches[0][-1], "value")

    @dtypes(torch.complex64)
    def test_bitwise_lazy_views(self, device, dtype):
        x = torch.tensor([1 + 2j, -0.0 - 3j], dtype=dtype, device=device)
        for view in (x.conj(), torch._neg_view(x)):
            self.assertTrue(_outputs_equal(view, view.resolve_conj().resolve_neg()))
            self.assertEqual(_diff_kind(view, view + 1), "value")
        nan = torch.tensor([0x7FC00001, -0x3FFFEF], dtype=torch.int32, device=device)
        a, b = nan.view(torch.float32), nan.flip(0).view(torch.float32)
        self.assertTrue(_outputs_equal(a, b))
        self.assertFalse(_outputs_equal(a, torch.zeros_like(a)))
        self.assertFalse(
            _outputs_equal(a, torch.tensor([float("nan"), 0], device=device))
        )
        self.assertFalse(_outputs_equal(a.view(torch.int32), b.view(torch.int32)))
        z = torch.tensor([-0.0, 0.0], device=device)
        self.assertFalse(_outputs_equal(z, z.flip(0)))
        self.assertEqual(_diff_kind(z, z.flip(0)), "signed-zero")
        real = torch.stack((a, torch.ones_like(a)), dim=-1)
        other = torch.stack((b, torch.ones_like(b)), dim=-1)
        self.assertTrue(
            _outputs_equal(torch.view_as_complex(real), torch.view_as_complex(other))
        )
        self.assertFalse(
            _outputs_equal(
                torch.view_as_complex(real),
                torch.view_as_complex(real.flip(-1).contiguous()),
            )
        )

    @dtypes(torch.float32)
    def test_conditional_pointwise_samples(self, device, dtype):
        for name in POINTWISE_CONDITIONAL:
            for op in (o for o in POINTWISE_OPS if o.name == name):
                with self.subTest(op=_op_id(op)):
                    sample_dtype = (
                        torch.int64 if name == "nn.functional.one_hot" else dtype
                    )
                    calls = self._collect_calls(device, op, sample_dtype)
                    self.assertTrue(calls)
                    for _, inp, args, kwargs in calls:
                        if name in POINTWISE_LOSSES:
                            self.assertEqual(kwargs["reduction"], "none")
                        elif name in POINTWISE_DROPOUT | {"nn.functional.rrelu"}:
                            self.assertFalse(kwargs["training"])

    @dtypes(torch.float32)
    def test_coverage_filtered_samples(self, device, dtype):
        self._coverage = CoverageRecord({"test": self._testMethodName})
        x = torch.ones(2, device=device, dtype=dtype)
        reduction = next(op for op in op_db if op.name == "sum")
        with self.assertRaisesRegex(AssertionError, "no usable sample"):
            self._sweep(
                device,
                reduction,
                dtype,
                POINTWISE_STRICT_CFG,
                calls=[("sample", x, (), {})],
            )
        self.assertEqual(self._coverage.filters["output_shape_not_pointwise"], 1)
        self.assertEqual(self._coverage.compared, 0)
        remainder = next(op for op in POINTWISE_OPS if op.name == "__rmod__")
        calls = self._collect_calls(device, remainder, dtype, backward=True)
        self.assertTrue(calls)
        self.assertTrue(
            all(isinstance(args[0], torch.Tensor) for _, _, args, _ in calls)
        )
        self.assertGreater(
            self._coverage.filters["scalar_remainder_has_no_eager_backward"], 0
        )
        for name in ("positive", "real", "conj_physical", "neg"):
            op = next(op for op in POINTWISE_OPS if op.name == name)
            for sample_dtype in (torch.float32, torch.int64, torch.complex64):
                required = name == "neg" or (
                    name in {"real", "conj_physical"} and sample_dtype.is_complex
                )
                self.assertEqual(_requires_pointwise_kernel(op, sample_dtype), required)
                with mock.patch.object(metrics, "generated_kernel_count", 0):
                    self._require_kernel(1, "no sample", op, sample_dtype)
                    self.assertFalse(self._coverage.kernel_required)

    def _is_tensor_output(self, out):
        return isinstance(out, torch.Tensor) or (
            isinstance(out, (tuple, list))
            and all(isinstance(t, torch.Tensor) for t in out)
        )

    def _is_pointwise_output(self, inp, args, kwargs, out, op=None):
        if (
            op is not None
            and op.name in POINTWISE_EXTRA | POINTWISE_CONDITIONAL | POINTWISE_GEOMETRY
        ):
            return True
        shapes = [
            t.shape
            for t in tree_leaves((inp, args, kwargs))
            if isinstance(t, torch.Tensor)
        ]
        try:
            bshape = torch.broadcast_shapes(*shapes)
        except Exception:
            return False
        outs = tree_leaves(out)
        return all(isinstance(o, torch.Tensor) and o.shape == bshape for o in outs)

    def _build_calls(self, samples):
        r"""Preserve every sampled signature, including scalar first arguments."""
        return [
            (sample.input, tuple(sample.args), dict(sample.kwargs))
            for sample in samples
        ]

    def _sample_inputs(self, op, device, dtype, *, requires_grad=False):
        samples = list(op.sample_inputs(device, dtype, requires_grad=requires_grad))
        if isinstance(op, BinaryUfuncInfo) and dtype.is_floating_point:
            samples.extend(
                generate_elementwise_binary_with_scalar_samples(
                    op, device=device, dtype=dtype, requires_grad=requires_grad
                )
            )
        return samples

    def _pointwise_call(self, op, inp, args, kwargs):
        kwargs = dict(kwargs)
        if op.name in POINTWISE_LOSSES:
            kwargs.pop("size_average", None)
            kwargs.pop("reduce", None)
            kwargs["reduction"] = "none"
        elif op.name in POINTWISE_DROPOUT or op.name == "nn.functional.rrelu":
            kwargs["training"] = False
        elif op.name == "nn.functional.instance_norm":
            if kwargs.get("running_mean") is None or kwargs.get("running_var") is None:
                return None
            kwargs["use_input_stats"] = False
        elif op.name == "nn.functional.batch_norm":
            if len(args) < 2 or args[0] is None or args[1] is None:
                return None
            kwargs["training"] = False
        elif op.name in POINTWISE_FIXED_STATS:
            if len(args) < 7 or args[2] is None or args[3] is None:
                return None
            args = (*args[:4], False, *args[5:])
        elif op.name == "tensordot":
            kwargs["dims"] = 0
        elif op.name == "einsum":
            equation = args[0].replace(" ", "")
            if "->" not in equation:
                return None
            left, right = equation.split("->")
            if set(left.replace(",", "")) - set(right):
                return None
        elif op.name == "nn.functional.one_hot":
            classes = args[0] if args else kwargs.get("num_classes", -1)
            if classes < 1:
                classes = int(inp.max()) + 1 if inp.numel() else 1
            if args:
                args = (classes, *args[1:])
            else:
                kwargs["num_classes"] = classes
        return inp, tuple(args), kwargs

    def _bitpattern_call(self, op, dtype, device):
        r"""Build valid raw-bit calls for each sampled signature and mask parity."""
        if op.name in POINTWISE_GEOMETRY | POINTWISE_CONDITIONAL:
            self._coverage_filter("raw_bits_require_original_geometry")
            return []
        key = (_op_id(op), _dtype_label(dtype))
        expanded = (
            ALL_SAMPLES
            or key in REFERENCE_INPUTS
            or key in POINTWISE_XFAIL | NONFLOAT_XFAIL | BACKWARD_XFAIL
            or (
                op.name == "float_power"
                and dtype in (torch.bfloat16, torch.float16, torch.float32)
            )
        )
        count = NUM_BITPATTERN_SAMPLES if expanded else 256
        if dtype in (torch.float16, torch.bfloat16):
            if expanded:
                x = _exhaustive_16bit(dtype, device)
            else:
                gen = torch.Generator(device=device).manual_seed(0)
                x = (
                    torch.randint(0, 65536, (count,), device=device, generator=gen)
                    .to(torch.int16)
                    .view(dtype)
                )
            y = x.flip(0)
        elif dtype == torch.float32:
            x = _sampled_fp32(count, device, seed=0)
            y = _sampled_fp32(count, device, seed=1)
        elif dtype == torch.float64 or dtype.is_complex:
            gen = torch.Generator(device=device).manual_seed(0)
            bits = torch.randint(
                0,
                256,
                (count * dtype.itemsize,),
                dtype=torch.uint8,
                device=device,
                generator=gen,
            )
            x = bits.view(dtype)
            y = x.flip(0)
        else:
            self._coverage_filter("no_raw_bit_generator_for_dtype")
            return []
        # Cross special values to cover missing pairs such as (+0, -0).
        sp = _SPECIALS.to(device=device, dtype=dtype)
        x = torch.cat([x, sp.repeat_interleave(sp.numel())])
        y = torch.cat([y, sp.repeat(sp.numel())])
        try:
            samples = self._sample_inputs(op, device, dtype)
        except Exception as error:
            self._coverage_filter(
                f"raw_bit_input_generation_error:{type(error).__name__}"
            )
            return []

        def sig(v):
            if not isinstance(v, torch.Tensor):
                return repr(v)
            return (
                v.dtype,
                v.device,
                tuple(v.shape),
                v.is_conj(),
                v.is_neg(),
                _scalar_tensor_signature(v) if v.numel() == 1 else None,
            )

        calls = []
        seen = set()
        single_input = None
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                self._coverage_filter("non_tensor_raw_bit_input")
                continue
            operands = (*sample.args, *sample.kwargs.values())
            inp = x
            if not expanded and not any(
                isinstance(t, torch.Tensor)
                for t in tree_leaves((sample.args, sample.kwargs))
            ):
                # With one tensor input, special-value pairs repeat the same call.
                if single_input is None:
                    single_input = (
                        torch.unique(
                            x.view(torch.uint8).reshape(-1, x.element_size()), dim=0
                        )
                        .reshape(-1)
                        .view(dtype)
                    )
                inp = single_input
            masked = any(
                isinstance(t, torch.Tensor) and t.dtype == torch.bool for t in operands
            )
            for parity in (0, 1) if masked else (0,):
                args = tuple(
                    _substitute(a, y, inp.numel(), parity) for a in sample.args
                )
                kwargs = {
                    k: _substitute(v, y, inp.numel(), parity)
                    for k, v in sample.kwargs.items()
                }
                key = (
                    parity,
                    tuple(sig(a) for a in args),
                    tuple(sorted((k, sig(v)) for k, v in kwargs.items())),
                )
                if key in seen:
                    self._coverage_filter("duplicate_raw_bit_signature")
                    continue
                seen.add(key)
                try:
                    with _native_polynomial_probe(op):
                        op.op(inp, *args, **kwargs)
                except Exception as error:
                    self._coverage_filter(
                        f"invalid_raw_bit_call:{type(error).__name__}"
                    )
                    continue  # e.g. per-channel PReLU weights
                calls.append((inp, args, kwargs))
        return calls

    def _collect_calls(self, device, op, dtype, *, backward=False):
        r"""Return (source, input, args, kwargs) calls for both sweeps."""
        key = (_op_id(op), _dtype_label(dtype))
        use_reference = ALL_SAMPLES or key in REFERENCE_INPUTS
        calls = []
        try:
            if use_reference:
                samples = list(
                    op.reference_inputs(device, dtype, requires_grad=backward)
                )
                calls = [("ref", *c) for c in self._build_calls(samples)]
            calls += [("bits", *c) for c in self._bitpattern_call(op, dtype, device)]
            if not use_reference:
                samples = self._sample_inputs(op, device, dtype, requires_grad=backward)
                if not ALL_SAMPLES:
                    samples = _representative_samples(samples)
                sample_calls = self._build_calls(samples)
                calls += [("sample", *c) for c in sample_calls]
        except Exception as e:
            self.fail(f"{_op_id(op)} {dtype} input generation failed: {e}")
        if not calls:
            self.skipTest("no usable sample")
        usable = []
        for tag, inp, args, kwargs in calls:
            # remainder.Scalar_Tensor has no eager derivative.
            if (
                backward
                and op.name == "__rmod__"
                and args
                and not isinstance(args[0], torch.Tensor)
            ):
                self._coverage_filter("scalar_remainder_has_no_eager_backward")
                continue
            call = self._pointwise_call(op, inp, args, kwargs)
            if call is None:
                self._coverage_filter("non_pointwise_mode")
                continue
            inp, args, kwargs = call
            detector = _RngOpDetector()
            with detector, _native_polynomial_probe(op):
                _pointwise_op(op)(inp, *args, **kwargs)
            if detector.has_rng and op.name not in POINTWISE_DROPOUT | {
                "nn.functional.rrelu"
            }:
                self._coverage_filter("rng_call")
                continue
            usable.append((tag, inp, args, kwargs))
        if not usable:
            self.skipTest("no deterministic pointwise sample")
        return usable

    def _require_kernel(self, tested, nothing_tested, op, dtype):
        self.assertGreater(tested, 0, nothing_tested)
        if self._coverage is not None and metrics.generated_kernel_count == 0:
            self._coverage.kernel_required = False

    def _audit_native_baseline(self, sweep, device, op, dtype, cfg, *, calls):
        from torch._functorch import config as aot_config
        from torch._inductor import ir

        def encode(value):
            if isinstance(value, ir.IRNode):
                return {
                    "dtype": str(value.get_dtype()),
                    "size": [str(s) for s in value.get_size()],
                    "device": str(value.get_device()),
                }
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            if isinstance(value, dict):
                return {str(k): encode(v) for k, v in value.items()}
            return repr(value)

        original_fallback = ir.FallbackKernel.codegen
        original_copy = ir.InplaceCopyFallback.codegen
        coverage = self._coverage
        numerical, metadata = set(), set()

        def signature(kernel, operator):
            args, kwargs = (
                kernel.unflatten_args(kernel.inputs, kernel.constant_args)
                if isinstance(kernel, ir.FallbackKernel)
                else (kernel.inputs, {})
            )
            # Storage order and compile counts can differ after decomposition.
            return json.dumps(
                {"op": operator, "args": encode(args), "kwargs": encode(kwargs)},
                sort_keys=True,
            )

        def fallback(kernel, wrapper):
            operator = str(kernel.op_overload)
            if isinstance(kernel, ir.ComplexView) or operator in (
                "aten.view_as_real.default",
                "aten.view_as_complex.default",
                "aten.view.dtype",
                "aten._conj.default",
                "aten._neg_view.default",
                "aten.alias.default",
            ):
                metadata.add(signature(kernel, operator))
            else:
                numerical.add(signature(kernel, operator))
            return original_fallback(kernel, wrapper)

        def copy(kernel, wrapper):
            numerical.add(signature(kernel, "aten.copy_.default"))
            return original_copy(kernel, wrapper)

        with (
            aot_config.patch(
                enable_autograd_cache=False, enable_remote_autograd_cache=False
            ),
            config.patch(fx_graph_cache=False, fx_graph_remote_cache=False),
            mock.patch.object(ir.FallbackKernel, "codegen", fallback),
            mock.patch.object(ir.InplaceCopyFallback, "codegen", copy),
        ):
            mismatches = sweep(device, op, dtype, cfg, calls=calls)
            strict_numerical, strict_metadata = numerical.copy(), metadata.copy()
            generated = (
                metrics.generated_kernel_count,
                metrics.generated_fallback_kernel_count,
            )
            if numerical:
                numerical.clear()
                metadata.clear()
                self._coverage = None
                try:
                    sweep(
                        device, op, dtype, {**cfg, "numerics": "default"}, calls=calls
                    )
                finally:
                    self._coverage = coverage
                    (
                        metrics.generated_kernel_count,
                        metrics.generated_fallback_kernel_count,
                    ) = generated
                self.assertFalse(
                    strict_numerical - numerical, sorted(strict_numerical - numerical)
                )
        if coverage is not None:
            coverage.native_baseline_verified = True
            coverage.numerical_fallbacks = len(strict_numerical)
            coverage.metadata_views = len(strict_metadata)
        return mismatches

    def _combined_calls(self, device, op, dtype, *, backward=False):
        with _native_polynomial_probe(op):
            try:
                calls = self._collect_calls(device, op, dtype, backward=backward)
            except unittest.SkipTest as error:
                self._coverage_filter(f"opinfo_samples_skipped:{error}")
                calls = []
            # Preserve the edge inputs previously generated in a separate seeded test.
            torch.manual_seed(0)
            calls += self._edge_calls(op, device, dtype, backward=backward)
        return calls

    def _assert_sample_ledgers(self, mismatches, op, dtype, what, xfail, xfail_name):
        ordinary = [row for row in mismatches if not row[0].startswith("edge_")]
        edges = [row for row in mismatches if row[0].startswith("edge_")]
        self._assert_ledger(ordinary, op, dtype, what, xfail, xfail_name)
        self._assert_ledger(edges, op, dtype, f"edges {what}", (), "")

    def _sweep(self, device, op, dtype, cfg, *, calls=None):
        if calls is None:
            calls = self._collect_calls(device, op, dtype)
        return self._audit_native_baseline(
            self._sweep_impl, device, op, dtype, cfg, calls=calls
        )

    def _sweep_impl(self, device, op, dtype, cfg, *, calls=None):
        r"""Return mismatching (source, index, shape, kwargs, kind) records."""
        if calls is None:
            calls = self._collect_calls(device, op, dtype)

        if op.name in NATIVE_POLYNOMIALS:
            return self._sweep_native_calls(device, op, dtype, cfg, calls)

        def fn(inp, args, kwargs):
            return _pointwise_op(op)(inp, *args, **kwargs)

        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            metrics.reset()
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                eager = fn(inp, args, kwargs)
                if not self._is_tensor_output(eager):
                    self._coverage_filter("non_tensor_output")
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, eager, op):
                    self._coverage_filter("output_shape_not_pointwise")
                    continue
                result = compiled(inp, args, kwargs)
                tested += 1
                equal = _outputs_equal(eager, result)
                kind = "" if equal else _diff_kind(eager, result)
                self._coverage_comparison(equal, kind, tag)
                if not equal:
                    mismatches.append(
                        (
                            tag,
                            idx,
                            [
                                tuple(t.shape)
                                for t in tree_leaves((inp, args, kwargs))
                                if isinstance(t, torch.Tensor)
                            ],
                            kwargs,
                            kind,
                        )
                    )

        if not mismatches:
            # A mismatch must not be turned into a no-kernel skip.
            self._require_kernel(tested, "no usable sample", op, dtype)
        return mismatches

    def _sweep_native_calls(self, device, op, dtype, cfg, calls):
        from torch._dynamo.backends.registry import lookup_backend
        from torch._inductor.graph import GraphLowering

        packet = NATIVE_POLYNOMIALS[op.name]
        observed = []
        wrappers = []

        def fn(inp, args, kwargs):
            return _pointwise_op(op)(inp, *args, **kwargs)

        def backend(gm, inputs):
            emitted = []
            with (
                _NativeCallProbe(packet, compiling=True),
                mock.patch.object(GraphLowering, "save_output_code", emitted.append),
            ):
                compiled = lookup_backend("inductor")(gm, inputs)

            def execute(*args):
                probe = _NativeCallProbe(packet)
                with probe:
                    result = compiled(*args)
                observed.append(probe)
                wrappers.extend(emitted)
                return result

            return execute

        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            metrics.reset()
            torch._dynamo.reset()
            compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                eager = _NativeCallProbe(packet)
                with eager:
                    expected = fn(inp, args, kwargs)
                observed.clear()
                wrappers.clear()
                result = compiled(inp, args, kwargs)
                equal = (
                    len(observed) == len(wrappers) == 1
                    and eager.matches(observed[0], expected, result)
                    and _native_call_wrapper_is_direct(wrappers[0], eager.frames[0][0])
                )
                if self._coverage is not None:
                    self._coverage.native_calls_compared += 1
                    self._coverage.native_calls_matched += int(equal)
                    self._coverage.native_calls_by_sample_kind[
                        "edges" if tag.startswith("edge_") else "opinfo"
                    ] += 1
                    self._coverage.native_calls_by_source[tag] += 1
                    self._coverage.generated_kernels = metrics.generated_kernel_count
                    self._coverage.generated_fallbacks = (
                        metrics.generated_fallback_kernel_count
                    )
                    if not equal:
                        self._coverage.mismatches["native_call"] += 1
                if not equal:
                    mismatches.append((tag, idx, [], kwargs, "native_call"))
        self.assertGreater(len(calls), 0)
        return mismatches

    def _assert_ledger(self, mismatches, op, dtype, what, xfail, xfail_name):
        r"""Require listed pairs to differ and unlisted pairs to match eager."""
        key = (_op_id(op), _dtype_label(dtype))
        if key in xfail:
            self.assertTrue(
                mismatches,
                f"{key} {what} now matches eager under strict numerics; "
                f"remove it from {xfail_name}.",
            )
        else:
            self.assertFalse(
                mismatches,
                f"{key} {what} differs from eager under strict numerics "
                f"on (source, index, shape, kwargs, kind): {mismatches}.",
            )

    @ops(POINTWISE_OPS)
    def test_pointwise_bitwise(self, device, dtype, op):
        calls = self._combined_calls(device, op, dtype)
        mismatches = self._sweep(device, op, dtype, POINTWISE_STRICT_CFG, calls=calls)
        self._assert_sample_ledgers(
            mismatches,
            op,
            dtype,
            "forward",
            POINTWISE_XFAIL | NONFLOAT_XFAIL,
            "POINTWISE_XFAIL/NONFLOAT_XFAIL",
        )

    @dtypes(torch.float32)
    @parametrize("output_count", (1, 2))
    def test_small_shapes_receive_all_upstream_edges(self, device, dtype, output_count):
        seen = []

        class Observe(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                outputs = tuple(x.clone() for _ in range(output_count))
                return outputs[0] if output_count == 1 else outputs

            @staticmethod
            def backward(ctx, *grads):
                seen.append(tuple(grad[0].detach().clone() for grad in grads))
                return sum(grads)

        class Op:
            name, variant_test_name = "clone", ""
            op = Observe.apply

        x = torch.ones(5, dtype=dtype, device=device).requires_grad_()
        with mock.patch(
            "torch.compile", return_value=lambda inp, args, kwargs: Observe.apply(inp)
        ):
            mismatches = self._sweep_backward(
                device,
                Op(),
                dtype,
                POINTWISE_STRICT_CFG,
                calls=[("edge_opinfo", x, (), {})],
            )
        self.assertFalse(mismatches)
        values = _pointwise_grad_values(dtype, device).view(torch.int32).tolist()
        expected = [
            tuple(
                values[(index + output_index) % len(values)]
                for output_index in range(output_count)
            )
            for index in range(len(values))
        ]
        actual = [
            tuple(grad.view(torch.int32).item() for grad in grads)
            for grads in seen[::2]
        ]
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(
            len(actual), len(expected) if ALL_SAMPLES else len(set(expected))
        )

    @dtypes(torch.float32)
    def test_bcomplex_cast_upstream_edges(self, device, dtype):
        output = torch.ones(5, dtype=dtype, device=device).to(torch.bcomplex32)
        grads = list(self._upstream_grads([output], edge=True))
        actual = torch.stack([grad[0][0] for grad in grads]).view(torch.bfloat16)
        values = _pointwise_grad_values(torch.bfloat16, device)
        zeros = torch.zeros_like(values)
        expected = torch.cat(
            (torch.stack((values, zeros), -1), torch.stack((zeros, values), -1))
        ).flatten()
        actual_bits = actual.view(torch.int32).tolist()
        expected_bits = expected.view(torch.int32).tolist()
        self.assertEqual(set(actual_bits), set(expected_bits))
        self.assertEqual(
            len(actual_bits),
            len(expected_bits) if ALL_SAMPLES else len(set(expected_bits)),
        )

    @dtypes(torch.float32)
    def test_repeated_input_grad(self, device, dtype):
        x = torch.tensor([-2.0, -0.0, 1.0], device=device, dtype=dtype)

        def fn(inp, args, kwargs):
            return inp * args[0] + kwargs["same"]

        _, grads = self._input_grads(fn, x, (x,), {"same": x}, torch.ones_like(x))
        self.assertEqual(len(grads), 1)
        self.assertEqual(grads[0], 2 * x + 1)

    @dtypes(torch.int32, torch.uint8)
    def test_integer_reverse_remainder_edges(self, device, dtype):
        op = next(o for o in POINTWISE_OPS if o.name == "__rmod__")
        calls = self._edge_calls(op, device, dtype)
        self.assertTrue(calls)
        for _, inp, args, kwargs in calls:
            self.assertTrue(bool((inp != 0).all()))
            cpu_args = tree_map(
                lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, args
            )
            _pointwise_op(op)(inp.cpu(), *cpu_args, **kwargs)

    @dtypes(torch.float32)
    def test_edge_samples_keep_valid_geometry(self, device, dtype):
        for name in (
            "outer",
            "view_as_complex",
            "nn.functional.binary_cross_entropy",
            "nn.functional.batch_norm",
        ):
            op = next(o for o in POINTWISE_OPS if o.name == name)
            calls = self._edge_calls(op, device, dtype, backward=True)
            self.assertTrue(calls)
            for _, inp, args, kwargs in calls:
                with self.subTest(op=name):
                    _pointwise_op(op)(inp, *args, **kwargs)
                    if name == "nn.functional.binary_cross_entropy":
                        self.assertTrue(bool(((inp >= 0) & (inp <= 1)).all()))
                    elif name == "nn.functional.batch_norm":
                        self.assertFalse(args[0].requires_grad)
                        self.assertFalse(args[1].requires_grad)
        values = _pointwise_edge_values(dtype)
        self.assertTrue(bool((torch.abs(values) == torch.finfo(dtype).max).any()))
        self.assertTrue(
            bool(((values.abs() > 0) & (values.abs() < torch.finfo(dtype).tiny)).any())
        )
        self.assertTrue(bool(((values == 0) & torch.signbit(values)).any()))

    def _edge_calls(self, op, device, dtype, *, backward=False):
        calls = []
        if op.name in ("logaddexp", "where") and dtype.is_complex:
            calls += _pointwise_edge_calls(op, device, dtype, backward=backward)
        if op.name in POINTWISE_EDGE_NAMES and dtype in POINTWISE_EDGE_DTYPES:
            calls += _pointwise_edge_calls(op, device, dtype, backward=backward)
        samples = self._sample_inputs(op, device, dtype, requires_grad=backward)
        candidates = []
        for inp, args, kwargs in self._build_calls(samples):
            if call := self._pointwise_call(op, inp, args, kwargs):
                tensors = [
                    t for t in tree_leaves(call[0]) if isinstance(t, torch.Tensor)
                ]
                size = sum(t.numel() for t in tensors)
                if 0 < size <= 1024:
                    candidates.append((size, call))
        if not candidates:
            self._coverage_filter("no_nonempty_edge_shape")
            return calls
        _, (inp, args, kwargs) = max(candidates, key=lambda candidate: candidate[0])
        tensors = [t for t in tree_leaves(inp) if isinstance(t, torch.Tensor)]
        for leaf_index, tensor in enumerate(tensors):
            if tensor.dtype != dtype or tensor.numel() == 0:
                continue
            values = _pointwise_edge_values(dtype)
            if op.name == "nn.functional.binary_cross_entropy":
                values = values[torch.isfinite(values) & (values >= 0) & (values <= 1)]
            elif op.name == "nn.functional.one_hot":
                classes = args[0] if args else kwargs["num_classes"]
                values = torch.tensor([0, classes - 1], dtype=dtype)
            elif op.name == "unravel_index":
                maximum = min(math.prod(args[0]) - 1, torch.iinfo(dtype).max)
                values = torch.tensor([0, 1, maximum - 1, maximum], dtype=dtype)
            elif op.name == "__rmod__" and not (
                dtype.is_floating_point or dtype.is_complex or dtype == torch.bool
            ):
                values = values[values != 0]
            # Preserve the original shapes and non-input parameters. In
            # particular, polynomial degrees and running variances stay valid.
            for offset in range(0, values.numel(), tensor.numel()):
                chunk = values[offset : offset + tensor.numel()]
                chunk = chunk.repeat(
                    (tensor.numel() + chunk.numel() - 1) // chunk.numel()
                )
                replacement = chunk[: tensor.numel()].reshape(tensor.shape).to(device)
                if tensor.is_conj():
                    replacement = replacement.conj()
                if tensor.is_neg():
                    replacement = torch._neg_view(replacement)
                replacement.requires_grad_(tensor.requires_grad)
                edge_input = tree_map(lambda t: replacement if t is tensor else t, inp)
                edge_args = args
                if (
                    not dtype.is_floating_point
                    and not dtype.is_complex
                    and dtype != torch.bool
                    and op.name in {"div", "floor_divide", "remainder", "fmod"}
                ):
                    # MIN / -1 is undefined for signed integer division.
                    edge_args = tuple(
                        torch.where(a == -1, torch.ones_like(a), a)
                        if isinstance(a, torch.Tensor)
                        else (1 if a == -1 else a)
                        for a in args
                    )
                calls.append(
                    (
                        f"edge_opinfo_{leaf_index}_{offset}",
                        edge_input,
                        edge_args,
                        kwargs,
                    )
                )
        if op.name == "to":
            x = _pointwise_edge_values(dtype).to(device)
            for target_dtype in sorted(op.supported_dtypes("cuda"), key=str):
                calls.append(
                    (f"edge_cast_{_dtype_label(target_dtype)}", x, (target_dtype,), {})
                )
        if op.name == "resolve_conj" and dtype.is_complex:
            calls += [
                (
                    "edge_conjugated",
                    torch.tensor([1 + 2j, -1 - 2j], device=device, dtype=dtype).conj(),
                    (),
                    {},
                )
            ]
        if op.name == "resolve_neg" and dtype != torch.bool:
            calls += [
                (
                    "edge_negative_view",
                    torch._neg_view(_pointwise_edge_values(dtype).to(device)),
                    (),
                    {},
                )
            ]
        return calls

    def _upstream_grads(self, outputs, *, edge):
        values = (
            [_pointwise_grad_values(out.dtype, out.device) for out in outputs]
            if edge
            else []
        )
        count = max(v.numel() for v in values) if edge else 1
        patterns = (
            [
                v.view(torch.uint8).reshape(v.numel(), v.element_size()).cpu().tolist()
                for v in values
            ]
            if edge and not ALL_SAMPLES
            else []
        )
        seen = set()
        for index in range(count):
            if patterns:
                # Deduplicate whole tuples so multiple outputs keep their pairings.
                key = tuple(
                    bytes(rows[(index + output_index) % len(rows)])
                    for output_index, rows in enumerate(patterns)
                )
                if key in seen:
                    continue
                seen.add(key)
            upstream = []
            for output_index, output in enumerate(outputs):
                if edge:
                    v = values[output_index]
                    grad = v[(index + output_index) % v.numel()].expand(output.shape)
                else:
                    gen = torch.Generator(device=output.device).manual_seed(
                        output_index
                    )
                    grad = torch.randn(
                        output.shape,
                        generator=gen,
                        device=output.device,
                        dtype=output.dtype,
                    )
                    flat = grad.reshape(-1)
                    if flat.numel() >= 4:
                        flat[0], flat[1] = 0.0, -0.0
                upstream.append(grad)
            yield upstream

    def _input_grads(
        self, call_fn, inp, args, kwargs, grad_output, *, respect_requires_grad=False
    ):
        leaves = []
        leaf_by_id = {}

        def leafify(t):
            if (
                isinstance(t, torch.Tensor)
                and (t.is_floating_point() or t.is_complex())
                and (t.requires_grad or not respect_requires_grad)
            ):
                if id(t) not in leaf_by_id:
                    leaf = t.detach().requires_grad_(True)
                    leaves.append(leaf)
                    leaf_by_id[id(t)] = leaf
                return leaf_by_id[id(t)]
            return t

        inp2, args2, kwargs2 = tree_map(leafify, (inp, args, kwargs))
        out = call_fn(inp2, args2, kwargs2)
        targets = _diff_outputs(out)
        upstream = (
            [grad_output] if isinstance(grad_output, torch.Tensor) else grad_output
        )
        if len(targets) != len(upstream):
            raise AssertionError(
                "every differentiable output needs an upstream gradient"
            )
        pairs = [(t, g) for t, g in zip(targets, upstream) if t.requires_grad]
        if not leaves or not pairs:
            return out, None
        targets, upstream = zip(*pairs)
        grads = torch.autograd.grad(
            targets, leaves, grad_outputs=upstream, allow_unused=True
        )
        return out, grads

    def _sweep_backward(self, device, op, dtype, cfg, *, calls=None):
        if calls is None:
            calls = self._collect_calls(device, op, dtype, backward=True)
        return self._audit_native_baseline(
            self._sweep_backward_impl, device, op, dtype, cfg, calls=calls
        )

    def _sweep_backward_impl(self, device, op, dtype, cfg, *, calls=None):
        r"""Compare input gradients with identical upstream gradients."""
        if calls is None:
            calls = self._collect_calls(device, op, dtype, backward=True)

        def fn(inp, args, kwargs):
            return _pointwise_op(op)(inp, *args, **kwargs)

        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            metrics.reset()
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                with torch.no_grad():
                    probe = fn(inp, args, kwargs)
                outputs = _diff_outputs(probe)
                if not outputs:
                    self._coverage_filter("no_differentiable_output_dtype")
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, probe, op):
                    self._coverage_filter("output_shape_not_pointwise")
                    continue
                for upstream in self._upstream_grads(
                    outputs, edge=tag.startswith("edge_")
                ):
                    respect_requires_grad = tag in ("ref", "sample") or tag.startswith(
                        "edge_opinfo_"
                    )
                    eager_out, eager_grads = self._input_grads(
                        fn,
                        inp,
                        args,
                        kwargs,
                        upstream,
                        respect_requires_grad=respect_requires_grad,
                    )
                    comp_out, comp_grads = self._input_grads(
                        compiled,
                        inp,
                        args,
                        kwargs,
                        upstream,
                        respect_requires_grad=respect_requires_grad,
                    )
                    tested += 1
                    if eager_grads is None:
                        equal = comp_grads is None and _outputs_equal(
                            eager_out, comp_out
                        )
                        kind = "" if equal else "gradient_contract"
                    elif comp_grads is None:
                        self._coverage_comparison(False, "shape", tag)
                        mismatches.append(
                            (
                                tag,
                                idx,
                                [
                                    tuple(t.shape)
                                    for t in tree_leaves(inp)
                                    if isinstance(t, torch.Tensor)
                                ],
                                kwargs,
                                "shape",
                            )
                        )
                        continue
                    else:
                        equal = _outputs_equal(eager_grads, comp_grads)
                        kind = "" if equal else _diff_kind(eager_grads, comp_grads)
                    self._coverage_comparison(equal, kind, tag)
                    if not equal:
                        mismatches.append(
                            (
                                tag,
                                idx,
                                [
                                    tuple(t.shape)
                                    for t in tree_leaves(inp)
                                    if isinstance(t, torch.Tensor)
                                ],
                                kwargs,
                                kind,
                            )
                        )

        if not mismatches:
            self._require_kernel(tested, "no differentiable sample", op, dtype)
        return mismatches

    @ops(BACKWARD_OPS, dtypes=OpDTypes.supported_backward)
    def test_pointwise_backward(self, device, dtype, op):
        calls = self._combined_calls(device, op, dtype, backward=True)
        mismatches = self._sweep_backward(
            device, op, dtype, POINTWISE_STRICT_CFG, calls=calls
        )
        self._assert_sample_ledgers(
            mismatches,
            op,
            dtype,
            "backward",
            BACKWARD_XFAIL,
            "BACKWARD_XFAIL",
        )


instantiate_device_type_tests(PointwiseStrictNumericsTest, globals(), only_for="cuda")

if directory := os.getenv("PYTORCH_POINTWISE_COVERAGE_DIR"):
    write_manifest(
        directory,
        globals().get("PointwiseStrictNumericsTestCUDA"),
        {
            "all_samples": ALL_SAMPLES,
            "sample_mode": "full" if ALL_SAMPLES else "compact",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_capability": torch.cuda.get_device_capability()
            if torch.cuda.is_available()
            else None,
            "strict_config": POINTWISE_STRICT_CFG,
            "selected_opinfo_ids": [_op_id(op) for op in POINTWISE_OPS],
        },
    )

if __name__ == "__main__":
    run_tests()
