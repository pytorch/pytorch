# Owner(s): ["module: inductor"]
"""Tests for strict numerics mode."""

import math
import os
import subprocess
import sys
import unittest
from unittest import mock


# Native reductions register during import, so enable the rollout first.
os.environ["PYTORCH_SUM_INNER_TREE"] = "1"

import torch
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
from torch.testing._internal.common_cuda import IS_SM100, IS_SM89, IS_SM90, SM90OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    OpDTypes,
    ops,
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
    TEST_CUTEDSL,
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


@instantiate_parametrized_tests
class StrictNumericsConfigTest(TestCase):
    @parametrize("numerics", NUMERICS_MODES)
    def test_config_patch_enables_eager_numerics(self, numerics):
        enabled = numerics in ("strict_pointwise", "strict")
        reduction_enabled = numerics in ("strict_reduction", "strict")
        with config.patch(_numerics_options("strict", False)):
            with config.patch(numerics=numerics):
                self.assertEqual(config.strict_pointwise, enabled)
                self.assertEqual(config.strict_reduction, reduction_enabled)
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
        self.assertEqual(
            actual.view(_BIT_VIEW[dtype]),
            torch.special.log_ndtr(x).view(_BIT_VIEW[dtype]),
        )

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
        self.assertEqual(actual.view(_BIT_VIEW[dtype]), fn(x).view(_BIT_VIEW[dtype]))

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
        self.assertEqual(result.view(torch.int32), op.op(x, y).view(torch.int32))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None,
    "requires NVIDIA CUDA and Triton",
)
class StrictNumericsCompileTest(TestCase):
    def _assert_bitwise_equal(self, actual, expected):
        from torch.utils._pytree import tree_map

        def bits(t):
            return t.view(_BIT_VIEW.get(t.dtype, t.dtype))

        self.assertEqual(tree_map(bits, actual), tree_map(bits, expected))

    @dtypes(torch.complex64, torch.complex128)
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

    @dtypes(torch.float32)
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
        # Unbacked geometry retains the existing decomposition; this checks
        # compilation and ordinary numerical agreement, not exact-bit equivalence.
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

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    @parametrize("operation", ("cast", "cube"))
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
            result = x.to(dtype) if operation == "cast" else x.pow(3)
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

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @torch._dynamo.config.patch(specialize_float=False)
    def test_addcdiv_options_gradients(self, device, dtype, numerics):
        inputs = [
            torch.randn(512, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        grad = torch.randn_like(inputs[0])

        def fn(a, b, c, value):
            return torch.addcdiv(a, b, c, value=value)

        compiled = torch.compile(fn, fullgraph=True, options={"numerics": numerics})
        for value in (0.7, 0.6, -0.7):
            expected = torch.autograd.grad(fn(*inputs, value), inputs, grad)
            actual = torch.autograd.grad(compiled(*inputs, value), inputs, grad)
            self._assert_bitwise_equal(actual, expected)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("upcast", (False, True))
    def test_fused_mul_nan_payload(self, device, dtype, numerics, upcast):
        values = _min_max_specials(dtype, device)
        nans = values[-4:]
        a = nans.repeat_interleave(values.numel())
        b = values.repeat(nans.numel())

        def fn(a, b):
            computed = -b.rsqrt()
            return a * computed, computed * a

        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={"numerics": numerics, "triton.codegen_upcast_to_fp32": upcast},
        )
        result, codes = run_and_get_code(compiled, a, b)
        self.assertIn("@triton.jit", "\n".join(codes))
        self._assert_bitwise_equal(result, fn(a, b))
        for scalar in (float("nan"), -float("nan"), 0.7):
            self._assert_bitwise_equal(compiled(scalar, nans), fn(scalar, nans))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_fused_sub_nan_payload(self, device, dtype, numerics):
        values = _min_max_specials(dtype, device)
        nans = values[-4:]
        a = nans.repeat_interleave(values.numel())
        b = values.repeat(nans.numel())

        def fn(a, b):
            computed = -b.rsqrt()
            return a - computed, computed - a

        compiled = torch.compile(fn, fullgraph=True, options={"numerics": numerics})
        result, codes = run_and_get_code(compiled, a, b)
        self.assertIn("@triton.jit", "\n".join(codes))
        self.assertTrue(_outputs_equal(result, fn(a, b)))

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
    @parametrize(
        "layout", ("aligned", "misaligned", "strided", "expanded", "transposed")
    )
    def test_mul_nan_payload_full_block_and_tail(self, device, dtype, numerics, layout):
        values = _min_max_specials(dtype, device)

        def operand(value):
            if layout == "misaligned":
                return value.repeat(1026)[1:]
            if layout == "strided":
                return value.repeat(2050)[::2]
            if layout == "expanded":
                return value.expand(1025)
            result = value.repeat(1025)
            return result.reshape(205, 5).t() if layout == "transposed" else result

        a, b = operand(values[7:8]), operand(values[6:7])
        compiled = torch.compile(
            torch.mul, fullgraph=True, options={"numerics": numerics}
        )
        actual, codes = run_and_get_code(compiled, a, b)
        self.assertIn("@triton.jit", "\n".join(codes))
        self.assertTrue(_outputs_equal(actual, a * b))

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize("cudagraphs", (False, True))
    @parametrize("input_dtype", (torch.float64, torch.int64, torch.float32))
    def test_mul_nan_payload_input_provenance(
        self, device, dtype, numerics, cudagraphs, input_dtype
    ):
        values = _min_max_specials(dtype, device)
        shared = torch.empty(2052, device=device, dtype=dtype)
        shared[:1026] = values[7]
        shared[1026:] = values[6]
        a, b = shared[:1025].view(input_dtype), shared[1026:2051].view(input_dtype)

        def fn(a, b):
            return a.view(dtype) * b.view(dtype)

        counter = CompileCounterWithBackend("inductor")
        skips_before = counters["inductor"]["cudagraph_skips"]
        with config.patch({"numerics": numerics, "triton.cudagraphs": cudagraphs}):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            actual, codes = run_and_get_code(compiled, a, b)
            self.assertIn("@triton.jit", "\n".join(codes))
            self.assertTrue(_outputs_equal(actual, fn(a, b)))
            del actual
            # Read-only input aliasing and pointer alignment can change without guards.
            for left, right in ((0, 1), (0, 0), (1, 1), (0, 0), (1, 1)):
                a = values[7:8].repeat(1025 + left)[left:].view(input_dtype)
                b = values[6:7].repeat(1025 + right)[right:].view(input_dtype)
                torch.compiler.cudagraph_mark_step_begin()
                actual = compiled(a, b)
                self.assertTrue(_outputs_equal(actual, fn(a, b)))
                del actual
        self.assertEqual(counter.frame_count, 1)
        if cudagraphs:
            self.assertGreater(counters["inductor"]["cudagraph_skips"], skips_before)

    @dtypes(torch.float64)
    @parametrize("numerics", ("strict_pointwise", "strict"))
    @parametrize(
        "hop", ("cond", "nested_cond", "invoke", "while_loop", "aliased_clone")
    )
    def test_mul_nan_payload_subgraph_input_alignment(
        self, device, dtype, numerics, hop
    ):
        values = _min_max_specials(dtype, device)

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
        skips_before = counters["inductor"]["cudagraph_skips"]
        with config.patch({"numerics": numerics, "triton.cudagraphs": True}):
            compiled = torch.compile(fn, fullgraph=True, backend=counter)
            for offset in (0, 1, 0):
                a = values[7:8].repeat(1025 + offset)[offset:]
                b = values[6:7].repeat(1025)
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
                    self.assertEqual(
                        actual.view(torch.int64), expected.view(torch.int64)
                    )
                    if partial_write:
                        grad = values[6:7].repeat(1025)
                        expected_grad = torch.autograd.grad(
                            expected, eager_args[1], grad
                        )[0]
                        actual_grad = torch.autograd.grad(actual, actual_args[1], grad)[
                            0
                        ]
                        self.assertEqual(
                            actual_grad.view(torch.int64),
                            expected_grad.view(torch.int64),
                        )
                        self.assertEqual(
                            actual_args[0].view(torch.int64),
                            eager_args[0].view(torch.int64),
                        )
                    if input_stride == 2:
                        self.assertEqual(
                            actual_args[0]._base.view(torch.int64),
                            eager_args[0]._base.view(torch.int64),
                        )
                    if producer == "out":
                        for index in (0, 2):
                            self.assertEqual(
                                actual_args[index].view(torch.uint8),
                                eager_args[index].view(torch.uint8),
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
                        self.assertEqual(
                            actual_value.view(torch.int64),
                            expected_value.view(torch.int64),
                        )
                    expected_grad = torch.autograd.grad(
                        expected[2], eager_args[2], torch.ones_like(eager_args[2])
                    )[0]
                    actual_grad = torch.autograd.grad(
                        actual[2], actual_args[2], torch.ones_like(actual_args[2])
                    )[0]
                    self.assertEqual(
                        actual_grad.view(torch.int64), expected_grad.view(torch.int64)
                    )
                    self.assertEqual(
                        actual_args[0]._base.view(torch.int64),
                        eager_args[0]._base.view(torch.int64),
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
                self.assertEqual(actual.view(torch.int64), expected.view(torch.int64))
                self.assertEqual(a.view(torch.int64), eager_a.view(torch.int64))
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
    def test_mul_nan_payload_observable_clone(self, device, dtype, numerics, producer):
        from torch._inductor.graph import GraphLowering

        values = torch.tensor(
            [0x7FF8000000000011, 0x7FF8000000000001, 0x7FF8000000000033],
            dtype=torch.int64,
            device=device,
        ).view(dtype)

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
                self.assertEqual(actual.view(torch.int64), eager.view(torch.int64))
                self.assertEqual(
                    args[0].view(torch.int64), eager_args[0].view(torch.int64)
                )
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
                self.assertEqual(
                    grads[0].view(torch.int64), expected_grad.view(torch.int64)
                )
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
                    self.assertEqual(output.view(torch.int64), eager.view(torch.int64))
                    actual = torch.autograd.grad(output, args[1], grad_output)[0]
                    self.assertEqual(
                        actual.view(torch.int64), expected.view(torch.int64)
                    )
                    for index in (0, 2) if producer == "out" else (0,):
                        self.assertEqual(
                            args[index].view(torch.uint8),
                            eager_args[index].view(torch.uint8),
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
        [op for op in op_db if op.name in ("abs", "neg", "angle")],
        allowed_dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
    )
    @parametrize("numerics", ("strict_pointwise", "strict"))
    def test_unary_integer(self, device, dtype, op, numerics):
        limits = torch.iinfo(dtype)
        x = torch.tensor([limits.min, 0, 1, limits.max], dtype=dtype, device=device)
        compiled = torch.compile(op.op, fullgraph=True, options={"numerics": numerics})
        self.assertEqual(compiled(x), op.op(x))

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
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32),
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
                "nn.functional.relu",
                "nn.functional.relu6",
                "nn.functional.hardtanh",
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

        bounds = torch.tensor([-1.0, 0.0, 1.0, 6.0], device=device, dtype=dtype)
        x = torch.cat(
            (
                x,
                bounds,
                torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
                torch.nextafter(bounds, torch.full_like(bounds, float("inf"))),
            )
        )

        x.requires_grad_()
        lower = -1 if op.name in ("clamp", "nn.functional.hardtanh") else 0

        def fn(x):
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
        expected_grad = torch.autograd.grad(expected, x, torch.ones_like(x))[0]
        result_grad = torch.autograd.grad(result, x, torch.ones_like(x))[0]
        self._assert_bitwise_equal(result_grad, expected_grad)

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
                    self.assertEqual(
                        actual.detach().view(torch.int64),
                        expected.detach().view(torch.int64),
                    )
                    self.assertEqual(x.view(torch.int64), eager_x.view(torch.int64))
                    self.assertEqual(z.view(torch.int64), eager_z.view(torch.int64))
                    if requires_grad:
                        gradient = torch.arange(2048, device=device, dtype=dtype) / 8
                        expected_grad = torch.autograd.grad(
                            expected, eager_y, gradient
                        )[0]
                        actual_grad = torch.autograd.grad(actual, y, gradient)[0]
                        self.assertEqual(
                            actual_grad.view(torch.int64),
                            expected_grad.view(torch.int64),
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
                    self.assertEqual(
                        actual.detach().view(torch.int64),
                        expected.detach().view(torch.int64),
                    )
                    self.assertEqual(x.view(torch.int64), eager_x.view(torch.int64))
                    self.assertEqual(z.view(torch.int64), eager_z.view(torch.int64))
                    gradient = torch.arange(2048, device=device, dtype=dtype) / 8
                    expected_grad = torch.autograd.grad(expected, eager_y, gradient)[0]
                    actual_grad = torch.autograd.grad(actual, y, gradient)[0]
                    self.assertEqual(
                        actual_grad.view(torch.int64),
                        expected_grad.view(torch.int64),
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
                        self.assertEqual(
                            actual.view(torch.int64), expected.view(torch.int64)
                        )
                        self.assertEqual(
                            actual_inputs[0].view(torch.int64),
                            expected_inputs[0].view(torch.int64),
                        )


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering(),
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
        if not TEST_CUTEDSL:
            return
        self.assertEqual(
            eager.contiguous().reshape(-1).view(torch.uint8),
            result.contiguous().reshape(-1).view(torch.uint8),
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


def _pointwise_edge_calls(op, device, dtype):
    calls = []

    def add(label, inp, args=(), kwargs=None):
        calls.append((f"edge_{label}", inp, args, kwargs or {}))

    if op.name in POINTWISE_BOUNDARIES:
        x = _pointwise_boundary_values(POINTWISE_BOUNDARIES[op.name], dtype, device)
        add("boundary", x)
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
    r"""Classify mismatches as shape, value, NaN payload, or signed zero."""
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
    differ = a.contiguous().view(int_dtype) != b.contiguous().view(int_dtype)
    nan = differ & torch.isnan(a) & torch.isnan(b)
    zero = differ & (a == 0) & (b == 0)
    if not bool((differ == (nan | zero)).all()):
        return "value"
    parts = []
    if bool(nan.any()):
        parts.append("nan-payload")
    if bool(zero.any()):
        parts.append("signed-zero")
    return "+".join(parts) or "value"


def _diff_outputs(out):
    return [
        t
        for t in tree_leaves(out)
        if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())
    ]


def _outputs_equal(a, b):
    r"""Compare raw bytes, including NaN payloads and signed zeros."""
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
    a, b = a.resolve_conj().resolve_neg(), b.resolve_conj().resolve_neg()
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


def _representative_samples(samples):
    def signature(t):
        if not isinstance(t, torch.Tensor):
            return t
        return (
            t.dtype,
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
            t.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .reshape(-1)
            .clone(memory_format=torch.contiguous_format)
            .view(torch.uint8)
            .tolist()
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
POINTWISE_XFAIL = frozenset(
    {
        ("float_power", "bfloat16"),
        ("float_power", "float16"),
        ("float_power", "float32"),
    }
)

BACKWARD_XFAIL: frozenset[tuple[str, str]] = frozenset()

NONFLOAT_XFAIL = frozenset(
    {
        ("nn_functional_silu_complex", "complex128"),
    }
)


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


# Skip raw-bit sweeps for prohibitively slow value-dependent series.
BITPATTERN_SLOW = frozenset(
    {
        "polygamma",
        "special.polygamma",
        "special.shifted_chebyshev_polynomial_t",
        "special.shifted_chebyshev_polynomial_u",
        "special.shifted_chebyshev_polynomial_v",
        "special.shifted_chebyshev_polynomial_w",
        "special.laguerre_polynomial_l",
        "special.legendre_polynomial_p",
    }
)


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
            if op.name in POINTWISE_REPRESENTATIONS | POINTWISE_DROPOUT
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
        with self.assertRaisesRegex(unittest.SkipTest, "no usable sample"):
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

    def _build_calls(self, samples, dtype):
        r"""Build calls without flattening so reference layouts exercise codegen."""
        calls = []
        for sample in samples:
            tensors = tree_leaves((sample.input, sample.args, sample.kwargs))
            if not any(isinstance(t, torch.Tensor) for t in tree_leaves(sample.input)):
                self._coverage_filter("non_tensor_input")
                continue
            if any(isinstance(t, torch.Tensor) and t.dtype == dtype for t in tensors):
                calls.append((sample.input, tuple(sample.args), dict(sample.kwargs)))
            else:
                self._coverage_filter("input_dtype_not_selected")
        return calls

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
        if op.name in BITPATTERN_SLOW or "polynomial" in op.name:
            self._coverage_filter("raw_bit_sweep_disabled_for_slow_series")
            return []
        key = (_op_id(op), _dtype_label(dtype))
        expanded = (
            ALL_SAMPLES
            or key in REFERENCE_INPUTS
            or key in POINTWISE_XFAIL | NONFLOAT_XFAIL | BACKWARD_XFAIL
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
            # Keep scalar-tensor and vector-tensor signatures distinct.
            return tuple(v.shape) if isinstance(v, torch.Tensor) else repr(v)

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
                calls = [("ref", *c) for c in self._build_calls(samples, dtype)]
            calls += [("bits", *c) for c in self._bitpattern_call(op, dtype, device)]
            if not use_reference:
                samples = self._sample_inputs(op, device, dtype, requires_grad=backward)
                if not ALL_SAMPLES:
                    samples = _representative_samples(samples)
                sample_calls = self._build_calls(samples, dtype)
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
            with detector:
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

    def _require_kernel(self, tested, nothing_tested, op):
        # ATen fallbacks compare eager against itself, so do not count them as coverage.
        if tested == 0:
            self.skipTest(nothing_tested)
        if (
            metrics.generated_kernel_count == 0
            and op.name not in POINTWISE_REPRESENTATIONS | POINTWISE_DROPOUT
        ):
            self.skipTest("no Triton kernel generated (op falls back to ATen)")

    def _combined_calls(self, device, op, dtype, *, backward=False):
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
        r"""Return mismatching (source, index, shape, kwargs, kind) records."""
        if calls is None:
            calls = self._collect_calls(device, op, dtype)

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
                            [tuple(t.shape) for t in tree_leaves(inp)],
                            kwargs,
                            kind,
                        )
                    )

        if not mismatches:
            # A mismatch must not be turned into a no-kernel skip.
            self._require_kernel(tested, "no usable sample", op)
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
        if op.name in POINTWISE_EDGE_NAMES and dtype in POINTWISE_EDGE_DTYPES:
            calls += _pointwise_edge_calls(op, device, dtype)
        samples = self._sample_inputs(op, device, dtype, requires_grad=backward)
        candidates = []
        for inp, args, kwargs in self._build_calls(samples, dtype):
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
                    _, eager_grads = self._input_grads(
                        fn,
                        inp,
                        args,
                        kwargs,
                        upstream,
                        respect_requires_grad=respect_requires_grad,
                    )
                    if eager_grads is None:
                        self._coverage_filter("no_input_gradients")
                        continue
                    _, comp_grads = self._input_grads(
                        compiled,
                        inp,
                        args,
                        kwargs,
                        upstream,
                        respect_requires_grad=respect_requires_grad,
                    )
                    tested += 1
                    if comp_grads is None:
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
            self._require_kernel(tested, "no differentiable sample", op)
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
