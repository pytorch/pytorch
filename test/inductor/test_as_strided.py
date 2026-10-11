# Owner(s): ["module: inductor"]
import re
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.fx_passes.as_strided import canonicalize_as_strided
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_utils import parametrize, subtest
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten


class TestAsStrided(TestCase):
    @parametrize("offset", [None, 0, 2])
    @parametrize("copy", [False, True])
    def test_view_chain(self, device, offset, copy):
        op = aten.as_strided_copy.default if copy else aten.as_strided.default

        def fn(x):
            base = x + 1
            view = base.t()[1:]
            return base, op(view, (3,), (1,), offset)

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        gm = make_fx(fn, tracing_mode="fake")(x)
        expected = gm(x)
        bases = OrderedSet()
        with patch.object(
            gm.graph,
            "materialize_symints",
            side_effect=AssertionError("Static parameters must not scan for symbols"),
        ):
            self.assertEqual(canonicalize_as_strided(gm, storage_bases=bases), 1)
        node = gm.graph.find_nodes(op="call_function", target=op)[0]
        self.assertEqual(node.args[0].target, aten.add.Tensor)
        self.assertEqual(bases, {node.args[0]})
        self.assertEqual(node.args[3], 1 if offset is None else offset)
        actual = gm(x)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._C._is_alias_of(actual[0], actual[1]), not copy)
        bases.clear()
        self.assertEqual(canonicalize_as_strided(gm, storage_bases=bases), 0)
        self.assertEqual(bases, {node.args[0]})

    @parametrize("count", [1, 8])
    def test_symbolic_offset(self, device, count):
        def fn(x):
            base = x + 1
            return tuple(
                base[x.shape[0] // 2 + i :, 1:].as_strided((2,), (1,))
                for i in range(count)
            )

        x = torch.arange(96, device=device, dtype=torch.float32).reshape(24, 4)
        gm = make_fx(fn, tracing_mode="symbolic")(x)
        nodes = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)
        sources = [node.args[0] for node in nodes]
        with patch.object(
            gm.graph,
            "materialize_symints",
            side_effect=AssertionError("Symbolic offsets must not scan for symbols"),
        ):
            self.assertEqual(canonicalize_as_strided(gm), count)
        for node, source in zip(nodes, sources):
            offset = node.args[3]
            self.assertIsInstance(offset, torch.fx.Node)
            self.assertEqual(offset.target, aten.sym_storage_offset.default)
            self.assertEqual(offset.args, (source,))
        self.assertEqual(canonicalize_as_strided(gm), 0)
        other = torch.arange(128, device=device, dtype=torch.float32).reshape(32, 4)
        self.assertEqual(gm(other), fn(other))

    def test_unfold_view(self, device):
        def fn(x):
            base = x + 1
            return base.unfold(1, 2, 1).as_strided((3,), (1,))

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        gm = make_fx(fn, tracing_mode="fake")(x)
        expected = gm(x)
        self.assertEqual(canonicalize_as_strided(gm), 1)
        node = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)[
            0
        ]
        self.assertEqual(node.args[0].target, aten.add.Tensor)
        self.assertEqual(gm(x), expected)

    @parametrize("reshape_copy", [False, True])
    def test_copy_boundary(self, device, reshape_copy):
        def fn(x):
            base = x + 1
            copied = base.t().reshape(-1) if reshape_copy else base.clone().view(-1)
            return copied[1:].as_strided((3,), (1,), 0)

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        gm = make_fx(fn, tracing_mode="fake")(x)
        expected = gm(x)
        self.assertEqual(canonicalize_as_strided(gm), 1)
        node = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)[
            0
        ]
        self.assertEqual(node.args[0].target, aten.clone.default)
        self.assertEqual(gm(x), expected)

    @parametrize("kind", ["conj", "neg"])
    @parametrize("slice_view", [False, True])
    def test_special_view_boundary(self, device, kind, slice_view):
        def fn(x):
            if kind == "conj":
                view = x.conj()
            else:
                view = torch._neg_view(x)
            if slice_view:
                view = view[::2]
            return view.as_strided((2,), (1,))

        x = torch.ones(4, device=device, dtype=torch.complex64)
        gm = make_fx(fn, tracing_mode="fake")(x)
        node = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)[
            0
        ]
        args = node.args
        bases = OrderedSet()
        self.assertEqual(canonicalize_as_strided(gm, storage_bases=bases), 0)
        self.assertEqual(bases, set())
        self.assertEqual(node.args, args)
        self.assertEqual(gm(x), fn(x))

    def test_reads_around_mutation(self, device):
        def fn(x, src):
            base = x + 1
            view = base.t()
            before = view.as_strided((4,), (1,), 0).clone()
            view.copy_(src)
            after = view.as_strided((4,), (1,), 0)
            return before, after, base

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        src = torch.full((4, 3), 20.0, device=device)
        gm = make_fx(fn, tracing_mode="fake")(x, src)
        expected = gm(x, src)
        nodes = list(gm.graph.nodes)
        self.assertEqual(canonicalize_as_strided(gm), 1)
        views = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)
        self.assertEqual(views[0].args[0].target, aten.add.Tensor)
        self.assertEqual(views[1].args[0].target, aten.copy_.default)
        self.assertEqual(list(gm.graph.nodes), nodes)
        actual = gm(x, src)
        self.assertEqual(actual, expected)
        self.assertTrue(torch._C._is_alias_of(actual[1], actual[2]))
        self.assertEqual(torch.compile(fn, fullgraph=True)(x, src), expected)

    def test_mutation_boundary(self, device):
        def fn(x, src):
            base = x.clone()
            updated = base.copy_(src)
            return updated.t().as_strided((3,), (1,), 0)

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        src = torch.full_like(x, 20.0)
        gm = make_fx(fn, tracing_mode="fake")(x, src)
        expected = gm(x, src)
        self.assertEqual(canonicalize_as_strided(gm), 1)
        node = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)[
            0
        ]
        self.assertEqual(node.args[0].target, aten.copy_.default)
        self.assertEqual(gm(x, src), expected)

    @parametrize("input_base", [False, True])
    def test_write_through_as_strided(self, device, input_base):
        def fn(x, src):
            base = x if input_base else x + 1
            view = base[:, 1:].as_strided((3,), (1,))
            view.copy_(src)
            return base

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        src = torch.full((3,), 20.0, device=device)
        gm = make_fx(fn, tracing_mode="fake")(x, src)
        expected_input = x.clone()
        expected = gm(expected_input, src)
        self.assertEqual(canonicalize_as_strided(gm), 1)
        actual_input = x.clone()
        self.assertEqual(gm(actual_input, src), expected)
        self.assertEqual(actual_input, expected_input)

    @parametrize("input_base", [False, True])
    def test_compile_write_through_as_strided(self, device, input_base):
        def fn(x, src):
            base = x if input_base else x + 1
            view = base.as_strided((3,), (1,), 1)
            view.copy_(src)
            return base

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        src = torch.full((3,), 20.0, device=device)
        expected_input = x.clone()
        expected = fn(expected_input, src)
        compiled_input = x.clone()
        self.assertEqual(
            torch.compile(fn, fullgraph=True)(compiled_input, src), expected
        )
        self.assertEqual(compiled_input, expected_input)

    @parametrize("post_grad", [False, True])
    @parametrize("offset", [None, 0])
    @parametrize("copy", [False, True])
    def test_compile_view_of_producer(self, device, post_grad, offset, copy):
        op = torch.as_strided_copy if copy else torch.as_strided

        def fn(x):
            base = x.t() + 1
            return op(base[1:], (3,), (1,), offset)

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        with config.patch(use_post_grad_passes=post_grad):
            self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x))

    @parametrize("input_base", [False, True])
    @parametrize("offset", [None, 0, 2])
    def test_compile_copy_nested_diagonal(self, device, input_base, offset):
        def fn(x):
            base = x if input_base else x + 1
            view = base.diagonal(1, 0, 1).diagonal()
            return base, torch.as_strided_copy(view, (1,), (1,), offset)

        x = torch.arange(13, device=device, dtype=torch.float32)[5:].reshape(2, 2, 2)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, fn(x))
        self.assertFalse(torch._C._is_alias_of(actual[0], actual[1]))

    @parametrize("input_base", [False, True])
    @parametrize("offset", [None, 0, 2])
    def test_compile_copy_symbolic_offset(self, device, input_base, offset):
        def fn(x):
            base = x if input_base else x + 1
            view = base[x.shape[0] // 2 :, 1:]
            return base, torch.as_strided_copy(view, (2,), (1,), offset)

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for rows in (6, 8):
            with self.subTest(rows=rows):
                x = torch.arange(5 + rows * 4, device=device, dtype=torch.float32)
                x = x[5:].reshape(rows, 4)
                actual = compiled(x)
                self.assertEqual(actual, fn(x))
                self.assertFalse(torch._C._is_alias_of(actual[0], actual[1]))

    def test_compile_repeat_view(self, device):
        def fn(x):
            base = x.repeat((3, 1))
            return base[:, ::2].as_strided((6, 2), base.stride())

        x = torch.arange(4, device=device, dtype=torch.float32).reshape(2, 2)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x))

    def test_compile_symbolic_offset(self, device):
        def fn(x):
            base = x + 1
            view = base[x.shape[0] // 2 :, 1:]
            return view.as_strided((2,), (1,)) + 1

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for rows in (6, 8):
            with self.subTest(rows=rows):
                x = torch.arange(rows * 4, device=device, dtype=torch.float32)
                x = x.reshape(rows, 4)
                self.assertEqual(compiled(x), fn(x))

    def test_compile_symbolic_input_offset(self, device):
        def fn(x):
            view = x[x.shape[0] // 2 :, 1:]
            return view.as_strided((2,), (1,)) + 1

        input_offset = 3
        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for rows in (6, 8):
            with self.subTest(rows=rows):
                x = torch.arange(
                    input_offset + rows * 4, device=device, dtype=torch.float32
                )
                x = x[input_offset:].reshape(rows, 4)
                self.assertEqual(compiled(x), fn(x))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_compile_unbacked_offsets(self, device):
        def fn(x, a, b):
            first = a.item()
            torch._check(first >= 0)
            torch._check(first <= x.shape[0] - 4)
            v1 = x.narrow(0, first, 4)
            out1 = v1.as_strided((2,), (1,)) + 1
            second = b.item()
            torch._check(second >= 0)
            torch._check(second <= x.shape[0] - 4)
            v2 = x.narrow(0, second, 4)
            return out1, v2.as_strided((2,), (1,)) + 2

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        x = torch.arange(40, dtype=torch.float32, device=device)[5:37]
        for first, second in ((2, 7), (9, 3)):
            with self.subTest(first=first, second=second):
                a = torch.tensor(first, device=device)
                b = torch.tensor(second, device=device)
                self.assertEqual(compiled(x, a, b), fn(x, a, b))

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_compile_unbacked_producer_offset(self, device):
        def fn(x):
            base = x.nonzero().flatten()
            torch._check(base.numel() >= 4)
            view = base[base.numel() // 2 :]
            return view.as_strided((2,), (1,)) + 1

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for count in (8, 12):
            with self.subTest(count=count):
                x = torch.arange(32, device=device) < count
                self.assertEqual(compiled(x), fn(x))

    @parametrize("copy", [False, True])
    def test_compile_missing_input_metadata(self, device, copy):
        op = torch.as_strided_copy if copy else torch.as_strided

        def fn(x):
            return op(x + 1, (2,), (1,), 1) + 2

        def remove_metadata(graph):
            for n in graph.nodes:
                if n.target in (aten.as_strided.default, aten.as_strided_copy.default):
                    n.args[0].meta.pop("val", None)

        x = torch.arange(8, dtype=torch.float32, device=device)
        with config.patch(
            post_grad_custom_post_pass=remove_metadata, force_disable_caches=True
        ):
            self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x))

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @parametrize("copy", [False, True])
    def test_compile_unbacked_base_strides(self, device, copy):
        op = torch.as_strided_copy if copy else torch.as_strided

        def fn(x):
            indices = x.nonzero()
            torch._check(indices.shape[0] >= 3)
            base = indices.t().contiguous() + 1
            return op(base[1:], (3,), (1,), 0) + 2

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for count in (5, 7):
            with self.subTest(count=count):
                x = torch.arange(24, device=device).reshape(6, 4) < count
                self.assertEqual(compiled(x), fn(x))

    @parametrize(
        "src_dtype,dst_dtype,tracing_mode,offset",
        [
            (torch.int32, torch.int16, "fake", None),
            (torch.int32, torch.int16, "symbolic", None),
            (torch.int16, torch.int32, "fake", None),
            (torch.int16, torch.int32, "symbolic", None),
            (torch.int32, torch.float32, "fake", None),
            (torch.int32, torch.float32, "symbolic", None),
            (torch.int32, torch.int16, "fake", 0),
            (torch.int32, torch.int16, "fake", 2),
        ],
    )
    def test_dtype_view_boundary(
        self, device, src_dtype, dst_dtype, tracing_mode, offset
    ):
        def fn(x):
            base = x + 1
            typed = base[::2].view(dst_dtype)
            view = typed[:, 1:]
            return base, typed, view.as_strided((3,), (1,), offset)

        x = torch.arange(32, dtype=src_dtype, device=device).reshape(4, 8)
        gm = make_fx(fn, tracing_mode=tracing_mode)(x)
        out = gm.graph.find_nodes(op="call_function", target=aten.as_strided.default)[0]
        out_args = out.args
        nodes = list(gm.graph.nodes)
        bases = OrderedSet()
        self.assertEqual(canonicalize_as_strided(gm, storage_bases=bases), 0)
        self.assertEqual(bases, set())
        self.assertEqual(out.args, out_args)
        self.assertEqual(list(gm.graph.nodes), nodes)
        for width in (8, 12) if tracing_mode == "symbolic" else (8,):
            with self.subTest(width=width):
                other = torch.arange(4 * width, dtype=src_dtype, device=device)
                other = other.reshape(4, width)
                actual = gm(other)
                self.assertEqual(actual, fn(other))
                self.assertTrue(torch._C._is_alias_of(actual[0], actual[1]))
                self.assertTrue(torch._C._is_alias_of(actual[0], actual[2]))

    @parametrize("copy", [False, True])
    @parametrize("dynamic", [False, True])
    def test_compile_complex_view_default_offset(self, device, copy, dynamic):
        op = torch.as_strided_copy if copy else torch.as_strided

        def fn(x):
            view = x.view(torch.float32)[2:]
            return op(view, (2,), (1,)) + 1

        real = torch.arange(10, dtype=torch.float32, device=device)
        base = torch.complex(real, real + 100)
        compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
        for offset in (1, 3):
            with self.subTest(offset=offset):
                x = base[offset:]
                self.assertEqual(compiled(x), fn(x))

    @parametrize("copy", [False, True])
    @parametrize("offset", [None, 0, 12])
    def test_compile_frozen_view_offset(self, device, copy, offset):
        op = torch.as_strided_copy if copy else torch.as_strided

        class Model(torch.nn.Module):
            def __init__(self, buffer):
                super().__init__()
                self.register_buffer("b", buffer)

            def forward(self, x):
                # Keep the view chain from being constant-folded during freezing.
                view = self.b[x.shape[0] :]
                return op(view, (3,), (1,), offset) + x[:3]

        buffer = torch.arange(64, dtype=torch.float32, device=device)[10:]
        model = Model(buffer).eval()
        with torch.no_grad(), config.patch(freezing=True):
            compiled = torch.compile(model, fullgraph=True, dynamic=True)
            for size in (6, 8):
                with self.subTest(size=size):
                    x = torch.ones(size, device=device)
                    self.assertEqual(compiled(x), model(x))

    @parametrize("dtypes", [(torch.int32, torch.int16), (torch.int16, torch.int32)])
    @parametrize("default_offset", [False, True])
    def test_compile_dtype_input_storage(self, device, dtypes, default_offset):
        src_dtype, dst_dtype = dtypes
        src_bytes = torch.iinfo(src_dtype).bits // 8
        dst_bytes = torch.iinfo(dst_dtype).bits // 8
        offset = None if default_offset else 8 * src_bytes // dst_bytes
        stride = max(1, src_bytes // dst_bytes)

        def fn(x):
            typed = x[::2].view(dst_dtype)
            return x, typed.as_strided((3,), (stride,), offset)

        x = torch.arange(48, dtype=src_dtype, device=device).reshape(6, 8)[1:5]
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[1].stride(), expected[1].stride())
        self.assertTrue(torch._C._is_alias_of(actual[0], actual[1]))

    def test_compile_inplace_view(self, device):
        def fn(x):
            base = x + 1
            view = base[:, 1:]
            view.as_strided_((3,), (1,))
            return view + 1

        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x))

    @onlyCPU
    @parametrize("strict", [False, True])
    @parametrize(
        "return_view",
        [False, subtest(True, decorators=[unittest.expectedFailure])],
    )
    def test_compile_shared_mkldnn_view(self, device, strict, return_view):
        if not torch.backends.mkldnn.is_available():
            self.skipTest("MKLDNN is required")

        def fn(x):
            base = x + 1
            view = base[:, :, ::2, ::2]
            conv = torch.ops.mkldnn._convolution_pointwise.default(
                view,
                torch.ones(4, 3, 1, 1, device=x.device),
                None,
                [0, 0],
                [1, 1],
                [1, 1],
                1,
                "none",
                [],
                None,
            )
            out = view.as_strided((3,), (1,), 0)
            return (base, view, conv, out) if return_view else (base, conv, out)

        x = torch.arange(192, dtype=torch.float32, device=device).reshape(1, 3, 8, 8)
        expected = fn(x)
        with config.patch(strict_output_strides=strict):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[-1].stride(), expected[-1].stride())
        self.assertTrue(torch._C._is_alias_of(actual[0], actual[-1]))
        if return_view:
            # https://github.com/pytorch/pytorch/pull/197712#issuecomment-6074453380
            self.assertEqual(
                (actual[1].stride(), torch._C._is_alias_of(actual[0], actual[1])),
                (expected[1].stride(), True),
            )

    @onlyCPU
    @parametrize("as_strided_source", ["none", "unrelated", "base", "view"])
    def test_compile_shared_mkldnn_view_reuses_copy(self, device, as_strided_source):
        if not torch.backends.mkldnn.is_available():
            self.skipTest("MKLDNN is required")

        def fn(x, w1, w2, other):
            base = x + 1
            view = base[:, :, ::2, ::2]
            conv = torch.ops.mkldnn._convolution_pointwise.default
            out1 = conv(view, w1, None, [0, 0], [1, 1], [1, 1], 1, "none", [], None)
            out2 = conv(view, w2, None, [0, 0], [1, 1], [1, 1], 1, "none", [], None)
            if as_strided_source == "unrelated":
                other_base = other + 2
                extra = other_base[1:].as_strided((2,), (1,), 0) + 1
                return out1, out2, extra
            if as_strided_source in ("base", "view"):
                source = base if as_strided_source == "base" else view
                extra = source.as_strided((3,), (1,), 0) + 1
                return out1, out2, extra
            return out1, out2

        x = torch.randn(1, 3, 8, 8, device=device)
        w1 = torch.randn(4, 3, 1, 1, device=device)
        w2 = torch.randn(5, 3, 1, 1, device=device)
        other = torch.randn(16, device=device)
        with config.patch(force_disable_caches=True):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), x, w1, w2, other
            )
        self.assertEqual(actual, fn(x, w1, w2, other))
        code = "\n".join(codes)
        inputs = re.findall(r"mkldnn\._convolution_pointwise\.default\((\w+),", code)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0], inputs[1])
        if as_strided_source in ("none", "unrelated"):
            self.assertNotIn("empty_strided_cpu((1, 3, 8, 8)", code)

    @onlyCPU
    def test_compile_neg_view_materialization(self, device):
        def fn(x):
            view = torch._neg_view(x)[:, ::2]
            return view.as_strided((1,), (1,), 0) + 1

        x = torch.arange(96, dtype=torch.float32, device=device).reshape(8, 12)
        actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, fn(x))
        code = "\n".join(codes)
        self.assertIn("empty_strided_cpu((8, 6), (6, 1), torch.float32)", code)
        self.assertNotIn("empty_strided_cpu((8, 12)", code)

    @onlyCPU
    @parametrize("slice_view", [False, True])
    def test_compile_shared_mkldnn_input(self, device, slice_view):
        if not torch.backends.mkldnn.is_available():
            self.skipTest("MKLDNN is required")

        def fn(x, weight):
            base = x + 1
            conv = torch.ops.mkldnn._convolution_pointwise.default(
                base, weight, None, [1, 1], [1, 1], [1, 1], 1, "none", [], None
            )
            view = base[:, :, ::2, :] if slice_view else base
            return conv, view.as_strided((16,), (1,), 0)

        x = torch.randn(2, 3, 8, 8, device=device)
        weight = torch.randn(4, 3, 3, 3, device=device)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x, weight), fn(x, weight))


devices = ["cpu"] if HAS_CPU else []
if HAS_GPU:
    devices.append(GPU_TYPE)
instantiate_device_type_tests(
    TestAsStrided, globals(), only_for=devices, allow_xpu=True
)


if __name__ == "__main__":
    if HAS_CPU or HAS_GPU:
        run_tests()
