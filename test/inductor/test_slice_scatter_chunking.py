# Owner(s): ["module: inductor"]

import gc
import operator
from unittest import mock, skipIf

import torch
import torch._inductor.config as inductor_config
from functorch import make_fx
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx
from torch._inductor.fx_passes import slice_scatter_chunking
from torch._inductor.fx_utils import FakeTensorUpdater
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_utils import IS_MACOS, parametrize, subtest
from torch.testing._internal.inductor_utils import requires_gpu
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten


class TestSliceScatterChunking(TestCase):
    def _run_pass(self, fn, *args, tracing_mode="fake", custom=None):
        gm = make_fx(fn, tracing_mode=tracing_mode)(*args)
        if custom is not None:
            for node in gm.graph.find_nodes(
                op="call_function", target=aten.slice_scatter.default
            ):
                node.meta["custom"] = custom.copy()
        fake_tensor_updater = FakeTensorUpdater(gm)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        with V.set_fake_mode(fake_mode):
            slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)
            fake_tensor_updater.incremental_update()
        gm.graph.lint()
        gm.recompile()
        return gm

    @parametrize("factory", ("empty", "empty_strided"))
    @parametrize("source_kind", ("distinct", "reused", "views"))
    def test_nonfunctional_chain(self, device, factory, source_kind):
        def fn(*inputs):
            if source_kind == "distinct":
                chunks = inputs
            elif source_kind == "reused":
                chunks = inputs * 3
            else:
                chunks = (inputs[0][:2], inputs[0][2:4], inputs[0][4:6])
            if factory == "empty":
                out = torch.empty(
                    (6, 4), device=inputs[0].device, dtype=inputs[0].dtype
                )
            else:
                out = torch.empty_strided(
                    (6, 4), (4, 1), device=inputs[0].device, dtype=inputs[0].dtype
                )
            out = aten.slice_scatter.default(out, chunks[0], 0, 0, 2)
            out = aten.slice_scatter.default(out, chunks[1], 0, 2, 4)
            return aten.slice_scatter.default(out, chunks[2], 0, 4, 6)

        if source_kind == "distinct":
            args = tuple(torch.randn(2, 4, device=device) for _ in range(3))
        elif source_kind == "reused":
            args = (torch.randn(2, 4, device=device),)
        else:
            args = (torch.randn(6, 4, device=device),)
        gm = self._run_pass(fn, *args)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        # Replacement tracing decomposes cat([x, x, x]) into expand/clone/view.
        self.assertEqual(
            targets.count(aten.cat.default), 0 if source_kind == "reused" else 1
        )
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(*args), fn(*args))

    @parametrize("dim", (0, -1))
    def test_non_pointwise_intermediates_use_copies(self, device, dim):
        def fn(a, b, c, d):
            head = a @ b
            tail = c @ d
            shape = list(head.shape)
            chunk_size = shape[dim]
            shape[dim] *= 2
            out = torch.empty(shape, device=a.device, dtype=a.dtype)
            out = aten.slice_scatter.default(out, head, dim, 0, chunk_size)
            return aten.slice_scatter.default(
                out, tail, dim, chunk_size, chunk_size * 2
            )

        args = tuple(
            torch.randn(shape, device=device)
            for shape in ((2, 4), (4, 3), (2, 4), (4, 3))
        )
        gm = self._run_pass(fn, *args)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), 2)
        self.assertEqual(gm(*args), fn(*args))
        self.assertEqual(gm(*args).stride(), fn(*args).stride())

    def test_nested_region_output_fusibility(self, device):
        graph = torch.fx.Graph()
        region = graph.call_function(
            torch.ops.higher_order.invoke_subgraph, args=(None, "region")
        )
        source = graph.call_function(operator.getitem, args=(region, 0))
        source.meta["val"] = torch.empty(1, device=device)

        self.assertEqual(
            slice_scatter_chunking._is_nested_compile_region_output(source),
            device != "cpu",
        )

    @parametrize("consumer", ("matmul", "pointwise"))
    def test_reused_pointwise_chunks(self, device, consumer):
        def fn(x, weight, tail):
            out = torch.empty((4, 4), device=x.device, dtype=x.dtype)
            head = x.cos()
            tail = tail.sin()
            if consumer == "matmul":
                aux = head @ weight + tail @ weight
            else:
                aux = head.sin() + tail.cos()
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            out = aten.slice_scatter.default(out, tail, 0, 2, 4)
            return out, aux

        args = (
            torch.randn(2, 4, device=device),
            torch.randn(4, 4, device=device),
            torch.randn(2, 4, device=device),
        )
        gm = self._run_pass(fn, *args)
        targets = [node.target for node in gm.graph.nodes]

        expected_copies = 2 if consumer == "matmul" else 0
        self.assertEqual(
            targets.count(aten.slice_scatter.default), 0 if expected_copies else 2
        )
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), expected_copies)
        self.assertEqual(gm(*args), fn(*args))

    def test_realized_pointwise_chunks_use_copies(self, device):
        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device)
            out = aten.slice_scatter.default(out, aten.angle.default(head), 0, 0, 2)
            return aten.slice_scatter.default(out, aten.angle.default(tail), 0, 2, 4)

        args = tuple(
            torch.randn(2, 4, device=device, dtype=torch.complex64) for _ in range(2)
        )
        with mock.patch.object(
            slice_scatter_chunking,
            "is_node_realized",
            wraps=slice_scatter_chunking.is_node_realized,
        ) as is_realized:
            gm = self._run_pass(fn, *args)
        targets = [node.target for node in gm.graph.nodes]

        self.assertTrue(is_realized.called)
        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), 2)
        self.assertEqual(gm(*args), fn(*args))

    @parametrize("with_aliases", (False, True))
    def test_functionalized_copy_chain(self, device, with_aliases):
        def alias(value):
            return aten.alias.default(value) if with_aliases else value

        def write(out, src, start, end):
            dst = aten.slice.Tensor(alias(out), 0, start, end)
            copied = aten.copy.default(dst, src, non_blocking=True)
            return alias(
                aten.slice_scatter.default(alias(alias(out)), copied, 0, start, end)
            )

        def fn(head, middle, tail):
            out = torch.empty((6, 4), device=head.device, dtype=head.dtype)
            out = write(out, head, 0, 2)
            out = write(out, middle, 2, 4)
            return write(out, tail, 4, 6)

        head = torch.randn(2, 4, device=device)
        middle = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, middle, tail)
        targets = [node.target for node in gm.graph.nodes]

        self.assertNotIn(aten.slice_scatter.default, targets)
        self.assertNotIn(aten.copy.default, targets)
        self.assertEqual(targets.count(aten.copy_.default), 3)
        self.assertEqual(targets.count(aten.cat.default), 0)
        copies = gm.graph.find_nodes(op="call_function", target=aten.copy_.default)
        self.assertEqual([node.args[2] for node in copies], [True, True, True])
        self.assertEqual(gm(head, middle, tail), torch.cat((head, middle, tail)))

    def test_view_inputs_as_kwargs(self, device):
        def write(out, src, start, end):
            dst = aten.slice.Tensor(aten.alias.default(out), 0, start, end)
            copied = aten.copy.default(dst, src)
            return aten.slice_scatter.default(
                aten.alias.default(out), copied, 0, start, end
            )

        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            out = write(out, head, 0, 2)
            return write(out, tail, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = make_fx(fn, tracing_mode="fake")(head, tail)
        for node in gm.graph.nodes:
            if node.target not in (aten.alias.default, aten.slice.Tensor):
                continue
            node.kwargs = {
                argument.name: value
                for argument, value in zip(
                    node.target._schema.arguments, node.args, strict=False
                )
            }
            node.args = ()

        fake_tensor_updater = FakeTensorUpdater(gm)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        with V.set_fake_mode(fake_mode):
            slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)
            fake_tensor_updater.incremental_update()
        gm.graph.lint()
        gm.recompile()
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), 2)
        self.assertEqual(gm(head, tail), fn(head, tail))

    @parametrize("base_kind", ("input", "pointwise", "aliasing"))
    def test_functionalized_copy_chain_rejects_base(self, device, base_kind):
        def write(out, src, start, end):
            copied = aten.copy.default(out[start:end], src)
            return aten.slice_scatter.default(out, copied, 0, start, end)

        def fn(head, tail, base):
            if base_kind == "pointwise":
                out = base.cos()
            elif base_kind == "aliasing":
                out = aten._unsafe_view.default(base, (4, 4))
            else:
                out = base
            out = write(out, head, 0, 2)
            return write(out, tail, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        base_shape = (16,) if base_kind == "aliasing" else (4, 4)
        base = torch.randn(base_shape, device=device)
        original = base.clone()
        gm = self._run_pass(fn, head, tail, base)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, tail, base), torch.cat((head, tail)))
        self.assertEqual(base, original)

    @onlyCUDA
    def test_rejects_cross_device_functional_copy(self, device):
        def write(out, src, start, end):
            copied = aten.copy.default(torch.empty_like(src), src, non_blocking=True)
            return aten.slice_scatter.default(out, copied, 0, start, end)

        def fn(head, tail):
            out = torch.empty((4, 4), device="cpu", pin_memory=True)
            out = write(out, head, 0, 2)
            return write(out, tail, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, tail), fn(head, tail))

    @parametrize("case", ("broadcast", "dtype"))
    def test_rejects_copy_that_changes_payload_metadata(self, device, case):
        def fn(head, tail):
            out = torch.empty((4, 4), device=device)
            first = aten.copy.default(out[:2], head)
            out = aten.slice_scatter.default(out, first, 0, 0, 2)
            last = aten.copy.default(out[2:], tail)
            return aten.slice_scatter.default(out, last, 0, 2, 4)

        shape = (1, 4) if case == "broadcast" else (2, 4)
        dtype = torch.float64 if case == "dtype" else torch.float32
        head = torch.randn(shape, device=device, dtype=dtype)
        tail = torch.randn(shape, device=device, dtype=dtype)
        gm = self._run_pass(fn, head, tail)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(gm(head, tail), fn(head, tail))

    @parametrize("functionalized", (False, True))
    def test_rejects_live_intermediate(self, device, functionalized):
        def write(out, src, start, end):
            if not functionalized:
                return aten.slice_scatter.default(out, src, 0, start, end)
            dst = aten.slice.Tensor(aten.alias.default(out), 0, start, end)
            copied = aten.copy.default(dst, src)
            return aten.slice_scatter.default(
                aten.alias.default(out), copied, 0, start, end
            )

        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            first = write(out, head, 0, 2)
            last = write(first, tail, 2, 4)
            live = aten.alias.default(first) if functionalized else first
            return live, last

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, tail)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        first, last = gm(head, tail)
        self.assertEqual(first[:2], head)
        self.assertEqual(last, torch.cat((head, tail)))

    @parametrize("base_kind", ("input", "pointwise", "offset", "extra_capacity"))
    def test_rejects_non_factory_base(self, device, base_kind):
        def fn(head, tail, storage):
            if base_kind == "input":
                out = storage
            elif base_kind == "pointwise":
                out = storage.cos()
            else:
                view = storage[1:5] if base_kind == "offset" else storage[:4]
                out = aten.select_scatter.default(view, storage[-1], 0, 0)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            return aten.slice_scatter.default(out, tail, 0, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        rows = 4 if base_kind in ("input", "pointwise") else 6
        storage = torch.randn(rows, 4, device=device)
        expected = fn(head, tail, storage)
        gm = self._run_pass(fn, head, tail, storage)
        actual = gm(head, tail, storage)
        targets = [node.target for node in gm.graph.nodes]

        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(actual, expected)
        if base_kind in ("offset", "extra_capacity"):
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(
                actual.untyped_storage().nbytes(), expected.untyped_storage().nbytes()
            )

    @parametrize("case", ("partial", "gap"))
    def test_requires_complete_chain(self, device, case):
        def fn(head, tail, base):
            out = torch.empty(base.shape, device=base.device, dtype=base.dtype)
            if case == "partial":
                out = aten.slice_scatter.default(out, head[:1], 0, 0, 1)
                return aten.slice_scatter.default(out, tail[:1], 0, 1, 2)
            if case == "gap":
                out = aten.slice_scatter.default(out, head[:1], 0, 0, 1)
                return aten.slice_scatter.default(out, tail[:1], 0, 2, 3)
            raise AssertionError(f"unexpected case: {case}")

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        base = torch.randn(4, 4, device=device)
        gm = self._run_pass(fn, head, tail, base)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)

    @parametrize("last_end", (None, 2**63 - 1))
    def test_dynamic_boundaries(self, device, last_end):
        def fn(head, tail):
            head_rows = head.shape[0]
            total_rows = head_rows + tail.shape[0]
            out = torch.empty(
                (total_rows, head.shape[1]), device=head.device, dtype=head.dtype
            )
            out = aten.slice_scatter.default(out, head, 0, None, head_rows)
            return aten.slice_scatter.default(out, tail, 0, head_rows, last_end)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(3, 4, device=device)
        gm = self._run_pass(fn, head, tail, tracing_mode="symbolic")

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        self.assertEqual(targets.count(aten.cat.default), 1)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, tail), torch.cat((head, tail)))

    @parametrize(
        "case",
        (
            subtest(((4, 4), (0, 1), None, 0), name="overlapping"),
            subtest(((4, 4), (1, 4), (1, 4), 2), name="noncontiguous"),
            subtest(((4, 1), (1, 99), (1, 99), 2), name="noncanonical"),
        ),
    )
    def test_factory_layout(self, device, case):
        shape, stride, expected_stride, expected_copies = case

        def fn(head, tail):
            out = torch.empty_strided(
                shape, stride, device=head.device, dtype=head.dtype
            )
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            return aten.slice_scatter.default(out, tail, 0, 2, 4)

        chunk_shape = (2, *shape[1:])
        head = torch.randn(chunk_shape, device=device)
        tail = torch.randn(chunk_shape, device=device)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(
            targets.count(aten.slice_scatter.default), 0 if expected_copies else 2
        )
        self.assertEqual(targets.count(aten.copy_.default), expected_copies)
        if expected_stride is not None:
            self.assertEqual(gm(head, tail).stride(), expected_stride)

    def test_requires_matching_dtype(self, device):
        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=torch.float32)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            return aten.slice_scatter.default(out, tail, 0, 2, 4)

        head = torch.randn(2, 4, device=device, dtype=torch.float16)
        tail = torch.randn(2, 4, device=device, dtype=torch.float16)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, tail).dtype, torch.float32)

    def test_noncontiguous_sources(self, device):
        def fn(head, tail):
            out = torch.empty((2, 3, 1, 1), device=head.device, dtype=head.dtype)
            out = aten.slice_scatter.default(out, head, 0, 0, 1)
            out = aten.slice_scatter.default(out, tail, 0, 1, 2)
            return aten.as_strided.default(out, (6,), (1,), 0)

        head = torch.randn(1, 3, 1, 1, device=device).to(
            memory_format=torch.channels_last
        )
        tail = torch.randn(1, 3, 1, 1, device=device).to(
            memory_format=torch.channels_last
        )
        self.assertTrue(head.is_contiguous())
        expected = fn(head, tail)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 0)
        self.assertEqual(targets.count(aten.copy_.default), 2)
        self.assertEqual(gm(head, tail), expected)

    @parametrize("replacement", ("cat", "copy"))
    @parametrize("same_context", (True, False))
    def test_custom_context(self, device, replacement, same_context):
        def write(out, src, start, end):
            if replacement == "copy":
                src = aten.copy.default(out[start:end], src)
            return aten.slice_scatter.default(out, src, 0, start, end)

        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            out = write(out, head, 0, 2)
            return write(out, tail, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        custom = {"stream": 1, "mempool": 2, "mempool_device": 0}
        gm = make_fx(fn, tracing_mode="fake")(head, tail)
        nodes = [
            node
            for node in gm.graph.nodes
            if node.target
            in (
                aten.empty.memory_format,
                aten.copy.default,
                aten.slice_scatter.default,
            )
        ]
        for node in nodes:
            node.meta["custom"] = custom.copy()
        if not same_context:
            nodes[-1].meta["custom"] = {**custom, "stream": 2}
        fake_tensor_updater = FakeTensorUpdater(gm)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        with V.set_fake_mode(fake_mode):
            slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)
            fake_tensor_updater.incremental_update()
        gm.graph.lint()
        gm.recompile()

        remaining = gm.graph.find_nodes(
            op="call_function", target=aten.slice_scatter.default
        )
        copies = gm.graph.find_nodes(op="call_function", target=aten.copy_.default)
        cats = gm.graph.find_nodes(op="call_function", target=aten.cat.default)
        expected_copies = 2 if same_context and replacement == "copy" else 0
        expected_cats = 1 if same_context and replacement == "cat" else 0
        self.assertEqual(len(copies), expected_copies)
        self.assertEqual(len(cats), expected_cats)
        self.assertEqual(len(remaining), 0 if same_context else 2)
        if same_context:
            replacement_node = copies[0] if replacement == "copy" else cats[0]
            self.assertEqual(replacement_node.meta.get("custom"), custom)
        self.assertEqual(gm(head, tail), fn(head, tail))

    def test_deep_live_view_chain(self, device):
        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            out = aten.slice_scatter.default(out, tail, 0, 2, 4)
            for _ in range(1100):
                out = out[1:]
            return out

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, tail)

        self.assertEqual(gm(head, tail), fn(head, tail))

    def test_ignores_dead_destination_views(self, device):
        def write(out, src, start, end):
            copied = aten.copy.default(out[start:end], src)
            return aten.slice_scatter.default(out, copied, 0, start, end)

        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            aten.slice.Tensor(out, 0, 0, 2)
            out = write(out, head, 0, 2)
            aten.slice.Tensor(out, 0, 2, 4)
            return write(out, tail, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice.Tensor), 2)
        self.assertNotIn(aten.slice_scatter.default, targets)
        self.assertEqual(targets.count(aten.copy_.default), 2)
        self.assertEqual(gm(head, tail), fn(head, tail))

    def test_rejects_intervening_mutation(self, device):
        def fn(x):
            x = x.clone()
            head = x[:2]
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            x.add_(10)
            return aten.slice_scatter.default(out, x[2:], 0, 2, 4)

        x = torch.randn(4, 4, device=device)
        gm = self._run_pass(fn, x)

        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(gm(x), fn(x))

    @onlyCUDA
    @inductor_config.patch(force_disable_caches=True)
    def test_mm_chunk_codegen_bounded_peak(self, device):
        def fn(a, b, c, d):
            rows = a.shape[0]
            out = torch.empty((rows * 2, b.shape[1]), device=a.device, dtype=a.dtype)
            out[:rows].copy_(a @ b)
            out[rows : rows * 2].copy_(c @ d)
            return out

        rows, width, inner = 2048, 4096, 8
        args = tuple(
            torch.randn(shape, device=device)
            for shape in (
                (rows, inner),
                (inner, width),
                (rows, inner),
                (inner, width),
            )
        )
        compiled = torch.compile(fn, fullgraph=True)
        actual = compiled(*args)

        self.assertEqual(actual, fn(*args))
        del actual
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        compiled(*args)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        self.assertLessEqual(peak, 97 * 2**20)

    @inductor_config.patch(force_disable_caches=True)
    def test_functionalized_copy_chain_compile(self, device):
        def fn(head, tail):
            out = torch.empty_strided(
                (4, 4), (1, 4), device=head.device, dtype=head.dtype
            )
            first = aten.copy.default(out[:2], head)
            out = aten.slice_scatter.default(out, first, 0, 0, 2)
            last = aten.copy.default(out[2:4], tail)
            return aten.slice_scatter.default(out, last, 0, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = make_fx(fn, tracing_mode="fake")(head, tail)
        with mock.patch.object(
            slice_scatter_chunking,
            "_replace_with_inplace_copies",
            wraps=slice_scatter_chunking._replace_with_inplace_copies,
        ) as replace:
            compiled = compile_fx(gm, [head, tail])

        self.assertEqual(replace.call_count, 1)
        actual = compiled(head, tail)
        expected = fn(head, tail)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.stride(), expected.stride())

    @onlyCPU
    @requires_gpu()
    @skipIf(IS_MACOS, "pinned memory is not available on macOS")
    @inductor_config.patch(force_disable_caches=True)
    def test_rejects_pinned_functionalized_copy_base(self, device):
        def fn(head, tail):
            out = torch.empty_strided((4, 4), (1, 4), device="cpu", pin_memory=True)
            first = aten.copy.default(out[:2], head)
            out = aten.slice_scatter.default(out, first, 0, 0, 2)
            last = aten.copy.default(out[2:4], tail)
            return aten.slice_scatter.default(out, last, 0, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        expected = fn(head, tail)
        gm = make_fx(fn, tracing_mode="fake")(head, tail)
        with mock.patch.object(
            slice_scatter_chunking,
            "_replace_with_inplace_copies",
            wraps=slice_scatter_chunking._replace_with_inplace_copies,
        ) as replace:
            compiled = compile_fx(gm, [head, tail])

        self.assertEqual(replace.call_count, 0)
        actual = compiled(head, tail)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.is_pinned(), expected.is_pinned())

    def test_mutation_regions_computed_once_after_rewrites(self, device):
        def fn(*chunks):
            states = []
            for head in chunks[::2]:
                out = torch.empty((2, 4), device=head.device, dtype=head.dtype)
                copied = aten.copy.default(out[:1], head)
                states.append(aten.slice_scatter.default(out, copied, 0, 0, 1))

            outputs = []
            for state, tail in zip(states, chunks[1::2], strict=True):
                copied = aten.copy.default(state[1:2], tail)
                outputs.append(aten.slice_scatter.default(state, copied, 0, 1, 2))
            return outputs

        chunks = tuple(torch.randn(1, 4, device=device) for _ in range(6))
        with mock.patch.object(
            slice_scatter_chunking,
            "compute_mutation_region_ids",
            wraps=slice_scatter_chunking.compute_mutation_region_ids,
        ) as compute:
            gm = self._run_pass(fn, *chunks)

        self.assertEqual(compute.call_count, 2)
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=aten.slice_scatter.default
                )
            ),
            0,
        )
        self.assertEqual(gm(*chunks), fn(*chunks))

    def test_no_candidate_skips_mutation_region_scan(self, device):
        def fn(x):
            return x.cos()

        x = torch.randn(4, device=device)
        with mock.patch.object(
            slice_scatter_chunking,
            "compute_mutation_region_ids",
            wraps=slice_scatter_chunking.compute_mutation_region_ids,
        ) as compute:
            gm = self._run_pass(fn, x)

        self.assertEqual(compute.call_count, 0)
        self.assertEqual(gm(x), fn(x))

    def test_rejected_chain_is_scanned_once(self, device):
        chain_length = 20

        def fn(base, chunks):
            out = base
            for index in range(chain_length):
                out = aten.slice_scatter.default(
                    out, chunks[index : index + 1], 0, index, index + 1
                )
            return out

        base = torch.randn(chain_length, device=device)
        chunks = torch.randn(chain_length, device=device)
        gm = make_fx(fn, tracing_mode="fake")(base, chunks)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        with mock.patch.object(
            slice_scatter_chunking,
            "_normalize_slice_scatter_args",
            wraps=slice_scatter_chunking._normalize_slice_scatter_args,
        ) as normalize:
            with V.set_fake_mode(fake_mode):
                slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)

        gm.graph.lint()
        gm.recompile()
        self.assertLessEqual(normalize.call_count, chain_length * 3)
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=aten.slice_scatter.default
                )
            ),
            chain_length,
        )
        self.assertEqual(gm(base, chunks), fn(base, chunks))

    def test_graph_order_is_computed_once(self, device):
        chain_count = 8

        def fn(*chunks):
            outputs = []
            for head, tail in zip(chunks[::2], chunks[1::2], strict=True):
                out = torch.empty((2, 4), device=head.device, dtype=head.dtype)
                out = aten.slice_scatter.default(out, head, 0, 0, 1)
                outputs.append(aten.slice_scatter.default(out, tail, 0, 1, 2))
            return outputs

        chunks = tuple(torch.randn(1, 4, device=device) for _ in range(chain_count * 2))
        gm = make_fx(fn, tracing_mode="fake")(*chunks)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        original_nodes = torch.fx.Graph.nodes
        graph_node_accesses = 0

        def counted_nodes(graph):
            nonlocal graph_node_accesses
            if graph is gm.graph:
                graph_node_accesses += 1
            return original_nodes.__get__(graph, torch.fx.Graph)

        with mock.patch.object(torch.fx.Graph, "nodes", property(counted_nodes)):
            with V.set_fake_mode(fake_mode):
                slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)

        gm.graph.lint()
        gm.recompile()
        self.assertLessEqual(graph_node_accesses, 3)
        self.assertEqual(gm(*chunks), fn(*chunks))

    def test_shared_alias_path_is_scanned_once(self, device):
        alias_length = 12
        chain_count = 8

        def fn(*chunks):
            base = torch.empty((2, 4), device=device)
            for _ in range(alias_length):
                base = aten.alias.default(base)
            outputs = []
            for head, tail in zip(chunks[::2], chunks[1::2], strict=True):
                out = aten.slice_scatter.default(base, head, 0, 0, 1)
                outputs.append(aten.slice_scatter.default(out, tail, 0, 1, 2))
            return outputs

        chunks = tuple(torch.randn(1, 4, device=device) for _ in range(chain_count * 2))
        gm = make_fx(fn, tracing_mode="fake")(*chunks)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        ordered_set_add = OrderedSet.add
        alias_adds = 0

        def counted_add(values, value):
            nonlocal alias_adds
            if isinstance(value, torch.fx.Node) and value.target is aten.alias.default:
                alias_adds += 1
            return ordered_set_add(values, value)

        with mock.patch.object(OrderedSet, "add", counted_add):
            with V.set_fake_mode(fake_mode):
                slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)

        gm.graph.lint()
        gm.recompile()
        self.assertLessEqual(alias_adds, alias_length * 3)
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=aten.slice_scatter.default
                )
            ),
            chain_count * 2,
        )
        self.assertEqual(gm(*chunks), fn(*chunks))

    def test_shared_copy_view_path_is_scanned_once(self, device):
        alias_length = 12
        chain_length = 8

        def fn(*chunks):
            base = torch.empty((chain_length, 4), device=device)
            dst = base[:1]
            for _ in range(alias_length):
                dst = aten.alias.default(dst)
            out = base
            for index, chunk in enumerate(chunks):
                copied = aten.copy.default(dst, chunk)
                out = aten.slice_scatter.default(out, copied, 0, index, index + 1)
            return out

        chunks = tuple(torch.randn(1, 4, device=device) for _ in range(chain_length))
        gm = make_fx(fn, tracing_mode="fake")(*chunks)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        is_view_op = slice_scatter_chunking._is_view_op
        view_checks = 0

        def counted_is_view_op(target):
            nonlocal view_checks
            view_checks += 1
            return is_view_op(target)

        with mock.patch.object(
            slice_scatter_chunking, "_is_view_op", counted_is_view_op
        ):
            with V.set_fake_mode(fake_mode):
                slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)

        gm.graph.lint()
        gm.recompile()
        self.assertLessEqual(view_checks, (alias_length + chain_length) * 3)
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=aten.slice_scatter.default
                )
            ),
            chain_length,
        )
        self.assertEqual(gm(*chunks), fn(*chunks))

    @onlyCUDA
    @inductor_config.patch(reorder_for_locality=False, force_disable_caches=True)
    def test_nested_compile_region_uses_cat(self, device):
        @torch.compiler.nested_compile_region
        def chunk(a, b):
            return a @ b

        def fn(a, b, c, d):
            rows = a.shape[0]
            out = torch.empty((rows * 2, b.shape[1]), device=a.device, dtype=a.dtype)
            out = aten.slice_scatter.default(out, chunk(a, b), 0, 0, rows)
            return aten.slice_scatter.default(out, chunk(c, d), 0, rows, rows * 2)

        rows, width, inner = 2048, 4096, 8
        args = tuple(
            torch.randn(shape, device=device)
            for shape in (
                (rows, inner),
                (inner, width),
                (rows, inner),
                (inner, width),
            )
        )
        can_fuse_as_cat = slice_scatter_chunking._can_fuse_as_cat
        saw_regional_cat = False

        def tracked_can_fuse_as_cat(chain, graph_input_storages):
            nonlocal saw_regional_cat
            result = can_fuse_as_cat(chain, graph_input_storages)
            if any(
                slice_scatter_chunking._is_nested_compile_region_output(src)
                for src in chain.sources
            ):
                saw_regional_cat |= result
            return result

        compiled = torch.compile(fn, fullgraph=True)
        with mock.patch.object(
            slice_scatter_chunking,
            "_can_fuse_as_cat",
            tracked_can_fuse_as_cat,
        ):
            actual = compiled(*args)

        self.assertTrue(saw_regional_cat)
        self.assertEqual(actual, fn(*args))

    @parametrize("dim", (0, -1))
    def test_pointwise_chunks_are_not_rewritten(self, device, dim):
        def fn(head, tail):
            shape = list(head.shape)
            chunk_size = shape[dim]
            shape[dim] *= 2
            out = torch.empty(shape, device=head.device, dtype=head.dtype)
            out = aten.slice_scatter.default(out, head.cos(), dim, 0, chunk_size)
            return aten.slice_scatter.default(
                out, tail.sin(), dim, chunk_size, chunk_size * 2
            )

        head = torch.randn(2, 3, device=device)
        tail = torch.randn(2, 3, device=device)
        gm = self._run_pass(fn, head, tail)
        targets = [node.target for node in gm.graph.nodes]
        actual = gm(head, tail)
        expected = fn(head, tail)

        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(targets.count(aten.cat.default), 0)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.stride(), expected.stride())

    @onlyCUDA
    @inductor_config.patch(force_disable_caches=True)
    @parametrize("source_kind", ("distinct", "reused", "views", "getitem"))
    def test_many_pointwise_chunks_preserve_fusion(self, device, source_kind):
        chunk_count = 16 if source_kind == "distinct" else 3

        def fn(*chunks):
            out = torch.empty(
                (2 * chunk_count, 1024),
                device=chunks[0].device,
                dtype=chunks[0].dtype,
            )
            if source_kind == "distinct":
                sources = tuple(chunk.cos() for chunk in chunks)
            elif source_kind == "reused":
                sources = (chunks[0].cos(),) * chunk_count
            elif source_kind == "views":
                source = chunks[0].cos()
                sources = tuple(source[2 * i : 2 * i + 2] for i in range(chunk_count))
            else:
                sources = tuple(aten.frexp.Tensor(chunk)[0] for chunk in chunks)
            for index, source in enumerate(sources):
                out = aten.slice_scatter.default(
                    out, source, 0, 2 * index, 2 * index + 2
                )
            return out

        args = tuple(
            torch.randn(
                (2 * chunk_count, 1024) if source_kind == "views" else (2, 1024),
                device=device,
            )
            for _ in range(chunk_count if source_kind in ("distinct", "getitem") else 1)
        )
        gm = self._run_pass(fn, *args)
        targets = [node.target for node in gm.graph.nodes]
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), *args)

        self.assertEqual(targets.count(aten.slice_scatter.default), chunk_count)
        self.assertEqual(actual, fn(*args))
        self.assertEqual(code.count(".run("), 1)

    @onlyCUDA
    @inductor_config.patch(force_disable_caches=True)
    @parametrize("source_kind", ("distinct", "reused", "views"))
    def test_external_chunks_codegen(self, device, source_kind):
        def fn(*inputs):
            if source_kind == "distinct":
                chunks = inputs
            elif source_kind == "reused":
                chunks = inputs * 3
            else:
                chunks = (inputs[0][:2], inputs[0][2:4], inputs[0][4:6])
            out = torch.empty((6, 1024), device=inputs[0].device, dtype=inputs[0].dtype)
            out = aten.slice_scatter.default(out, chunks[0], 0, 0, 2)
            out = aten.slice_scatter.default(out, chunks[1], 0, 2, 4)
            return aten.slice_scatter.default(out, chunks[2], 0, 4, 6)

        if source_kind == "distinct":
            args = tuple(torch.randn(2, 1024, device=device) for _ in range(3))
        elif source_kind == "reused":
            args = (torch.randn(2, 1024, device=device),)
        else:
            args = (torch.randn(6, 1024, device=device),)
        actual = torch.compile(fn, fullgraph=True)(*args)

        self.assertEqual(actual, fn(*args))

    @onlyCUDA
    def test_does_not_block_cos_fusion(self, device):
        def fn(x, src):
            return aten.slice_scatter.default(x.cos(), src, 0, 1, 3)

        x = torch.randn(4, 1024, device=device)
        src = torch.randn(2, 1024, device=device)
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x, src)

        self.assertEqual(actual, fn(x, src))
        self.assertEqual(code.count(".run("), 1)


instantiate_device_type_tests(
    TestSliceScatterChunking,
    globals(),
    only_for=("cpu", "cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests(needs="filelock")
