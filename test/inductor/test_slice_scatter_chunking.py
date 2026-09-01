# Owner(s): ["module: inductor"]

import gc
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
    def test_complete_chain(self, device, factory):
        def fn(head, middle, tail):
            if factory == "empty":
                out = torch.empty((6, 4), device=head.device, dtype=head.dtype)
            else:
                out = torch.empty_strided(
                    (6, 4), (4, 1), device=head.device, dtype=head.dtype
                )
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            out = aten.slice_scatter.default(out, middle, 0, 2, 4)
            return aten.slice_scatter.default(out, tail, 0, 4, 6)

        head = torch.randn(2, 4, device=device)
        middle = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, middle, tail)
        targets = [node.target for node in gm.graph.nodes]

        self.assertNotIn(aten.slice_scatter.default, targets)
        self.assertEqual(targets.count(aten.cat.default), 1)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, middle, tail), torch.cat((head, middle, tail)))

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
    def test_cat_rejects_non_factory_base(self, device, base_kind):
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

    def test_dynamic_boundaries(self, device):
        def fn(head, tail):
            head_rows = head.shape[0]
            total_rows = head_rows + tail.shape[0]
            out = torch.empty(
                (total_rows, head.shape[1]), device=head.device, dtype=head.dtype
            )
            out = aten.slice_scatter.default(out, head, 0, 0, head_rows)
            return aten.slice_scatter.default(out, tail, 0, head_rows, total_rows)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(3, 4, device=device)
        gm = self._run_pass(fn, head, tail, tracing_mode="symbolic")

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(aten.slice_scatter.default, targets)
        self.assertEqual(gm(head, tail), torch.cat((head, tail)))

    @parametrize(
        "case",
        (
            subtest(((4, 4), (0, 1), None), name="overlapping"),
            subtest(((4, 4), (1, 4), (1, 4)), name="noncontiguous"),
            subtest(((4, 1), (1, 99), (1, 99)), name="noncanonical"),
        ),
    )
    def test_rejects_unsupported_factory_layout(self, device, case):
        shape, stride, expected_stride = case

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
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
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

    def test_skips_noncontiguous_sources(self, device):
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
        self.assertEqual(targets.count(aten.slice_scatter.default), 2)
        self.assertEqual(targets.count(aten.copy_.default), 0)
        self.assertEqual(gm(head, tail), expected)

    @parametrize(
        "same_context",
        (
            subtest(True, name="matching"),
            subtest(False, name="different"),
        ),
    )
    def test_custom_context(self, device, same_context):
        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            return aten.slice_scatter.default(out, tail, 0, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        custom = {"stream": 1, "mempool": 2, "mempool_device": 0}
        gm = make_fx(fn, tracing_mode="fake")(head, tail)
        slice_scatters = gm.graph.find_nodes(
            op="call_function", target=aten.slice_scatter.default
        )
        slice_scatters[0].meta["custom"] = custom.copy()
        slice_scatters[1].meta["custom"] = (
            custom.copy() if same_context else {**custom, "stream": 2}
        )
        fake_tensor_updater = FakeTensorUpdater(gm)
        fake_mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
        with V.set_fake_mode(fake_mode):
            slice_scatter_chunking.slice_scatter_chunking_pass(gm.graph)
            fake_tensor_updater.incremental_update()
        gm.graph.lint()

        cats = gm.graph.find_nodes(op="call_function", target=aten.cat.default)
        remaining = gm.graph.find_nodes(
            op="call_function", target=aten.slice_scatter.default
        )
        self.assertEqual(len(cats), 1 if same_context else 0)
        self.assertEqual(len(remaining), 0 if same_context else 2)
        if same_context:
            self.assertEqual(cats[0].meta.get("custom"), custom)

    def test_ignores_dead_destination_views(self, device):
        def fn(head, tail):
            out = torch.empty((4, 4), device=head.device, dtype=head.dtype)
            aten.slice.Tensor(out, 0, 0, 2)
            out = aten.slice_scatter.default(out, head, 0, 0, 2)
            aten.slice.Tensor(out, 0, 2, 4)
            return aten.slice_scatter.default(out, tail, 0, 2, 4)

        head = torch.randn(2, 4, device=device)
        tail = torch.randn(2, 4, device=device)
        gm = self._run_pass(fn, head, tail)

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(aten.slice.Tensor, targets)
        self.assertNotIn(aten.slice_scatter.default, targets)
        self.assertEqual(targets.count(aten.cat.default), 1)
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
    def test_mm_chunk_codegen(self, device):
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
        self.assertLessEqual(peak, 65 * 2**20)

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
            out = torch.empty_strided((4, 4), (1, 4), pin_memory=True)
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
    def test_nested_compile_region_peak(self, device):
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
        self.assertLessEqual(peak, 129 * 2**20)

    @onlyCUDA
    def test_pointwise_chunks_fuse(self, device):
        def fn(x):
            rows = x.shape[0] // 2
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            out = aten.slice_scatter.default(out, x[:rows].cos(), 0, 0, rows)
            return aten.slice_scatter.default(out, x[rows:].sin(), 0, rows, x.shape[0])

        x = torch.randn(8192, 1024, device=device)
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)

        self.assertEqual(actual, fn(x))
        self.assertEqual(code.count(".run("), 1)

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
    only_for=("cpu", "cuda"),
)


if __name__ == "__main__":
    run_tests(needs="filelock")
