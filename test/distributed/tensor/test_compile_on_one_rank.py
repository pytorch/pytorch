# Owner(s): ["oncall: distributed"]

import difflib
import functools
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import pytest

import torch
import torch.compiler.config as compiler_config
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.fx._graph_pickler import GraphPickler, Options
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def extract_graph(fx_g, _, graph_cell):
    """Extract the FX graph code into a mutable cell."""
    graph_cell[0] = fx_g.code
    return fx_g


class TestCompileOnOneRank(DTensorTestBase):
    def _assert_graphs_identical_across_ranks(self, local_graph_code):
        """Gather compiled graph code from all ranks and assert they are identical."""
        self.assertIsNotNone(local_graph_code, "Graph was not captured")

        graph_bytes = local_graph_code.encode("utf-8")
        graph_tensor = torch.tensor(
            list(graph_bytes), dtype=torch.uint8, device=self.device_type
        )

        # Pad to same length across ranks
        local_len = torch.tensor(
            [len(graph_bytes)], dtype=torch.int64, device=self.device_type
        )
        all_lens = [
            torch.zeros(1, dtype=torch.int64, device=self.device_type)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_lens, local_len)
        max_len = int(max(l.item() for l in all_lens))

        padded_tensor = torch.zeros(max_len, dtype=torch.uint8, device=self.device_type)
        padded_tensor[: len(graph_bytes)] = graph_tensor

        all_graphs = [
            torch.zeros(max_len, dtype=torch.uint8, device=self.device_type)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_graphs, padded_tensor)

        graph_codes = []
        for graph_t, len_t in zip(all_graphs, all_lens):
            length = int(len_t.item())
            graph_str = bytes(graph_t[:length].tolist()).decode("utf-8")
            graph_codes.append(graph_str)

        rank0_graph = graph_codes[0]
        for rank, graph_code in enumerate(graph_codes[1:], start=1):
            if rank0_graph != graph_code:
                diff = difflib.unified_diff(
                    rank0_graph.splitlines(keepends=True),
                    graph_code.splitlines(keepends=True),
                    fromfile="rank0_graph",
                    tofile=f"rank{rank}_graph",
                )
                diff_str = "".join(diff)
                self.fail(
                    f"Graph on rank {rank} differs from rank 0. "
                    f"This indicates rank-specific literals were baked into the graph.\n"
                    f"Unified diff:\n{diff_str}"
                )

    def _compile_and_capture_graph(self, model):
        """Compile model with a graph-capturing backend and return the graph cell."""
        fw_graph_cell = [None]
        fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)

        from functorch.compile import min_cut_rematerialization_partition
        from torch._dynamo.backends.common import aot_autograd

        aot_eager_graph = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=fw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        compiled_model = torch.compile(model, backend=aot_eager_graph)
        return compiled_model, fw_graph_cell

    @with_comms
    @compiler_config.patch(compile_on_one_rank=True)
    def test_compiled_rowwise_embedding_graph_consistency(self):
        """Test that compiled graphs are identical across all ranks.

        When rowwise sharded embeddings are compiled with torch.compile, the
        _MaskPartial._mask_tensor() function generates bounds checking
        operations (lt, ge, sub, index_put) with rank-specific values that get
        baked into the compiled graph:
        - Rank 0: lt(index, 0), ge(index, 64), sub(index, 0)
        - Rank 1: lt(index, 64), ge(index, 128), sub(index, 64)

        These values should be symbolic/dynamic, not baked-in literals, to
        ensure graph consistency across ranks.
        """
        mesh = self.build_device_mesh()

        class Network(nn.Module):
            def __init__(self, num_embeddings, embedding_dim, device):
                super().__init__()
                self.tok_embeddings = nn.Embedding(
                    num_embeddings, embedding_dim, device=device
                )

            def forward(self, x):
                return self.tok_embeddings(x)

        torch.manual_seed(0)
        num_embeddings = 256
        embedding_dim = 64

        model = Network(num_embeddings, embedding_dim, device=self.device_type)

        parallelize_module(
            model,
            mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
            },
        )

        compiled_model, fw_graph_cell = self._compile_and_capture_graph(model)

        torch.manual_seed(42)
        inp = torch.randint(0, num_embeddings, (64, 16), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)

        compiled_model(replicated_inp)
        self._assert_graphs_identical_across_ranks(fw_graph_cell[0])

    @with_comms
    @compiler_config.patch(compile_on_one_rank=True)
    def test_all_reduce_with_explicit_pg_input(self):
        pg = dist.distributed_c10d._get_default_group()

        def f(t, group):
            t = t.clone()
            dist.all_reduce(t, group=group)
            return t + 1

        x = torch.arange(4, dtype=torch.float32, device=self.device_type)
        opt = torch.compile(f, backend="inductor", fullgraph=True)
        out = opt(x, pg)
        self.assertEqual(out, f(x, pg))

    @with_comms
    @compiler_config.patch(compile_on_one_rank=True)
    def test_compiled_dtensor_rng_op_graph_consistency(self):
        """Compiled random ops on sharded DTensors should produce identical graphs."""
        mesh = self.build_device_mesh()
        dt = DTensor.from_local(
            torch.empty(8, 4, device=self.device_type), mesh, [Shard(0)]
        )

        fw_graph_cell = [None]
        fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)

        from functorch.compile import min_cut_rematerialization_partition
        from torch._dynamo.backends.common import aot_autograd

        compiled_f = torch.compile(
            lambda x: torch.rand_like(x),
            backend=aot_autograd(
                fw_compiler=fw_compiler,
                partition_fn=min_cut_rematerialization_partition,
            ),
        )

        compiled_f(dt)
        self._assert_graphs_identical_across_ranks(fw_graph_cell[0])

    @with_comms
    @compiler_config.patch(compile_on_one_rank=True)
    def test_all_reduce_with_implicit_world_group(self):
        """`dist.all_reduce(t)` with no `group=` (implicit `dist.group.WORLD`)
        should compile under compile_on_one_rank=True.

        `WorldMetaClassVariable.tp_getattro_impl` was routing the WORLD lookup through
        `SourcelessBuilder`, dropping the source it had just constructed for the
        guard. The resulting `CustomClassObjectVariable` had the raw ProcessGroup
        as its `proxy` field and blew up later in `as_proxy()` when the PG was
        passed to `_c10d_functional.all_reduce` (which only happens with
        compile_on_one_rank=True, since otherwise the PG is converted to a
        string group name before becoming an op arg).

        Uses backend="aot_eager" to isolate the Dynamo-side fix.
        Regression test for https://github.com/pytorch/pytorch/issues/181890.
        """

        def f(t):
            t = t.clone()
            dist.all_reduce(t)
            return t + 1

        x = torch.arange(4, dtype=torch.float32, device=self.device_type)
        opt = torch.compile(f, backend="aot_eager", fullgraph=True)
        out = opt(x)
        self.assertEqual(out, f(x))


def _factory_from_input_device(x):
    # Factory op whose device + dtype are derived from an input tensor, mirroring
    # real CooR graphs (e.g. token_dispatcher.py: torch.zeros(..., device=x.device)
    # and SimpleFSDP mixed-precision casts). Shape is incidental.
    return torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)


def _indexed_cuda_device_nodes(gm):
    """Nodes carrying a concrete, indexed cuda device in their args/kwargs.

    These are the rank-specific constants that make a make_fx graph non
    device-agnostic. A device-agnostic graph fetches the device in-graph (via the
    current_device() node) and so has none of these.
    """
    found = []
    for node in gm.graph.nodes:
        operands = list(node.args) + list(node.kwargs.values())
        for operand in operands:
            if (
                isinstance(operand, torch.device)
                and operand.type == "cuda"
                and operand.index is not None
            ):
                found.append(node)
                break
    return found


def _current_device_nodes(gm):
    """Nodes that fetch the current device in-graph."""
    target = torch.ops.coor.current_device.default
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


class TestCompileOnOneRankDeviceAsParameter(TestCase):
    """Device-as-parameter for the make_fx tracing path used by graph_trainer/CooR.

    Under compile_on_one_rank, a factory/cast op whose device matches the current
    accelerator (e.g. cuda:0, or a bare cuda) traces with its device= fed by a single
    in-graph current_device() node, instead of baking the concrete device. At
    runtime the device follows each rank's current accelerator device (not any input),
    so one compiled artifact runs on each rank's real GPU without --virtual-local-rank.
    A device that is a different accelerator, or a different index of the current
    accelerator, is refused (it could not run SPMD).
    """

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_factory_device_replaced_with_current_device(self):
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        ca = _current_device_nodes(gm)
        self.assertEqual(
            len(ca), 1, "device should be fetched in-graph via a single node"
        )
        self.assertTrue(ca[0].users, "the current_device() node must be consumed")
        baked = _indexed_cuda_device_nodes(gm)
        self.assertEqual(
            baked,
            [],
            lambda msg: f"{msg}\nno node should bake a concrete indexed cuda device; found: {baked}",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_dynamo_output_graph_factory_device_not_baked(self):
        # The same factory pattern as test_factory_device_replaced_with_current_device,
        # but reached through Dynamo instead of calling make_fx directly.
        #
        # Dynamo constant-folds x.device to a concrete torch.device and bakes it into
        # its output graph. The current_device() substitution runs later, during
        # make_fx, so it cannot undo what Dynamo already froze. This is the graph
        # tlparse records as dynamo_output_graph, and in a real CooR job it differs
        # across ranks (index=0 vs index=7), which keeps it from being shareable.
        from torch._dynamo.testing import EagerAndRecordGraphs

        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        torch.compile(_factory_from_input_device, backend=backend, fullgraph=True)(
            torch.randn(2, 8, device="cuda:0")
        )
        self.assertEqual(len(backend.graphs), 1)
        baked = _indexed_cuda_device_nodes(backend.graphs[0])
        self.assertEqual(
            baked,
            [],
            f"dynamo baked a rank-specific device into its output graph: {baked}",
        )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_runtime_follows_current_device_not_input(self):
        # The runtime device follows the process's current device, not the input's.
        # The input is kept on cuda:0 in both runs; only the current device changes.
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        with torch.cuda.device(0):
            self.assertEqual(
                gm(torch.randn(2, 8, device="cuda:0")).device, torch.device("cuda:0")
            )
        with torch.cuda.device(1):
            self.assertEqual(
                gm(torch.randn(2, 8, device="cuda:0")).device, torch.device("cuda:1")
            )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_default_path_unchanged_bakes_device(self):
        # Without compile_on_one_rank the device stays baked (the feature must be
        # gated so it does not perturb the default tracing path).
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        self.assertEqual(_current_device_nodes(gm), [])
        self.assertTrue(_indexed_cuda_device_nodes(gm))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_to_copy_explicit_device_replaced(self):
        # An explicit-device dtype cast (the SimpleFSDP mixed-precision pattern,
        # aten._to_copy with a device= kwarg) also gets its baked device rewired to
        # the current_device() node, alongside the factory-op path.
        def f(x):
            return x.to(device="cuda:0", dtype=torch.bfloat16)

        gm = make_fx(f, tracing_mode="fake")(torch.randn(2, 8, device="cuda:0"))
        self.assertEqual(len(_current_device_nodes(gm)), 1)
        self.assertEqual(_indexed_cuda_device_nodes(gm), [])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_unindexed_accelerator_device_replaced(self):
        # A bare device="cuda" (index None) matching the current accelerator is also
        # replaced by the current_device() node.
        def f(x):
            return torch.zeros(4, x.shape[1], device="cuda", dtype=x.dtype)

        gm = make_fx(f, tracing_mode="fake")(torch.randn(2, 8, device="cuda:0"))
        self.assertEqual(len(_current_device_nodes(gm)), 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_cpu_device_left_alone(self):
        # cpu is portable on every rank, so a cpu factory is not rewritten.
        def f(x):
            return torch.zeros(4, x.shape[1], device="cpu")

        gm = make_fx(f, tracing_mode="fake")(torch.randn(2, 8, device="cuda:0"))
        self.assertEqual(_current_device_nodes(gm), [])
        cpu_ops = [
            n
            for n in gm.graph.nodes
            if any(
                isinstance(o, torch.device) and o.type == "cpu"
                for o in list(n.args) + list(n.kwargs.values())
            )
        ]
        self.assertTrue(cpu_ops, "the cpu device should stay baked")

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_coor_check_current_accelerator(self):
        # The shared validator (used by the make_fx input check, the operand rewrite, and the
        # benchmark-harness device renderer) must accept the current accelerator (bare or its
        # index) and cpu, and refuse a non-current accelerator -- so a device cannot be
        # silently re-rendered as the current one. End-to-end harness rendering (bare "cuda",
        # no "cuda:N") is covered by test_inductor_compiles_under_coor.
        from torch.fx.experimental.proxy_tensor import (
            _coor_check_current_accelerator,
            _coor_current_accelerator,
        )

        with torch.cuda.device(0):
            cur = _coor_current_accelerator()
            _coor_check_current_accelerator(torch.device("cuda:0"), cur)
            _coor_check_current_accelerator(torch.device("cuda"), cur)
            _coor_check_current_accelerator(torch.device("cpu"), cur)
            with self.assertRaisesRegex(RuntimeError, "device-agnostic"):
                _coor_check_current_accelerator(torch.device("cuda:1"), cur)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch(cpp_wrapper=True)
    def test_cpp_wrapper_under_coor_rejected(self):
        # cpp_wrapper/AOTInductor bakes the compile-time device index into the C++ device
        # guard, which is not rank-portable. Compile-on-one-rank must refuse it rather than
        # silently emit a non-portable artifact.
        # Exception (not RuntimeError) because dynamo wraps this in BackendCompilerFailed,
        # so the regex has to carry the specificity.
        with torch.cuda.device(0):
            with self.assertRaisesRegex(
                Exception,
                r"compile-on-one-rank .*not supported with cpp_wrapper/AOTInductor",
            ):
                torch.compile(lambda x: x + 1)(torch.randn(8, device="cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch(fx_wrapper=True)
    def test_fx_wrapper_under_coor_rejected(self):
        # fx_wrapper's device-context codegen is a no-op, so it would bake the compile-time
        # device index like cpp_wrapper. Compile-on-one-rank must refuse it rather than
        # silently emit a non-portable artifact.
        with torch.cuda.device(0):
            with self.assertRaisesRegex(
                Exception, r"compile-on-one-rank .*not supported with .*fx_wrapper"
            ):
                torch.compile(lambda x: x + 1)(torch.randn(8, device="cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_factory_without_matching_input_succeeds(self):
        # Unlike provenance-following, matching the current accelerator needs no input
        # on that device: a cuda factory in a cpu-input graph is now rewritten, not
        # rejected.
        def f(x):
            return torch.zeros(x.shape[0], device="cuda:0")

        gm = make_fx(f, tracing_mode="fake")(torch.randn(2, device="cpu"))
        self.assertEqual(len(_current_device_nodes(gm)), 1)
        self.assertEqual(_indexed_cuda_device_nodes(gm), [])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_wrong_index_raises(self):
        # A cuda index that is not the current device's index cannot be made SPMD, so
        # the rewrite refuses it (raised during tracing before the fake op runs, so
        # this needs only the current device to exist).
        def f(x):
            return torch.zeros(4, device="cuda:1")

        with torch.cuda.device(0):
            with self.assertRaisesRegex(
                RuntimeError, "index differs from the current accelerator"
            ):
                make_fx(f, tracing_mode="fake")(torch.randn(2, device="cuda:0"))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_noncurrent_device_tensor_rejected(self):
        # CooR rejects a device *operand* that isn't the current accelerator (see
        # test_wrong_index_raises), but its single-device invariant also requires the
        # graph's *tensors* to be on the current device: the inductor wrapper collapses
        # every device reference to the runtime current device (_coor_device_idx), so a
        # cuda:1 graph would be run on cuda:0. A graph whose input is on a non-current GPU
        # has no device operand to catch, so make_fx must reject it on the tensor device.
        def f(x):
            return x + 1

        with torch.cuda.device(0):
            with self.assertRaisesRegex(RuntimeError, "device-agnostic"):
                make_fx(f, tracing_mode="fake")(torch.randn(4, device="cuda:1"))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_graph_code_identical_across_devices(self):
        # The functional FX graph text (.code) must be byte-identical across ranks: the
        # device operand is the current_device() node, never a baked cuda:N. (.code
        # carries no per-tensor device annotations, so it stays clean while the tensor meta
        # remains on its real cuda:N device.)
        def code_on(dev):
            with torch.cuda.device(dev):
                return make_fx(_factory_from_input_device, tracing_mode="fake")(
                    torch.randn(2, 8, device=f"cuda:{dev}")
                ).code

        code0, code1 = code_on(0), code_on(1)
        self.assertEqual(code0, code1)
        self.assertNotIn("cuda:", code0)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_dynamo_output_graph_identical_across_devices(self):
        # The Dynamo counterpart of test_graph_code_identical_across_devices, and the
        # form the divergence actually takes in a real job: the same factory traces to
        # device(type='cuda', index=0) on one rank and index=N on another, so the two
        # ranks' dynamo_output_graph artifacts are not the same text and the graph
        # cannot be shared between them.
        from torch._dynamo.testing import EagerAndRecordGraphs

        def graph_on(dev):
            with torch.cuda.device(dev):
                torch._dynamo.reset()
                backend = EagerAndRecordGraphs()
                torch.compile(
                    _factory_from_input_device, backend=backend, fullgraph=True
                )(torch.randn(2, 8, device=f"cuda:{dev}"))
                return backend.graphs[0].print_readable(print_output=False)

        graph0, graph1 = graph_on(0), graph_on(1)
        self.assertEqual(graph0, graph1)
        self.assertNotIn("index=0", graph0)

    # ---- tensor guards must be rank-invariant without losing their teeth ----
    # A TENSOR_MATCH guard records the device as two independent pieces: the type
    # rides in the DispatchKeySet, and the index is a separate scalar rendered as
    # "device=N". Only the index is rank-specific, so only the index may be relaxed,
    # and it must be relaxed into a check against the *current* device rather than
    # dropped -- CooR's single-accelerator invariant (one accelerator device, with
    # cpu free to coexist) is enforced when tracing, so at runtime the guard is the
    # only thing left watching for a stray device.

    @staticmethod
    def _tensor_guard_parts(fn):
        """The check_tensor(...) guard lines installed for fn."""
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        parts = []
        for entry in _debug_get_cache_entry_list(fn):
            parts += [
                line.strip()
                for line in str(entry.guard_manager).splitlines()
                if "check_tensor(" in line
            ]
        return parts

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_does_not_bake_device_index_under_coor(self):
        import re

        def f(x):
            return x + 1

        torch._dynamo.reset()
        torch.compile(f, backend="eager")(torch.randn(4, device="cuda"))
        parts = self._tensor_guard_parts(f)
        self.assertTrue(parts, "expected a check_tensor guard to be installed")
        baked = [p for p in parts if re.search(r"device=\d", p)]
        self.assertEqual(
            baked, [], f"guard baked a rank-specific device index: {baked}"
        )
        self.assertTrue(
            all("device=current" in part for part in parts),
            f"guard did not use the current device: {parts}",
        )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_still_rejects_noncurrent_device_index_under_coor(self):
        # Relaxing the index must not mean ignoring it: a tensor on a device that is
        # not the current one still has to fail the guard. Deleting the check outright
        # would silently pass here.
        #
        # Retrying the call does not recompile -- tracing refuses a non-current
        # accelerator input outright (test_noncurrent_device_input_refused_under_coor),
        # which is what keeps the guard decision derivable rather than recorded.
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        from torch._dynamo.testing import CompileCounter

        def f(x):
            return x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend=cnt)
            compiled(torch.randn(4, device="cuda:0"))
            other = torch.randn(4, device="cuda:1")
            root = _debug_get_cache_entry_list(f)[0].guard_manager.root
            debug_info = root.check_verbose({"x": other})
            self.assertFalse(debug_info.result)
            self.assertIn(
                "current device (0), actual 1",
                "\n".join(debug_info.verbose_code_parts),
            )
            with self.assertRaisesRegex(RuntimeError, "current accelerator"):
                compiled(other)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_noncurrent_device_input_refused_under_coor(self) -> None:
        # A graph whose input is on an accelerator other than the current one cannot be
        # made rank-portable, so CooR refuses it while tracing. The make_fx backends
        # already do (_coor_check_current_accelerator); Dynamo has to as well, or an
        # eager-backend compile quietly produces a pinned, non-portable artifact --
        # and it is the only thing that makes the relative-vs-exact guard decision
        # derivable on any rank rather than something to record and replay.
        def f(x):
            return x + 1

        torch._dynamo.reset()
        with torch.cuda.device(0):
            with self.assertRaisesRegex(RuntimeError, "current accelerator"):
                torch.compile(f, backend="eager", fullgraph=True)(
                    torch.randn(4, device="cuda:1")
                )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_hits_when_current_device_changes_under_coor(self):
        # The whole point of the relaxation: one compiled artifact serves every rank.
        # Move the current device to 1 and hand it a tensor that followed, and the
        # guard should match the entry compiled on device 0 rather than recompile.
        #
        # This is the test that distinguishes a real fix from a cosmetic one: it fails
        # unless the runtime check became device-relative. Rewording the guard string
        # alone leaves it failing. Read together with
        # test_guard_still_rejects_noncurrent_device_index_under_coor -- same cuda:1
        # tensor, opposite expectation -- the pair pins the check to "the current
        # device" rather than to any fixed index.
        from torch._dynamo.testing import CompileCounter

        def f(x):
            return x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend=cnt)
            compiled(torch.randn(4, device="cuda:0"))
            before = cnt.frame_count
        with torch.cuda.device(1):
            compiled(torch.randn(4, device="cuda:1"))
        self.assertEqual(
            cnt.frame_count,
            before,
            "a tensor on the new current device should reuse the existing compile",
        )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_cloned_tensor_guard_tracks_current_device_under_coor(self):
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        def f(x):
            return x + 1

        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend="eager")
            compiled(torch.zeros(1, device="cuda:0"))
            root = _debug_get_cache_entry_list(f)[0].guard_manager.root
            cloned_root = root.clone_manager(lambda _: True)

        with torch.cuda.device(1):
            inputs = {"x": torch.zeros(1, device="cuda:1")}
            self.assertTrue(cloned_root.check(inputs))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("origin", ("input", "intermediate"))
    def test_device_passthrough_still_reuses_compile_under_coor(self, origin):
        # The payoff of the relaxed index guard, measured the only way it shows up:
        # a recompile count. A tensor's device handed straight to a factory has to
        # keep serving every rank from one artifact, whether it came off an input or
        # off an intermediate.
        #
        # The inductor tests nearby check that the *generated code* is
        # device-agnostic, which anything that merely re-pinned the guard would not
        # change -- only counting frames catches that.
        from torch._dynamo.testing import CompileCounterWithBackend

        def f(x):
            y = x + 1 if origin == "intermediate" else x
            return torch.zeros(4, y.shape[1], device=y.device, dtype=y.dtype) + y.sum()

        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend=cnt)
            compiled(torch.randn(2, 8, device="cuda:0"))
            before = cnt.frame_count
        with torch.cuda.device(1):
            out = compiled(torch.randn(2, 8, device="cuda:1"))
        self.assertEqual(out.device, torch.device("cuda:1"))
        self.assertEqual(
            cnt.frame_count,
            before,
            "passing a device to a factory is not an observation and must not guard",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_still_rejects_device_type_under_coor(self):
        # cpu and cuda tensors coexist freely in one process, so the device *type*
        # must stay guarded; only the index is redundant under CooR.
        from torch._dynamo.testing import CompileCounter

        def f(x):
            return x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt)
        compiled(torch.randn(4, device="cuda"))
        before = cnt.frame_count
        compiled(torch.randn(4))
        self.assertEqual(
            cnt.frame_count, before + 1, "device type must still be guarded"
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_cpu_tensor_guard_unchanged_under_coor(self):
        # The invariant is single-*accelerator*, not single-device: cpu tensors
        # coexist with the accelerator freely under CooR (a cpu factory op is not
        # even rewritten -- see test_cpu_device_left_alone), and a cpu device is
        # portable across ranks already. So a cpu tensor's guard has to come out
        # exactly as it would with the feature off, never relaxed to "current".
        def f(x):
            return x + 1

        torch._dynamo.reset()
        torch.compile(f, backend="eager")(torch.randn(4))
        parts = self._tensor_guard_parts(f)
        self.assertTrue(parts, "expected a check_tensor guard to be installed")
        relaxed = [p for p in parts if "device=current" in p]
        self.assertEqual(
            relaxed, [], f"a cpu tensor's guard must not be relaxed: {relaxed}"
        )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    def test_device_index_still_guarded_without_coor(self):
        # Multi-GPU in one process is legal outside CooR (e.g. model parallel), so the
        # relaxation must be gated: with the feature off, the index stays baked and a
        # different index still recompiles.
        import re

        from torch._dynamo.testing import CompileCounter

        def f(x):
            return x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend=cnt)
            compiled(torch.randn(4, device="cuda:0"))
            before = cnt.frame_count
            self.assertTrue(
                [p for p in self._tensor_guard_parts(f) if re.search(r"device=\d", p)],
                "without compile_on_one_rank the index should stay baked",
            )
            compiled(torch.randn(4, device="cuda:1"))
            self.assertEqual(cnt.frame_count, before + 1)

    # ---- inductor codegen and launcher must be device-agnostic across ranks ----
    # A device-derived factory + a reduction, so inductor emits a real kernel.
    @staticmethod
    def _coor_inductor_fn(x):
        z = torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)
        return z + x.sum()

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_compiles_under_coor(self):
        # The current_device() node must lower through inductor, and the generated code
        # must be device-agnostic: the device is resolved at runtime with no baked
        # rank-specific index, so one compiled artifact is shareable across ranks.
        from torch._C import FileCheck
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        compiled = torch.compile(
            self._coor_inductor_fn, backend="inductor", fullgraph=True
        )
        out, codes = run_and_get_code(compiled, torch.randn(2, 8, device="cuda"))
        self.assertEqual(out.device.type, "cuda")
        code = "\n".join(codes)
        FileCheck().check("torch.cuda.current_device()").run(code)
        self._assert_no_baked_device(code)

    @staticmethod
    def _coor_template_fn(a, b):
        return (a @ b).relu()

    @staticmethod
    def _coor_combo_fn(a, b):
        return a.sin(), b.cos()

    @staticmethod
    def _make_coor_user_defined_triton_fn():
        import triton
        import triton.language as tl

        @triton.jit
        def add_one_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask) + 1, mask=mask)

        def fn(x):
            out = torch.empty_like(x)
            add_one_kernel[(1,)](x, out, x.numel(), BLOCK=128)
            return out

        return fn

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON"
    )
    def test_inductor_template_no_baked_device(self):
        # A Triton template must not bake the rank-specific device index either.
        #
        # Templates build triton_meta separately from TritonKernel, so this guards
        # the select_algorithm.py construction path.
        #
        # test_inductor_compiles_under_coor does not catch this: _coor_inductor_fn
        # is a factory plus a reduction, which only produces inductor-generated
        # kernels and never reaches the template path.
        from torch._inductor import utils as inductor_utils

        torch._dynamo.reset()
        compiled = torch.compile(
            self._coor_template_fn, backend="inductor", fullgraph=True
        )
        # The metadata path is dtype-independent; float32 keeps it covered on pre-SM80.
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        with (
            inductor_utils.fresh_cache(),
            patch.object(inductor_utils, "is_big_gpu", return_value=True),
        ):
            _, codes = inductor_utils.run_and_get_code(compiled, a, b)
        code = "\n".join(codes)
        self.assertIn("triton_tem_fused", code)
        self._assert_no_baked_device(code)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch(
        combo_kernels=True,
        benchmark_combo_kernel=False,
        combo_kernel_peak_memory_increase_gb=None,
        combo_kernel_peak_memory_pct_threshold=None,
    )
    def test_inductor_combo_kernel_no_baked_device(self):
        # Combo kernels build triton_meta separately from ordinary pointwise kernels.
        # Disable benchmarking and memory gating to isolate that codegen path.
        from torch._inductor.utils import fresh_cache, run_and_get_code

        torch._dynamo.reset()
        compiled = torch.compile(
            self._coor_combo_fn, backend="inductor", fullgraph=True
        )
        args = (
            torch.randn(8192, device="cuda"),
            torch.randn(4096, device="cuda"),
        )
        with fresh_cache():
            _, codes = run_and_get_code(compiled, *args)
        code = "\n".join(codes)
        self.assertIn("combo_grid_meta", code)
        self._assert_no_baked_device(code)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_user_defined_triton_kernel_no_baked_device(self):
        # A user-defined @triton.jit kernel gets its triton_meta from a third site,
        # define_user_defined_triton_kernel in codegen/wrapper.py, which is neither
        # the TritonKernel path nor the template path.
        from torch._inductor.utils import fresh_cache, run_and_get_code

        torch._dynamo.reset()
        fn = self._make_coor_user_defined_triton_fn()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(128, device="cuda")
        with fresh_cache():
            out, codes = run_and_get_code(compiled, x)
        self.assertEqual(out, x + 1)
        self._assert_no_baked_device("\n".join(codes))

    def _assert_no_baked_device(self, code):
        # A baked index reaches generated code in more forms than "cuda:N": repr() of a
        # torch.device gives device(type='cuda', index=0), and triton_meta renders
        # DeviceProperties(..., index=0). Check all three.
        self.assertNotRegex(code, r"cuda:\d")
        self.assertNotRegex(code, r"device\(type=.cuda., index=\d")
        self.assertNotRegex(code, r"DeviceProperties\([^)]*index=\d")

    def _inductor_code_on_device(self, dev, fn, input_shapes):
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        with torch.cuda.device(dev):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            inputs = tuple(
                torch.randn(shape, device=f"cuda:{dev}") for shape in input_shapes
            )
            _, codes = run_and_get_code(compiled, *inputs)
        return "\n".join(codes)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize(
        "path",
        (
            "benchmark_kernel",
            "compile_time_autotune",
            "template",
            "combo_kernel",
            "user_defined_triton",
        ),
    )
    def test_inductor_code_identical_across_devices(self, path):
        # The inductor-side rewrite is a dozen independent opt-in `if _coor_enabled():`
        # sites with no structural funnel, so pattern-matching one rank's output cannot
        # show that none was missed. Diffing the code generated on two different devices
        # can. Cover the ordinary kernel's benchmark and compile-time autotune harnesses,
        # plus every independent triton_meta construction path.
        import re
        from contextlib import nullcontext

        from torch._inductor import utils as inductor_utils

        def norm(s):
            return re.sub(r"AOT ID: \['\d+_", "AOT ID: ['N_", s)

        cases = {
            "benchmark_kernel": (
                self._coor_inductor_fn,
                ((2, 8),),
                {"benchmark_kernel": True},
                False,
            ),
            "compile_time_autotune": (
                self._coor_inductor_fn,
                ((2, 8),),
                {"triton.autotune_at_compile_time": True},
                False,
            ),
            "template": (
                self._coor_template_fn,
                ((256, 256), (256, 256)),
                {
                    "deterministic": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                },
                True,
            ),
            "combo_kernel": (
                self._coor_combo_fn,
                ((8192,), (4096,)),
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                },
                False,
            ),
            "user_defined_triton": (
                self._make_coor_user_defined_triton_fn,
                ((128,),),
                {},
                False,
            ),
        }
        fn, input_shapes, cfg, force_big_gpu = cases[path]
        if path == "user_defined_triton":
            fn = fn()

        def code_on(dev):
            big_gpu = (
                patch.object(inductor_utils, "is_big_gpu", return_value=True)
                if force_big_gpu
                else nullcontext()
            )
            with (
                torch._inductor.config.patch(**cfg),
                inductor_utils.fresh_cache(),
                big_gpu,
            ):
                return self._inductor_code_on_device(dev, fn, input_shapes)

        code0 = code_on(0)
        code1 = code_on(1)
        if norm(code0) != norm(code1):
            diff = "".join(
                difflib.unified_diff(
                    norm(code0).splitlines(keepends=True),
                    norm(code1).splitlines(keepends=True),
                    fromfile="cuda:0",
                    tofile="cuda:1",
                )
            )
            self.fail(
                f"inductor code differs across devices under CooR for {path}:\n{diff}"
            )
        if path == "template":
            self.assertIn("triton_tem_fused", code0)
        elif path == "combo_kernel":
            self.assertIn("combo_grid_meta", code0)
        self._assert_no_baked_device(code0)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch({"triton.force_cooperative_reductions": True})
    def test_cooperative_reduction_workspace_name_not_baked(self):
        # The cooperative-reduction semaphore workspace is named after its device, and
        # that name is emitted into the wrapper -- so an index in it makes the wrapper
        # differ across ranks even though no "cuda:N" literal appears. This runs on
        # whatever device is current rather than a second GPU: the index would be baked
        # as semaphores_cuda_0 just the same, so the check keeps working on the
        # single-GPU runners that make up most of CI.
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        compiled = torch.compile(lambda x: x.sum(), backend="inductor", fullgraph=True)
        _, codes = run_and_get_code(compiled, torch.randn(4096, 4096, device="cuda"))
        code = "\n".join(codes)
        self.assertIn("semaphores_cuda", code)  # the workspace is actually in play
        self.assertNotRegex(code, r"semaphores_cuda_\d")
        self._assert_no_baked_device(code)

    @unittest.skipIf(
        torch.version.cuda is None and torch.version.hip is None,
        "needs a GPU-enabled build whose devices can be hidden",
    )
    def test_coor_compiles_on_gpu_build_with_no_visible_device(self):
        # A GPU-enabled build running where no device is visible -- a container started
        # without --gpus, a scheduler setting CUDA_VISIBLE_DEVICES="", every GPU already
        # allocated -- must still compile a cpu graph under CooR. current_accelerator()
        # reports what the *build* supports rather than what is present, so without an
        # availability check the device-index lookup raises "No CUDA GPUs are available"
        # from inside wrapper codegen.
        #
        # This needs a subprocess: CI never runs that combination directly (GPU jobs have
        # GPUs, CPU jobs have no GPU build), and the devices have to be hidden before
        # torch initializes them, so hiding them in-process is not possible.
        script = textwrap.dedent(
            """
            import torch
            import torch.compiler.config as compiler_config

            assert torch.cuda.device_count() == 0, "expected no visible devices"
            with compiler_config.patch(compile_on_one_rank=True):
                compiled = torch.compile(
                    lambda x: x + 1, backend="inductor", fullgraph=True
                )
                out = compiled(torch.randn(4))
            assert out.device.type == "cpu", out.device
            """
        )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"CooR compile failed with no visible device:\n{proc.stderr[-3000:]}",
        )

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_device_passthrough_custom_backend_tracks_current_device_under_coor(self):
        from torch._dynamo.testing import CompileCounter

        def f(x):
            return torch.zeros(x.shape[0], device=x.device)

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            compiled(torch.ones(1, device="cuda:0"))
        with torch.cuda.device(1):
            x = torch.ones(1, device="cuda:1")
            actual = compiled(x)
            expected = f(x)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(actual.device, expected.device)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("origin", ("input", "factory"))
    def test_device_observations_track_current_device_under_coor(self, origin):
        from torch._dynamo.testing import CompileCounterWithBackend

        def f(x):
            y = torch.zeros(x.shape, device=x.device) if origin == "factory" else x
            return (
                y + y.device.index,
                y.device == torch.device("cuda:1"),
                torch.device("cuda:1") == y.device,
                y.device != torch.device("cuda"),
                y.device,
            )

        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            x = torch.zeros(1, device="cuda:0")
            self.assertEqual(compiled(x), f(x))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            self.assertEqual(compiled(x), f(x))

        self.assertEqual(cnt.frame_count, 1)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("consumer", ("synchronize", "current_stream", "get_device_module"))
    def test_current_device_consumers_under_coor(self, consumer):
        from torch._dynamo.testing import CompileCounter

        def f(x):
            if consumer == "synchronize":
                torch.cuda.synchronize(x.device)
                return x + 1
            if consumer == "current_stream":
                return x + 1, torch.accelerator.current_stream(x.device)
            module = torch.get_device_module(x.device)
            return x + (1 if module is torch.cuda else 2)

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            compiled(torch.zeros(1, device="cuda:0"))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            if consumer == "synchronize":
                with patch.object(torch.accelerator, "synchronize") as synchronize:
                    actual = compiled(x)
                self.assertEqual(actual, x + 1)
                self.assertEqual(synchronize.call_args.args, (torch.device("cuda"),))
            else:
                self.assertEqual(compiled(x), f(x))

        self.assertEqual(cnt.frame_count, 1)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_current_device_context_under_coor(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        def f(x):
            with torch.cuda.device(x.device):
                return torch.ones(1, device="cuda")

        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            compiled(torch.zeros(1, device="cuda:0"))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            self.assertEqual(compiled(x), f(x))

        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_current_device_context_preserves_type_under_coor(self):
        def f(x):
            ctx = torch.cuda.device(x.device)
            return x + (1 if isinstance(ctx, torch.cuda.device) else 2)

        x = torch.zeros(1, device="cuda")
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), f(x))

        def make_context(x):
            return torch.cuda.device(x.device)

        ctx = torch.compile(make_context, backend="eager", fullgraph=True)(x)
        self.assertIsInstance(ctx, torch.cuda.device)

        def make_context_across_graph_break(x):
            ctx = torch.cuda.device(x.device)
            torch._dynamo.graph_break()
            return ctx

        ctx = torch.compile(make_context_across_graph_break, backend="eager")(x)
        self.assertIsInstance(ctx, torch.cuda.device)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_current_device_context_restores_device_across_graph_break(self):
        def f(x):
            with torch.cuda.device(x.device):
                torch._dynamo.graph_break()
                torch.cuda.set_device(1)
                return x + 1

        torch._dynamo.reset()
        compiled = torch.compile(f, backend="eager")
        with torch.cuda.device(0):
            x = torch.zeros(1, device="cuda:0")
            self.assertEqual(compiled(x), x + 1)
            self.assertEqual(torch.cuda.current_device(), 0)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("stream_kind", ("cuda", "generic"))
    @parametrize("use_kwarg", (True, False))
    def test_current_device_stream_constructor_under_coor(self, stream_kind, use_kwarg):
        from torch._dynamo.testing import CompileCounterWithBackend

        def f(x):
            if stream_kind == "cuda":
                stream = (
                    torch.cuda.Stream(device=x.device)
                    if use_kwarg
                    else torch.cuda.Stream(x.device)
                )
            else:
                stream = (
                    torch.Stream(device=x.device)
                    if use_kwarg
                    else torch.Stream(x.device)
                )
            return x + 1, stream

        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            _, stream = compiled(torch.zeros(1, device="cuda:0"))
            self.assertEqual(stream.device, torch.device("cuda:0"))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            result, stream = compiled(x)
            self.assertEqual(result, x + 1)
            self.assertEqual(stream.device, torch.device("cuda:1"))

        self.assertEqual(cnt.frame_count, 1)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize(
        "origin",
        (
            "constructor",
            "generic_default_constructor",
            "generic_none_constructor",
            "generic_bare_constructor",
            "generic_cpu_constructor",
            "cuda_default_constructor",
            "cuda_negative_constructor",
            "accelerator_current",
            "cuda_current",
        ),
    )
    @parametrize("attr", ("device", "device_index", "device_index_via_device"))
    def test_current_device_stream_observation_under_coor(self, origin, attr):
        from torch._dynamo.testing import CompileCounter

        def f(x):
            if origin == "constructor":
                stream = torch.Stream(device=x.device)
            elif origin == "generic_default_constructor":
                stream = torch.Stream()
            elif origin == "generic_none_constructor":
                stream = torch.Stream(device=None)
            elif origin == "generic_bare_constructor":
                stream = torch.Stream(device="cuda")
            elif origin == "generic_cpu_constructor":
                stream = torch.Stream(device="cpu")
            elif origin == "cuda_default_constructor":
                stream = torch.cuda.Stream()
            elif origin == "cuda_negative_constructor":
                stream = torch.cuda.Stream(device=-1)
            elif origin == "accelerator_current":
                stream = torch.accelerator.current_stream(x.device)
            else:
                stream = torch.cuda.current_stream(x.device)
            observation = (
                stream.device.index
                if attr == "device_index_via_device"
                else getattr(stream, attr)
            )
            return x + 1, observation

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            compiled(torch.zeros(1, device="cuda:0"))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            self.assertEqual(compiled(x), f(x))

        self.assertEqual(cnt.frame_count, 1)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("stream_kind", ("generic", "cuda"))
    @parametrize("comparison", ("eq", "ne"))
    @parametrize("matches_current", (True, False))
    def test_current_stream_equality_under_coor(
        self, stream_kind, comparison, matches_current
    ):
        from torch._dynamo.testing import CompileCounter

        def f(x, stream):
            current = (
                torch.accelerator.current_stream(x.device)
                if stream_kind == "generic"
                else torch.cuda.current_stream(x.device)
            )
            matches = stream == current if comparison == "eq" else stream != current
            return x + (1 if matches else 2)

        def make_stream(matches):
            if stream_kind == "generic":
                return torch.accelerator.current_stream() if matches else torch.Stream()
            return torch.cuda.current_stream() if matches else torch.cuda.Stream()

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            x = torch.zeros(1, device="cuda:0")
            stream = make_stream(matches_current)
            compiled(x, stream)
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            stream = make_stream(matches_current)
            self.assertEqual(compiled(x, stream), f(x, stream))

            opposite = make_stream(not matches_current)
            self.assertEqual(compiled(x, opposite), f(x, opposite))

        self.assertEqual(cnt.frame_count, 2)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("stream_kind", ("generic", "cuda"))
    def test_nested_current_device_stream_observation_under_coor(self, stream_kind):
        from torch._dynamo.testing import CompileCounter

        def f(x):
            stream = (
                torch.Stream(device=x.device)
                if stream_kind == "generic"
                else torch.cuda.Stream(device=x.device)
            )
            with stream:
                current = (
                    torch.accelerator.current_stream(x.device)
                    if stream_kind == "generic"
                    else torch.cuda.current_stream(x.device)
                )
                return x + current.device.index, current == stream

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=cnt, fullgraph=True)
        with torch.cuda.device(0):
            compiled(torch.zeros(1, device="cuda:0"))
        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            self.assertEqual(compiled(x), f(x))

        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_device_index_predicate_is_data_dependent_under_coor(self):
        def f(x):
            return x + (1 if x.device.index == 0 else 2)

        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, "Could not guard on data-dependent expression"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(
                torch.zeros(1, device="cuda")
            )

        torch._dynamo.reset()
        x = torch.zeros(1, device="cuda")
        self.assertEqual(torch.compile(f, backend="eager")(x), f(x))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @parametrize("origin", ("input", "intermediate"))
    def test_get_device_tracks_current_device_under_coor(self, origin):
        # get_device() returns an int, and an int has no index-less form. Folding the
        # tracing rank's index would make every rank sharing the artifact report
        # cuda:0 -- no guard fails, nothing recompiles, and the answer is just wrong.
        #
        # Both assertions matter and neither prejudges the fix. Correctness rules
        # out folding the index; the single cache entry rules out "fixing" it with
        # an exact device guard, which would trade a wrong answer for a per-rank
        # artifact and give up what CooR is for.
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        def f(x):
            y = x + 1 if origin == "intermediate" else x
            return y + y.get_device()

        torch._dynamo.reset()
        compiled = torch.compile(f, backend="eager")
        with torch.cuda.device(0):
            compiled(torch.zeros(1, device="cuda:0"))

        with torch.cuda.device(1):
            x = torch.zeros(1, device="cuda:1")
            self.assertEqual(
                compiled(x),
                f(x),
                "get_device() folded the tracing rank's index, so rank 1 was told "
                "it is on cuda:0",
            )
        self.assertLessEqual(
            len(_debug_get_cache_entry_list(f)),
            1,
            "the traced frame must stay rank-portable; a second cache entry means a "
            "device guard forced a per-rank recompile",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_get_device_predicate_is_data_dependent_under_coor(self):
        # Branching on the index is the one thing that cannot be honoured. The value
        # is unknown until the artifact runs, so under fullgraph=True there is no
        # answer to fold and the data-dependent error is the correct outcome -- an
        # error is what should happen, not something to be worked around. Pinned
        # because the tempting "fix" is to specialize the branch, which silently
        # hands every rank the tracing rank's answer.
        def branch(x):
            return x + (1 if x.get_device() == 0 else 2)

        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, "Could not guard on data-dependent expression"
        ):
            torch.compile(branch, backend="eager", fullgraph=True)(
                torch.zeros(1, device="cuda")
            )

        # Without fullgraph the same predicate is a graph break, not an error, and
        # the eager continuation answers it against the device actually in use.
        torch._dynamo.reset()
        x = torch.zeros(1, device="cuda")
        self.assertEqual(torch.compile(branch, backend="eager")(x), branch(x))

        # A predicate the index bound already settles must still fold. get_device()
        # is emitted with a non-negative range, so this is provable for every rank
        # and erroring on it would be over-eager.
        def provable(x):
            return x + (1 if x.get_device() >= 0 else 2)

        torch._dynamo.reset()
        self.assertEqual(
            torch.compile(provable, backend="eager", fullgraph=True)(x), provable(x)
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_code_identical_across_cache_dirs(self):
        # The artifact may be built on one machine and run on another (compatible) one,
        # so generated text must not embed machine-specific paths -- the inductor cache
        # dir is absolute and carries the building user's name. Two fresh cache roots
        # stand in for two machines; a cross-device diff cannot see this class at all
        # because both ranks there share one cache dir.
        import re

        from torch._inductor.utils import fresh_cache

        def code_with_fresh_cache():
            with fresh_cache():
                return self._inductor_code_on_device(
                    0, self._coor_inductor_fn, ((2, 8),)
                )

        code_a, code_b = code_with_fresh_cache(), code_with_fresh_cache()

        def norm(s):
            return re.sub(r"AOT ID: \['\d+_", "AOT ID: ['N_", s)

        if norm(code_a) != norm(code_b):
            diff = "".join(
                difflib.unified_diff(
                    norm(code_a).splitlines(keepends=True),
                    norm(code_b).splitlines(keepends=True),
                    fromfile="cache_dir_a",
                    tofile="cache_dir_b",
                )
            )
            self.fail(
                f"inductor code depends on the cache dir, so it is not portable to "
                f"another machine:\n{diff}"
            )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_extern_kernel_device_arg_not_baked(self):
        # An aten fallback that keeps a device= argument (randperm has no inductor
        # lowering) renders that argument through val_to_arg_str's repr() path, which
        # bakes the index as device(type='cuda', index=0) -- a form the "cuda:N" checks
        # miss entirely.
        from torch._inductor.utils import run_and_get_code

        def f(x):
            return torch.randperm(8, device=x.device) + 0

        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend="inductor", fullgraph=True)
            out, codes = run_and_get_code(compiled, torch.randn(2, 8, device="cuda:0"))
        code = "\n".join(codes)
        self.assertIn("torch.ops.aten.randperm", code)  # still the fallback path
        self._assert_no_baked_device(code)
        self.assertEqual(sorted(out.tolist()), list(range(8)))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch({"triton.cudagraphs": True})
    def test_cudagraphs_under_coor_runs_on_nonzero_device(self):
        # cudagraphs is not refused under CooR (see the guard in compile_fx.py): its
        # device dependence lives in the wrapper-level artifact, which is not shared
        # across ranks today. Pin that it works on a rank's own device.
        inp = torch.randn(2, 8, device="cuda:1")
        ref = self._coor_inductor_fn(inp)
        torch._dynamo.reset()
        with torch.cuda.device(1):
            compiled = torch.compile(
                self._coor_inductor_fn, backend="inductor", fullgraph=True
            )
            for _ in range(3):  # replay, not just record
                out = compiled(inp)
        self.assertEqual(out.device, torch.device("cuda:1"))
        self.assertEqual(out, ref)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_runs_on_nonzero_device(self):
        # Problem 2 (runtime): a graph compiled under CooR must run on a rank's own
        # (non-zero) device -- the device guard, stream, and kernel load must follow the
        # runtime current device, not a baked index.
        torch._dynamo.reset()
        with torch.cuda.device(1):
            compiled = torch.compile(
                self._coor_inductor_fn, backend="inductor", fullgraph=True
            )
            out = compiled(torch.randn(2, 8, device="cuda:1"))
        self.assertEqual(out.device, torch.device("cuda:1"))

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_compiled_on_one_device_runs_on_another(self):
        # Problem 3 (shareable artifact): a graph first compiled on cuda:0 must produce a
        # correct result when the same code is compiled and run on cuda:1 with the on-disk
        # cache warm from the cuda:0 run.
        #
        # NB: this does NOT yet exercise reload-of-the-cuda:0-artifact. The FX graph cache
        # key embeds the input device (FxGraphCachePickler normalizes indices only under
        # device_id_agnostic=True, which the real key does not use), so the cuda:1 compile
        # misses and rebuilds. Cross-rank reuse of one artifact needs a device-agnostic
        # key, which in turn needs CompiledFxGraph.device_idxs to stop carrying the
        # compile-time index -- a follow-up, not something this PR implements. The
        # miss is asserted below so that landing the device-agnostic key trips this test
        # instead of silently changing what it covers.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import clear_caches, fresh_cache

        inp1 = torch.randn(2, 8, device="cuda:1")
        ref = self._coor_inductor_fn(inp1)
        with fresh_cache():
            with torch.cuda.device(0):
                compiled = torch.compile(
                    self._coor_inductor_fn, backend="inductor", fullgraph=True
                )
                compiled(torch.randn(2, 8, device="cuda:0"))  # populate cache on cuda:0
            # Drop in-memory caches (keeping the on-disk bundle) so the cuda:1 run reloads
            # from disk -- simulating a fresh per-rank process rather than reusing the
            # cuda:0 launcher in memory.
            torch._dynamo.reset()
            clear_caches()
            counters.clear()
            with torch.cuda.device(1):
                compiled = torch.compile(
                    self._coor_inductor_fn, backend="inductor", fullgraph=True
                )
                out = compiled(inp1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(out.device, torch.device("cuda:1"))
        self.assertEqual(out, ref)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_shared_kernel_reused_in_process_across_devices(self):
        # A rank only ever drives one device, but CooR's kernel cache key is
        # device-agnostic, so within one process the in-memory autotuner hands the same
        # loaded launcher to whatever device is current. A loaded CUfunction is
        # device-bound, so the launcher must keep per-device handles; otherwise a kernel
        # first loaded on cuda:0 and then launched on a cuda:1 stream raises `invalid
        # resource handle`. This is not the production execution model, but it is
        # reachable from any process that compiles for two devices -- including this
        # test file -- and it only shows up on a cold cache, so it is pinned here.
        from torch._inductor.utils import clear_caches, fresh_cache

        inp0 = torch.randn(2, 8, device="cuda:0")
        ref0 = self._coor_inductor_fn(inp0)
        inp1 = torch.randn(2, 8, device="cuda:1")
        ref1 = self._coor_inductor_fn(inp1)
        torch._dynamo.reset()
        clear_caches()
        with fresh_cache():
            compiled = torch.compile(
                self._coor_inductor_fn, backend="inductor", fullgraph=True
            )
            with torch.cuda.device(0):
                out0 = compiled(inp0)
            # The same in-process autotuner (loaded on cuda:0) now launches on cuda:1.
            with torch.cuda.device(1):
                out1 = compiled(inp1)
        self.assertEqual(out0.device, torch.device("cuda:0"))
        self.assertEqual(out1.device, torch.device("cuda:1"))
        self.assertEqual(out0, ref0)
        self.assertEqual(out1, ref1)

    @pytest.mark.multigpu
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_user_defined_triton_kernel_reused_in_process_across_devices(self):
        # Same cross-device reuse as the test above, for a user-defined triton.jit
        # kernel. That one passes only because an inductor-generated kernel gets the
        # static launcher, which keeps its handles per device. USER_AUTOTUNE is refused
        # by check_can_launch unless static_launch_user_defined_triton_kernels is set,
        # and that defaults off, so these kernels fall back to TritonCompileResult --
        # which binds one CUfunction at _init_handles() time and reuses it for every
        # later launch.
        #
        # Before the index was dropped here, the baked DeviceProperties(index=N) gave
        # each device its own cache key and hence its own autotuner, so the single
        # baked function was never reached. Dropping it is what makes one artifact
        # serve both devices.
        from torch._inductor.utils import clear_caches, fresh_cache

        fn = self._make_coor_user_defined_triton_fn()
        inp0 = torch.randn(128, device="cuda:0")
        inp1 = torch.randn(128, device="cuda:1")
        torch._dynamo.reset()
        clear_caches()
        with fresh_cache():
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            with torch.cuda.device(0):
                out0 = compiled(inp0)
            # The same in-process autotuner (loaded on cuda:0) now launches on cuda:1.
            with torch.cuda.device(1):
                out1 = compiled(inp1)
        self.assertEqual(out0.device, torch.device("cuda:0"))
        self.assertEqual(out1.device, torch.device("cuda:1"))
        self.assertEqual(out0, inp0 + 1)
        self.assertEqual(out1, inp1 + 1)


instantiate_parametrized_tests(TestCompileOnOneRankDeviceAsParameter)


def _baked_pg_constants(gm):
    """get_attr nodes that resolve to a torchbind ProcessGroup baked onto the gm.

    These are unserializable: torch.classes.c10d.ProcessGroup has no
    __getstate__, so GraphPickler.dumps fails on them.
    """
    out = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        val = gm
        for part in node.target.split("."):
            val = getattr(val, part)
        if isinstance(val, torch.ScriptObject) and "ProcessGroup" in val._type().name():
            out.append(node.target)
    return out


def _call_targets(gm):
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


# GraphPickler metadata filter mirroring graph_trainer's distributed filter:
# distributed ops (mesh_get_process_group) keep a real ProcessGroup in
# node.meta["val"]/["eager_input_vals"], which is not picklable and not needed.
def _drop_distributed_meta(key):
    return key not in (
        "val",
        "eager_input_vals",
        "source_fn_stack",
        "nn_module_stack",
        "fwd_source_fn_stack",
    )


@unittest.skipIf(not dist.is_available(), "distributed not available")
class TestCompileOnOneRankLegacyCollective(TestCase):
    """Legacy in-place c10d collectives (dist.all_reduce) under compile_on_one_rank.

    The in-place op ``c10d.allreduce_`` binds the ProcessGroup directly, so make_fx
    bakes it onto the GraphModule as a torchbind constant that GraphPickler cannot
    serialize. Under compile_on_one_rank two things change so the group flows into
    the graph from the (input) mesh instead of being baked in:
      - DeviceMesh.get_group() emits a mesh_get_process_group op, and
      - legacy collectives are remapped to functional collectives that take the
        group as an op argument.
    Single process with a fake PG -- this is the failing precompile CI step.
    """

    def setUp(self):
        super().setUp()
        self.store = FakeStore()
        dist.init_process_group(backend="fake", store=self.store, rank=0, world_size=2)
        self.mesh = init_device_mesh("cpu", (2,))

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    @staticmethod
    def _fn(t, mesh):
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=mesh.get_group())
        return t + 1

    @compiler_config.patch(compile_on_one_rank=True)
    def test_legacy_all_reduce_serializes_under_coor(self):
        gm = make_fx(self._fn, tracing_mode="fake")(torch.arange(4.0), self.mesh)
        targets = _call_targets(gm)

        # Legacy in-place collective is remapped to a functional collective whose
        # group comes from the mesh in-graph; nothing is baked.
        self.assertIn("_dtensor.mesh_get_process_group.default", targets)
        self.assertIn("_c10d_functional.all_reduce.default", targets)
        self.assertNotIn("c10d.allreduce_.default", targets)
        self.assertEqual(_baked_pg_constants(gm), [])

        mgpg = [n for n in gm.graph.nodes if "mesh_get_process_group" in str(n.target)]
        self.assertTrue(mgpg and all(n.users for n in mgpg))

        # Serializes once the distributed node metadata is stripped (the actual
        # failure mode: a baked torchbind ProcessGroup would raise here).
        GraphPickler.dumps(
            gm,
            Options(ops_filter=None, node_metadata_key_filter=_drop_distributed_meta),
        )

    @staticmethod
    def _rs_fn(t, mesh):
        out = torch.empty_like(t)
        group = mesh.get_group()
        dist.reduce_scatter(out, [t, t + 1], op=dist.ReduceOp.MAX, group=group)
        return out + 1

    @compiler_config.patch(compile_on_one_rank=True)
    def test_legacy_reduce_scatter_serializes_under_coor(self):
        gm = make_fx(self._rs_fn, tracing_mode="fake")(torch.arange(4.0), self.mesh)
        targets = _call_targets(gm)

        self.assertIn("_dtensor.mesh_get_process_group.default", targets)
        self.assertIn("_c10d_functional.reduce_scatter_tensor.default", targets)
        self.assertNotIn("c10d.reduce_scatter_.default", targets)
        self.assertEqual(_baked_pg_constants(gm), [])

        GraphPickler.dumps(
            gm,
            Options(ops_filter=None, node_metadata_key_filter=_drop_distributed_meta),
        )

    def test_default_path_bakes_pg(self):
        # Without compile_on_one_rank the legacy in-place op is unchanged and bakes
        # the ProcessGroup as a torchbind constant (the gated-against behavior).
        gm = make_fx(self._fn, tracing_mode="fake")(torch.arange(4.0), self.mesh)
        self.assertIn("c10d.allreduce_.default", _call_targets(gm))
        self.assertTrue(_baked_pg_constants(gm))


if __name__ == "__main__":
    run_tests()
