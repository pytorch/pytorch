# Owner(s): ["oncall: distributed"]

import difflib
import functools
import os
import subprocess
import sys
import textwrap
import unittest

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

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_string_identical_across_devices_under_coor(self):
        # The serialized guard is part of a precompile artifact, so it has to match
        # across ranks even if the runtime check itself were already device-relative.
        def guards_on(dev):
            with torch.cuda.device(dev):
                torch._dynamo.reset()

                def f(x):
                    return x + 1

                torch.compile(f, backend="eager")(torch.randn(4, device=f"cuda:{dev}"))
                return self._tensor_guard_parts(f)

        self.assertEqual(guards_on(0), guards_on(1))

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_guard_still_rejects_noncurrent_device_index_under_coor(self):
        # Relaxing the index must not mean ignoring it: a tensor on a device that is
        # not the current one still has to miss the cache. Deleting the check outright
        # would silently pass here.
        from torch._dynamo.testing import CompileCounter

        def f(x):
            return x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        with torch.cuda.device(0):
            compiled = torch.compile(f, backend=cnt)
            compiled(torch.randn(4, device="cuda:0"))
            before = cnt.frame_count
            compiled(torch.randn(4, device="cuda:1"))
            self.assertEqual(
                cnt.frame_count,
                before + 1,
                "a tensor on a non-current device must still fail the guard",
            )

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

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    @torch._inductor.config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON"
    )
    def test_inductor_template_no_baked_device(self):
        # A Triton template must not bake the rank-specific device index either.
        #
        # The device-index drop is applied where TritonKernel builds triton_meta
        # (codegen/triton.py), but a template's triton_meta is built separately in
        # select_algorithm.py and calls DeviceProperties.create() on the concrete
        # device, so it still emits DeviceProperties(..., index=N). The same gap
        # exists in triton_combo_kernel.py.
        #
        # test_inductor_compiles_under_coor does not catch this: _coor_inductor_fn
        # is a factory plus a reduction, which only produces inductor-generated
        # kernels and never reaches the template path.
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        compiled = torch.compile(
            self._coor_template_fn, backend="inductor", fullgraph=True
        )
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        _, codes = run_and_get_code(compiled, a, b)
        self._assert_no_baked_device("\n".join(codes))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_user_defined_triton_kernel_no_baked_device(self):
        # A user-defined @triton.jit kernel gets its triton_meta from a third site,
        # define_user_defined_triton_kernel in codegen/wrapper.py, which is neither
        # the TritonKernel path nor the template path.
        import triton
        import triton.language as tl

        from torch._inductor.utils import run_and_get_code

        @triton.jit
        def add_one_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask) + 1, mask=mask)

        def fn(x):
            out = torch.empty_like(x)
            add_one_kernel[(1,)](x, out, x.numel(), BLOCK=128)
            return out

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(128, device="cuda")
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

    def _inductor_code_on_device(self, dev):
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        with torch.cuda.device(dev):
            compiled = torch.compile(
                self._coor_inductor_fn, backend="inductor", fullgraph=True
            )
            _, codes = run_and_get_code(
                compiled, torch.randn(2, 8, device=f"cuda:{dev}")
            )
        return "\n".join(codes)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_code_identical_across_devices(self):
        # The inductor-side rewrite is a dozen independent opt-in `if _coor_enabled():`
        # sites with no structural funnel, so pattern-matching one rank's output cannot
        # show that none was missed. Diffing the code generated on two different devices
        # can. benchmark_kernel is enabled because that harness is emitted into the
        # kernel's own source (hence its hash), so a baked index there breaks
        # rank-identity too.
        import re

        # The AOT id is a per-process compile counter ("0_inference", "1_inference"), not
        # a rank-specific value -- it differs only because both compiles run in this one
        # process, so normalize it before diffing.
        def norm(s):
            return re.sub(r"AOT ID: \['\d+_", "AOT ID: ['N_", s)

        # Each config emits device references through a different path: the kernel's own
        # benchmark harness, and the compile-time autotune harness
        # (generate_example_value). Both are part of generated text that must match.
        for cfg in (
            {"benchmark_kernel": True},
            {"triton.autotune_at_compile_time": True},
        ):
            with self.subTest(cfg=cfg), torch._inductor.config.patch(**cfg):
                code0 = self._inductor_code_on_device(0)
                code1 = self._inductor_code_on_device(1)
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
                        f"inductor code differs across devices under CooR "
                        f"with {cfg}:\n{diff}"
                    )
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
                return self._inductor_code_on_device(0)

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
