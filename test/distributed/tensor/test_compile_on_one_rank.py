# Owner(s): ["oncall: distributed"]

import difflib
import functools
import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
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
    @dist_config.patch(compile_on_one_rank=True)
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
    @dist_config.patch(compile_on_one_rank=True)
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
    @dist_config.patch(compile_on_one_rank=True)
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
    @dist_config.patch(compile_on_one_rank=True)
    def test_all_reduce_with_implicit_world_group(self):
        """`dist.all_reduce(t)` with no `group=` (implicit `dist.group.WORLD`)
        should compile under compile_on_one_rank=True.

        `WorldMetaClassVariable.var_getattr` was routing the WORLD lookup through
        `SourcelessBuilder`, dropping the source it had just constructed for the
        guard. The resulting `TorchScriptObjectVariable` had the raw ProcessGroup
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
    device-agnostic. A device-agnostic graph derives device in-graph (via
    prim.device) and so has none of these.
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


class TestCompileOnOneRankDeviceAsParameter(TestCase):
    """Device-as-parameter for the make_fx tracing path used by graph_trainer/CooR.

    Under compile_on_one_rank, a factory op whose device is derived from an input
    tensor's .device should trace with its device= fed by an in-graph
    prim.device(input) node, instead of baking the input's concrete device index.
    That keeps the traced graph device-agnostic so one compiled artifact runs on
    each rank's real GPU without --virtual-local-rank.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @dist_config.patch(compile_on_one_rank=True)
    def test_factory_device_derived_from_input_not_baked(self):
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        targets = [n.target for n in gm.graph.nodes]
        self.assertIn(
            torch.ops.prim.device.default,
            targets,
            "device should be derived in-graph from the input via prim.device",
        )
        baked = _indexed_cuda_device_nodes(gm)
        self.assertEqual(
            baked,
            [],
            f"no node should bake a concrete indexed cuda device; found: {baked}",
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    @dist_config.patch(compile_on_one_rank=True)
    def test_factory_device_is_runtime_agnostic(self):
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        self.assertEqual(
            gm(torch.randn(2, 8, device="cuda:0")).device, torch.device("cuda:0")
        )
        self.assertEqual(
            gm(torch.randn(2, 8, device="cuda:1")).device, torch.device("cuda:1")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_default_path_unchanged_bakes_device(self):
        # Without compile_on_one_rank the device stays baked (the feature must be
        # gated so it does not perturb the default tracing path).
        gm = make_fx(_factory_from_input_device, tracing_mode="fake")(
            torch.randn(2, 8, device="cuda:0")
        )
        self.assertNotIn(
            torch.ops.prim.device.default, [n.target for n in gm.graph.nodes]
        )
        self.assertTrue(_indexed_cuda_device_nodes(gm))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @dist_config.patch(compile_on_one_rank=True)
    def test_to_copy_explicit_device_derived_from_input(self):
        # An explicit-device dtype cast (the SimpleFSDP mixed-precision pattern,
        # aten._to_copy with a device= kwarg) also gets its baked device rewired
        # to prim.device(input), alongside the factory-op path.
        def f(x):
            return x.to(device="cuda:0", dtype=torch.bfloat16)

        gm = make_fx(f, tracing_mode="fake")(torch.randn(2, 8, device="cuda:0"))
        self.assertIn(torch.ops.prim.device.default, [n.target for n in gm.graph.nodes])
        self.assertEqual(_indexed_cuda_device_nodes(gm), [])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @dist_config.patch(compile_on_one_rank=True)
    def test_no_input_on_device_raises(self):
        # A cuda factory in a graph whose only input is on CPU has no input to
        # derive the device from, so we raise rather than silently emit a
        # non-portable graph. (The general fix -- a synthesized device input
        # parameter supplied by the executor -- is a planned follow-up.)
        def f(x):
            return torch.zeros(x.shape[0], device="cuda:0")

        with self.assertRaisesRegex(RuntimeError, "no graph input is on that device"):
            make_fx(f, tracing_mode="fake")(torch.randn(2, device="cpu"))


if __name__ == "__main__":
    run_tests()
