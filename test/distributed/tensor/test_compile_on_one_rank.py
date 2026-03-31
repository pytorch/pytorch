# Owner(s): ["oncall: distributed"]

import difflib
import functools
import sys

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
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


if __name__ == "__main__":
    run_tests()
