# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10dBackendTest,
    CUDA_BACKENDS,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests


class AbstractCUDAGraphsTest(C10dBackendTest):
    def test_all_reduce(self):
        self._init_pg()
        warmup = torch.ones(1, device=self.device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize()

        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        expected = float(sum(range(1, self.world_size + 1)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(tensor)
        self.assertEqual(tensor, torch.full_like(tensor, float(self.rank + 1)))

        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(tensor, torch.full_like(tensor, expected))

        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(tensor, torch.full_like(tensor, expected * self.world_size))


instantiate_backend_tests(
    globals(), "CUDAGraphs", AbstractCUDAGraphsTest, CUDA_BACKENDS
)


if __name__ == "__main__":
    run_tests()
