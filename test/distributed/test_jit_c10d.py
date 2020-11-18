import unittest
from sys import platform
import torch
import torch.distributed as c10d

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import requires_nccl, skip_if_rocm_single_process
from torch.testing._internal.common_utils import TestCase, load_tests, TEST_WITH_TSAN

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)


if platform == 'darwin':
    LOOPBACK = 'lo0'
else:
    LOOPBACK = 'lo'

@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupNCCLJitTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1

    def _create_nccl_pg(self):
        addr = "localhost"
        port = common.find_free_port()
        tcp_store = torch.classes.dist_c10d.TCPStore(addr, port, 1, True)
        opts = torch.classes.dist_c10d.ProcessGroupNCCLOptions(0, True)

        return torch.classes.dist_c10d.ProcessGroupNCCL(tcp_store, self.rank, self.world_size, opts)  

    @requires_nccl()
    @skip_if_rocm_single_process
    def test_init_process_group_nccl_torchbind(self):
        self._create_nccl_pg()

    @requires_nccl()
    @skip_if_rocm_single_process
    def test_process_group_nccl_torchbind_alltoall(self):
        nccl_pg = self._create_nccl_pg()

        input = torch.rand(16).cuda()
        output = torch.rand(16).cuda()

        @torch.jit.script
        def run_pg_nccl_alltoall(
            pg: torch.classes.dist_c10d.ProcessGroupNCCL,
            output: torch.Tensor,
            input: torch.Tensor
        ):
            output_split_sizes: List[int] = []
            input_split_sizes: List[int] = []
            work = pg.alltoall_base(output, input, output_split_sizes, input_split_sizes)
            work.wait()
            return work.result()

        run_pg_nccl_alltoall(nccl_pg, output, input)
