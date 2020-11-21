import unittest
import tempfile
from sys import platform
import torch
import torch.distributed as c10d
from typing import List
from datetime import datetime,timezone

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import requires_nccl, skip_if_rocm_single_process
from torch.testing._internal.common_utils import load_tests, TEST_WITH_TSAN, run_tests
from torch.testing._internal.jit_utils import JitTestCase

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
class ProcessGroupNCCLJitTest(JitTestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def _create_nccl_pg(self):
        addr = "localhost"
        port = common.find_free_port()
        tcp_store = torch.classes.dist_c10d.TCPStore(addr, port, 1, True)
        opts = torch.classes.dist_c10d.ProcessGroupNCCLOptions(0, True)

        return torch.classes.dist_c10d.ProcessGroupNCCL(tcp_store, self.rank, self.world_size, opts)  

    def _create_nccl_pg_as_base_process_group(self, name):
        addr = "localhost"
        port = common.find_free_port()
        tcp_store = torch.classes.dist_c10d.TCPStore(addr, port, 1, True)

        return torch.classes.dist_c10d.frontend().new_process_group_helper(
            self.world_size, self.rank, [], "NCCL", tcp_store, name, 0)

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

    @requires_nccl()
    @skip_if_rocm_single_process
    def test_init_process_group_nccl_as_base_process_group_torchbind(self):
        # Append timestamp to process group name to make it unique, so that
        # when tests run multiple times or in parallel there wouldn't be
        # name conflicts.
        now = datetime.now(timezone.utc).timestamp()
        self._create_nccl_pg_as_base_process_group("creation_test_process_group_%d" % now)

#    @requires_nccl()
#    @skip_if_rocm_single_process
#    def test_process_group_nccl_as_base_process_group_torchbind_alltoall(self):
#        # Append timestamp to process group name to make it unique, so that
#        # when tests run multiple times or in parallel there wouldn't be
#        # name conflicts.
#        now = datetime.now(timezone.utc).timestamp()
#        nccl_pg = self._create_nccl_pg_as_base_process_group("alltoall_test_process_group_%d" % now)
#
#        input = torch.rand(16).cuda()
#        output = torch.rand(16).cuda()
#
#        @torch.jit.script
#        def run_pg_nccl_alltoall(
#            pg: torch.classes.dist_c10d.ProcessGroup,
#            output: torch.Tensor,
#            input: torch.Tensor
#        ):
#            output_split_sizes: List[int] = []
#            input_split_sizes: List[int] = []
#            work = pg.alltoall_base(output, input, output_split_sizes, input_split_sizes)
#            work.wait()
#            return work.result()
#
#        run_pg_nccl_alltoall(nccl_pg, output, input)

    def test_process_group_nccl_serialization(self):
        class TestModule(torch.nn.Module):
            def __init__(self, pg_nccl):
                super(TestModule, self).__init__()
                self.pg = pg_nccl

            def forward(self, input: torch.Tensor):
                if self.pg is None:
                    return input + 1
                else:
                    return input + 2

        self.checkModule(TestModule(self._create_nccl_pg()), torch.rand((2, 3)))


class C10dFrontendJitTest(JitTestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def test_frontend_singleton(self):
        frontend1 = torch.classes.dist_c10d.frontend()
        frontend2 = torch.classes.dist_c10d.frontend()

        addr = "localhost"
        port = common.find_free_port()
        tcp_store = torch.classes.dist_c10d.TCPStore(addr, port, 1, True)

        # Append timestamp to process group name to make it unique, so that
        # when tests run multiple times or in parallel there wouldn't be
        # name conflicts.
        now = datetime.now(timezone.utc).timestamp()
        pg_name = "singleton_test_process_group_%d" % now

        ProcessGroupNCCL1 = frontend1.new_process_group_helper(
            self.world_size, self.rank, [], "NCCL", tcp_store, pg_name, 0)

        ProcessGroupNCCL2 = frontend2.get_process_group_by_name(pg_name)
        self.assertEqual(ProcessGroupNCCL1, ProcessGroupNCCL2)

class C10dProcessGroupSerialization(JitTestCase):
    def test_process_group_as_module_member(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                addr = "localhost"
                port = common.find_free_port()
                tcp_store = torch.classes.dist_c10d.TCPStore(addr, port, 1, True)

                # Append timestamp to process group name to make it unique, so
                # that when tests run multiple times or in parallel there
                # wouldn't be name conflicts.
                now = datetime.now(timezone.utc).timestamp()
                self.pg = torch.classes.dist_c10d.frontend().new_process_group_helper(
                    1, 0, [], "NCCL", tcp_store, "module_member_process_group_%d" % now, 0)

            def forward(self, input: torch.Tensor):
                if self.pg is None:
                    return input + 1
                else:
                    return input + 2

        self.checkModule(TestModule(), torch.rand((2, 3)))


if __name__ == "__main__":
    run_tests()
