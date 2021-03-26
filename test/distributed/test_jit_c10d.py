import unittest
import tempfile
import sys
import torch
import torch.distributed as c10d
import time
from datetime import timedelta
from typing import List

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import requires_nccl
from torch.testing._internal.common_utils import load_tests, TEST_WITH_TSAN, run_tests, IS_WINDOWS
from torch.testing._internal.jit_utils import JitTestCase

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)

if sys.platform == 'darwin':
    LOOPBACK = 'lo0'
else:
    LOOPBACK = 'lo'

def unique_process_group_name(prefix):
    # Append timestamp to process group name to make it unique, so
    # that when tests run multiple times or in parallel there
    # wouldn't be name conflicts.
    now = int(time.time() * 1000)
    return "%s_%d" % (prefix, now)

def _create_tcp_store():
    addr = "localhost"
    port = common.find_free_port()
    timeout = timedelta(minutes=5)
    timeout_millisecond = int(timeout / timedelta(milliseconds=1))
    return torch.classes.dist_c10d.TCPStore(addr, port, 1, True, timeout_millisecond)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
@unittest.skipIf(IS_WINDOWS, "TCPStore not available on Windows")
class ProcessGroupNCCLJitTest(JitTestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def _create_nccl_pg(self, name_prefix):
        tcp_store = _create_tcp_store()
        opts = torch.classes.dist_c10d.ProcessGroupNCCLOptions(0, True)

        name = unique_process_group_name(name_prefix)

        return torch.classes.dist_c10d.ProcessGroupNCCL(tcp_store, self.rank, self.world_size, opts, name)

    def _create_nccl_pg_as_base_process_group(self, name):
        tcp_store = _create_tcp_store()

        return torch.classes.dist_c10d.frontend().new_process_group_helper(
            self.world_size, self.rank, [], "nccl", tcp_store, name, 0)

    @requires_nccl()
    def test_init_process_group_nccl_torchbind(self):
        self._create_nccl_pg("raw_process_group_nccl_torchbind")

    @requires_nccl()
    def test_process_group_nccl_torchbind_alltoall(self):
        nccl_pg = self._create_nccl_pg("process_group_nccl_as_base_class")

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
    def test_init_process_group_nccl_as_base_process_group_torchbind(self):
        name = unique_process_group_name("creation_test_process_group")
        self._create_nccl_pg_as_base_process_group(name)

    @requires_nccl()
    def test_process_group_nccl_as_base_process_group_torchbind_alltoall(self):
        name = unique_process_group_name("alltoall_test_process_group")
        nccl_pg = self._create_nccl_pg_as_base_process_group(name)

        input = torch.rand(16).cuda()
        output = torch.rand(16).cuda()

        @torch.jit.script
        def run_pg_nccl_alltoall(
            pg: torch.classes.dist_c10d.ProcessGroup,
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

        pg_nccl = self._create_nccl_pg("nccl_process_group_as_module_member")
        self.checkModule(TestModule(pg_nccl), (torch.rand((2, 3)),))


class StoreTest(JitTestCase):
    def setUp(self):
        super(StoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filestore = torch.classes.dist_c10d.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"

    def test_create_file_store(self):
        # test FileStore creation in JIT
        @torch.jit.script
        def create_file_store(
            path: str,
            num_workers: int
        ) -> torch.classes.dist_c10d.FileStore:
            return torch.classes.dist_c10d.FileStore(path, num_workers)

        create_file_store(self.file.name, 1)

    def test_create_prefix_store(self):
        # test PrefixStore creation in JIT
        @torch.jit.script
        def create_prefix_file_store(
            store: torch.classes.dist_c10d.Store,
            prefix: str
        ) -> torch.classes.dist_c10d.PrefixStore:
            return torch.classes.dist_c10d.PrefixStore(prefix, store)

        create_prefix_file_store(self.filestore, self.prefix)


@unittest.skipIf(IS_WINDOWS, "TCPStore not available on Windows")
class C10dFrontendJitTest(JitTestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    @requires_nccl()
    def test_frontend_singleton(self):
        frontend1 = torch.classes.dist_c10d.frontend()
        frontend2 = torch.classes.dist_c10d.frontend()

        tcp_store = _create_tcp_store()

        pg_name = unique_process_group_name("singleton_test_process_group")

        ProcessGroupNCCL1 = frontend1.new_process_group_helper(
            self.world_size, self.rank, [], "nccl", tcp_store, pg_name, 0)

        ProcessGroupNCCL2 = frontend2.get_process_group_by_name(pg_name)
        self.assertEqual(frontend2.get_name_of_process_group(ProcessGroupNCCL2), pg_name)

@unittest.skipIf(IS_WINDOWS, "TCPStore not available on Windows")
class C10dProcessGroupSerialization(JitTestCase):
    def setUp(self):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    @requires_nccl()
    def test_process_group_as_module_member(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                tcp_store = _create_tcp_store()

                name = unique_process_group_name("module_member_process_group")
                self.pg = torch.classes.dist_c10d.frontend().new_process_group_helper(
                    1, 0, [], "nccl", tcp_store, name, 0)

            def forward(self, input: torch.Tensor):
                if self.pg is None:
                    return input + 1
                else:
                    return input + 2

        self.checkModule(TestModule(), (torch.rand((2, 3)),))


if __name__ == "__main__":
    run_tests()
