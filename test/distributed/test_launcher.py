# Owner(s): ["oncall: distributed"]

import os
import sys
from contextlib import closing

import torch.distributed as dist
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port
import subprocess

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
    run_tests,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


class TestDistributedLaunch(TestCase):
    def test_launch_user_script(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--monitor_interval=1",
            "--start_method=spawn",
            "--master_addr=localhost",
            f"--master_port={master_port}",
            "--node_rank=0",
            "--use_env",
            path("bin/test_script.py"),
        ]
        launch.main(args)

    def test_launch_deprecated_warning(self):
        script_path = path("bin/test_script.py")
        processes = []

        for node_rank in [0, 1]:
            launch_script = f'''
import torch
import torch.distributed.launch as launch
launch.main([
    '--nnodes', '2',
    '--node_rank', '{node_rank}',
    '{script_path}'])
'''

            processes.append(subprocess.Popen(
                ['python', '-c', launch_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE))

        for rank, process in enumerate(processes):
            process.wait()
            process_out, process_err = process.communicate()
            if rank == 0:
                self.assertIn('launch is deprecated', str(process_err))
            else:
                self.assertNotIn('launch is deprecated', str(process_err))

    def test_omp_threads_warning(self):
        script_path = path("bin/test_script.py")
        processes = []

        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']

        for node_rank in [0, 1]:
            processes.append(subprocess.Popen([
                'torchrun',
                '--nnodes', '2',
                '--nproc_per_node', '2',
                '--node_rank', str(node_rank),
                script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE))

        for rank, process in enumerate(processes):
            process.wait()
            process_out, process_err = process.communicate()
            if rank == 0:
                self.assertIn('Setting OMP_NUM_THREADS', str(process_err))
            else:
                self.assertNotIn('Setting OMP_NUM_THREADS', str(process_err))

if __name__ == "__main__":
    run_tests()
