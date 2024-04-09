#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import Mock, patch

import torch.distributed.run as launch

import torch.testing._internal.common_utils as testing_common
from torch.distributed.elastic.utils.distributed import get_free_port

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


# TODO: merge with run_test.py after it is fixed
class TorchrunSupervisorLaunchTest(testing_common.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    from typing import Dict, List, Optional

    def _build_supervisor_args(
        self,
        drop_args: Optional[List[str]] = None,
        set_value: Optional[Dict[str, str]] = None,
        additional: Optional[List[str]] = None,
    ):
        drop_args = [] if drop_args is None else drop_args
        set_value = {} if set_value is None else set_value
        additional = [] if additional is None else additional

        args = {
            "--nnodes": "1",
            "--nproc-per-node": "8",
            "--rdzv-id": "rdzv_id",
            "--monitor-interval": "1",
            "--rdzv-endpoint": "tcp://localhost:55555",
            "--rdzv-conf": "root=True",  # start as a supervisor
        }
        for drop_arg in drop_args:
            del args[drop_arg]

        for k, v in set_value.items():
            args[k] = v

        args = [f"{k}={str(v)}" for k, v in args.items()]

        script_args = [
            "--rdzv-backend=supervisor",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]

        args = args + script_args + additional

        return args

    @patch("torch.distributed.elastic.supervisor.launchers.Context")
    @patch("subprocess.Popen")
    def test_supervisor_args(self, mock_popen, mock_ctx):
        host_process_mock = Mock()
        mock_popen.return_value = host_process_mock
        host_process_mock.wait.return_value = 0
        specs = {
            "endpoint_required_for_multinode_job": [
                self._build_supervisor_args(
                    drop_args=["--rdzv-endpoint"], set_value={"--nnodes": 2}
                ),
                ValueError,
            ],
            "endpoint_format": [
                self._build_supervisor_args(
                    set_value={"--rdzv-endpoint": "hostonly_no_port"}
                ),
                ValueError,
            ],
        }

        for name, (args, exception) in specs.items():
            with self.subTest(msg=name, args=args, exception=exception):
                with self.assertRaises(exception):
                    launch.main(args)

    @patch("torch.distributed.elastic.supervisor.launchers.Context", autospec=True)
    @patch("subprocess.Popen", autospec=True)
    @patch("socket.getfqdn", return_value="host_fqdn")
    @patch.dict(
        "torch.distributed.elastic.supervisor.launchers.policy_registry",
        {"default": Mock()},
    )
    def test_supervisor_endpoint_arg(
        self,
        mock_treat_as_supervisor: Mock,
        mock_popen: Mock,
        mock_ctx: Mock,
    ):
        host_process_mock = Mock()
        mock_popen.return_value = host_process_mock
        host_process_mock.wait.return_value = 0

        args = self._build_supervisor_args(drop_args=["--rdzv-endpoint"])
        launch.main(args)
        mock_ctx.assert_called_with(port=55555)
        self.assertEqual(mock_popen.call_args.args[0][-1], "tcp://host_fqdn:55555")

        args = self._build_supervisor_args(set_value={"--rdzv-endpoint": "host1:54321"})
        launch.main(args)
        mock_ctx.assert_called_with(port=54321)
        self.assertEqual(mock_popen.call_args.args[0][-1], "tcp://host1:54321")

    @patch("socket.getfqdn", return_value="host_fqdn")
    # @patch("torch.distributed.elastic.supervisor.launchers.hostmanager")
    @patch("torch.distributed.elastic.supervisor.hostmanager.main")
    def test_supervisor_hostmanager_role_endpoint_arg(
        self, hostmanager_mock: Mock, mock_treat_as_hostmanager: Mock
    ):
        args = self._build_supervisor_args(
            set_value={"--rdzv-endpoint": "other_host:54321"}, drop_args=["--rdzv-conf"]
        )
        launch.main(args)
        hostmanager_mock.assert_called_with("tcp://other_host:54321")

    def test_supervisor_launch_single_node(self):
        run_id = str(uuid.uuid4().int)
        nproc_per_node = 4
        world_size = nproc_per_node

        port = get_free_port()
        host = "localhost"
        import socket

        host = socket.getfqdn()

        args = [
            f"--rdzv-endpoint={host}:{port}",
            "--nnodes=1",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=supervisor",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]

        launch.main(args)

        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_supervisor_launch_multinode(self):
        port = get_free_port()
        host = "localhost"
        run_id = str(uuid.uuid4().int)
        nnodes = 3
        nproc_per_node = 2
        world_size = nnodes * nproc_per_node
        args = {
            "--nnodes": f"{nnodes}",
            "--nproc-per-node": f"{nproc_per_node}",
            "--rdzv-endpoint": f"{host}:{port}",
        }
        hostmanagers_args = self._build_supervisor_args(
            set_value={**args, "--rdzv-conf": "root=False'"}
        )
        supervisor_args = self._build_supervisor_args(
            set_value=args,
        )

        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[hostmanagers_args])
            procs.append(p)
            p.start()

        # # supervisor
        p = mp.Process(target=launch.main, args=[supervisor_args])
        procs.append(p)
        p.start()

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # # make sure all the workers ran
        # # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_supervisor_launch_retry(self):
        port = get_free_port()
        host = "localhost"
        run_id = str(uuid.uuid4().int)
        nnodes = 3
        nproc_per_node = 2
        world_size = nnodes * nproc_per_node

        max_restarts = 2
        args = {
            "--nnodes": f"{nnodes}",
            "--nproc-per-node": f"{nproc_per_node}",
            "--rdzv-endpoint": f"{host}:{port}",
            "--max-restarts": f"{max_restarts}",
        }
        hostmanagers_args = self._build_supervisor_args(
            set_value={**args, "--rdzv-conf": "root=False'"}
        )
        supervisor_args = self._build_supervisor_args(
            # set_value={**args, "--rdzv-conf": "join_timeout=10"}, additional=["--fail"]
            set_value=args,
            additional=["--fail"],
        )

        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[hostmanagers_args])
            procs.append(p)
            p.start()

        # # # supervisor
        p = mp.Process(target=launch.main, args=[supervisor_args])
        procs.append(p)
        p.start()

        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(1, p.exitcode, f"Node process {i} did not fail")

        # check attempts, not all files will be generates due to cleanup stage on first failure
        def extract_attempt_idx(S: str):
            return int(S[(len("attempt_")) :].split("_")[0])

        attempts = {extract_attempt_idx(file) for file in os.listdir(self.test_dir)}
        self.assertSetEqual({0, 1, 2}, attempts)

    def test_supervisor_launch_elastic_min(self):
        port = get_free_port()
        host = "localhost"
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 3
        N = min_nodes

        nproc_per_node = 2
        world_size = N * nproc_per_node
        args = {
            "--nnodes": f"{min_nodes}:{max_nodes}",
            "--nproc-per-node": f"{nproc_per_node}",
            "--rdzv-endpoint": f"{host}:{port}",
        }

        supervisor_args = self._build_supervisor_args(
            set_value={**args, "--rdzv-conf": "join_timeout=5,root=True"}
        )

        procs = []
        # run only a supervisor that will fork one hostmanager
        p = mp.Process(target=launch.main, args=[supervisor_args])
        procs.append(p)
        p.start()

        for i in range(N):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_supervisor_launch_elastic_max(self):
        port = get_free_port()
        host = "localhost"
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 3
        N = max_nodes

        nproc_per_node = 2
        world_size = N * nproc_per_node
        args = {
            "--nnodes": f"{min_nodes}:{max_nodes}",
            "--nproc-per-node": f"{nproc_per_node}",
            "--rdzv-endpoint": f"{host}:{port}",
        }
        hostmanagers_args = self._build_supervisor_args(
            set_value={**args, "--rdzv-conf": "root=False'"}
        )
        supervisor_args = self._build_supervisor_args(
            set_value={**args, "--rdzv-conf": "join_timeout=5,root=True"}
        )

        procs = []

        for _ in range(N - 1):
            p = mp.Process(target=launch.main, args=[hostmanagers_args])
            procs.append(p)
            p.start()

        # # supervisor
        p = mp.Process(target=launch.main, args=[supervisor_args])
        procs.append(p)
        p.start()

        for i in range(N):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_supervisor_no_python(self):
        run_id = str(uuid.uuid4().int)
        nproc_per_node = 4
        world_size = nproc_per_node

        port = get_free_port()
        host = "localhost"
        import socket

        host = socket.getfqdn()

        args = [
            f"--rdzv-endpoint={host}:{port}",
            "--nnodes=1",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=supervisor",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--no-python",
            path("bin/test_script.sh"),
            f"{self.test_dir}",
        ]

        launch.main(args)

        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )


if __name__ == "__main__":
    testing_common.run_tests()
