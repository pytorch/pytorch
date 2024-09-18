#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import multiprocessing as mp
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import closing, redirect_stderr, redirect_stdout
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import torch.distributed.run as launch
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.utils import get_socket_with_port
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


def launch_in_proc(args):
    launch.main(args)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


def get_child_pids(pid):
    pgrep = subprocess.Popen(args=f"pgrep -P {pid}", shell=True, stdout=subprocess.PIPE)
    pgrep.wait()
    out = pgrep.stdout.read().decode("utf-8").rstrip().split("\n")
    pids = []
    for pid in out:
        if pid:
            pids.append(int(pid))
    return pids


def pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class MockException(Exception):
    pass


class ElasticLaunchTest(TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # remove any lingering environment variables
        for env in os.environ.keys():
            if env.startswith("PET_"):
                del os.environ[env]

        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_launch_user_script_python(self):
        self._test_launch_user_script_python()

    def _test_launch_user_script_python(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_launch_user_script_python_caffe2_bc(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_user_script_bash(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            launch.main(args + ["--module"] + script_args)

        launch.main(args + script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_user_script_default_nproc(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        world_size = 1
        args = [
            f"--nnodes={nnodes}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            launch.main(args + ["--module"] + script_args)

        launch.main(args + script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_with_env_vars(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node

        os.environ["PET_NNODES"] = str(nnodes)
        os.environ["PET_NPROC_PER_NODE"] = str(nproc_per_node)
        os.environ["PET_RDZV_ID"] = run_id
        os.environ["PET_MONITOR_INTERVAL"] = "1"
        os.environ["PET_START_METHOD"] = "spawn"
        os.environ["PET_NO_PYTHON"] = "1"

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            os.environ["PET_MODULE"] = "1"
            launch.main(script_args)

        os.environ["PET_MODULE"] = "0"
        launch.main(script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def _test_nproc_launch_configuration(self, nproc_type, expected_number):
        run_id = str(uuid.uuid4().int)
        nnodes = 1

        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_type}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        launch.main(args + script_args)

        world_size = nnodes * expected_number
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=False)
    def test_nproc_launch_auto_configurations(self, _mock1):
        self._test_nproc_launch_configuration("auto", os.cpu_count())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_number_configurations(self):
        self._test_nproc_launch_configuration("4", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_unknown_configurations(self):
        with self.assertRaises(ValueError):
            self._test_nproc_launch_configuration("unknown", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=3)
    def test_nproc_gpu_launch_configurations(self, _mock1, _mock2):
        self._test_nproc_launch_configuration("auto", 3)
        self._test_nproc_launch_configuration("gpu", 3)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # we are only launching 1 node (even though max = 2)
        world_size = nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @mock.patch("torch.distributed.elastic.events.record")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        """
        Asserts that when the worker program fails and lancher raieses exception
        to indicate that worker process failed

        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            "--fail",
        ]
        with self.assertRaises(ChildFailedError):
            launch.main(args)

        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception
        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv_conf=timeout=5",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]

        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            launch.main(args)
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_standalone(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--standalone",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_run_path(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            "--run-path",
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv_conf=timeout=5",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[args])
            procs.append(p)
            p.start()
        launch.main(args)
        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_min_max_nodes_parse(self):
        min_nodes, max_nodes = launch.parse_min_max_nnodes("1")
        self.assertEqual(min_nodes, max_nodes)
        self.assertEqual(1, min_nodes)
        min_nodes, max_nodes = launch.parse_min_max_nnodes("2:20")
        self.assertEqual(2, min_nodes)
        self.assertEqual(20, max_nodes)
        with self.assertRaises(RuntimeError):
            launch.parse_min_max_nnodes("2:20:30")

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    def test_launch_shutdown(self, agent_mock_cls):
        nnodes = 1
        nproc_per_node = 4
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            launch.main(args)
            rdzv_handler_mock.shutdown.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched(self):
        # launch test script with torchelastic and validate that
        # torch.distributed.is_torchelastic_launched() returns True

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )

        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual("True", is_torchelastic_launched)

    @patch("torch.distributed.run.metadata")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched_with_logs_spec_defined(self, metadata_mock):
        # mock the entrypoint API to avoid version issues.
        entrypoints = MagicMock()
        metadata_mock.entry_points.return_value = entrypoints

        group = MagicMock()
        entrypoints.select.return_value = group

        ep = MagicMock()
        ep.load.return_value = DefaultLogsSpecs

        group.select.return_value = ep
        group.__getitem__.return_value = ep

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        if os.path.exists(out_file):
            os.remove(out_file)
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                "--logs_specs=default",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )

        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual("True", is_torchelastic_launched)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_logs_logs_spec_entrypoint_must_be_defined(self):
        with self.assertRaises(ValueError):
            launch.main(
                [
                    "--run-path",
                    "--nnodes=1",
                    "--nproc-per-node=1",
                    "--monitor-interval=1",
                    "--logs_specs=DOESNOT_EXIST",
                    path("bin/test_script_is_torchelastic_launched.py"),
                ]
            )

    def test_is_not_torchelastic_launched(self):
        # launch test script without torchelastic and validate that
        # torch.distributed.is_torchelastic_launched() returns False

        out_file = f"{os.path.join(self.test_dir, 'out')}"

        # need to run the script with runpy in the same interpreter
        # as the test because otherwise (depending on the environment)
        # it will not find torch as a dependency
        with patch.object(
            sys,
            "argv",
            [
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ],
        ):
            runpy.run_path(sys.argv[0], run_name="__main__")
            with open(out_file) as fp:
                is_torchelastic_launched = fp.readline()
                self.assertEqual("False", is_torchelastic_launched)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_init_method_tcp_with_torchelastic(self):
        port = get_free_port()
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=localhost",
                f"--master-port={port}",
                "--monitor-interval=1",
                path("bin/test_script_init_method.py"),
                f"--init-method=tcp://localhost:{port}",
            ]
        )
        # nothing to validate, just make sure it runs

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_init_method_env_with_torchelastic(self):
        port = get_free_port()
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=localhost",
                f"--master-port={port}",
                "--monitor-interval=1",
                path("bin/test_script_init_method.py"),
                "--init-method=env://",
            ]
        )
        # nothing to validate, just make sure it runs

    def test_capture_logs_using_default_logs_specs(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--redirect=3",
            "--tee=3",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            with patch.dict(
                os.environ, {"TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": "[rank${rank}]: "}
            ):
                launch.main(args + script_args)

        for i in range(nproc_per_node):
            self.assertTrue(f"[rank{i}]: creating " in captured_out.getvalue())


if __name__ == "__main__":
    run_tests()
