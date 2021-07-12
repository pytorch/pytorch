#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import tempfile
import unittest
from contextlib import closing

import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port
from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_launch_without_env(self):
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
            "--start_method=fork",
            "--master_addr=localhost",
            f"--master_port={master_port}",
            "--node_rank=0",
            path("bin/test_script_local_rank.py"),
        ]
        launch.main(args)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_launch_with_env(self):
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
            "--start_method=fork",
            "--master_addr=localhost",
            f"--master_port={master_port}",
            "--node_rank=0",
            "--use_env",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        launch.main(args)
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )
