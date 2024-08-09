#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None
import os
import subprocess
import sys

import time
from contextlib import contextmanager
from pathlib import Path

import torch.testing._internal.common_utils as testing_common

from torch.distributed.elastic.supervisor import Context


@contextmanager
def context(*args, **kwargs):
    try:
        ctx = Context(*args, **kwargs)
        yield ctx
    finally:
        ctx.shutdown()


@contextmanager
def host_sockets(N: int):
    context: zmq.Context = zmq.Context(1)

    def create_socket():
        backend = context.socket(zmq.DEALER)
        backend.setsockopt(zmq.IPV6, True)
        backend.connect("tcp://127.0.0.1:55555")
        return backend

    sockets = [create_socket() for i in range(N)]
    try:
        yield sockets
    finally:
        context.destroy(linger=500)


def emulate_launch(args, N: int = 4):
    # for testing using basic launcher
    def create_host(i):
        env = {**os.environ}
        env["HOSTNAMES"] = "localhost"
        env["TORCH_ELASTIC_SUPERVISOR"] = str(i == 0)
        # fast heartbeat so we do not wait so long to see the timeout
        # in the tests
        env["TORCH_SUPERVISOR_HEARTBEAT_INTERVAL"] = str(0.1)
        # dist-info with "test_launcher" entrypoint definition
        env["PYTHONPATH"] = str(Path(__file__).parent)
        return subprocess.Popen(args, env=env)

    hosts = [create_host(i) for i in range(N)]
    expiry = time.time() + 20
    try:
        r = [h.wait(timeout=max(0, expiry - time.time())) for h in hosts]
        return r[0]
    finally:
        for h in hosts:
            if h.poll() is None:
                h.kill()


class SupervisorIntegrationTests(testing_common.TestCase):
    def launch(
        self,
        health,
        train,
        expect,
        N=4,
        run_fraction=1,
        rank_fraction=1,
        connections=4,
    ):
        test_name = Path(__file__).parent / "supervisor_integration.py"
        config = {
            "N": N,
            "health": health,
            "train": train,
            "run_fraction": run_fraction,
            "rank_fraction": rank_fraction,
        }
        result = emulate_launch(
            [sys.executable, test_name, "--supervise", repr(config)], connections
        )
        self.assertEqual(result, 0)
        return result

    @unittest.skip("Test is flaky, times-out on CI")
    def test_success(self):
        self.launch(health=[[4, 3, 2, 1]], train=["........"], expect="....")

    @unittest.skip("Test is flaky, times-out on CI")
    def test_fail(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1]],
            train=["....F...", "........"],
            expect="....",
        )

    @unittest.skip("Test is flaky, times-out on CI")
    def test_hang(self):
        self.launch(
            health=[[4, 3, "hang", 1], [4, 3, 2, 1]],
            train=["......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            expect="....",
        )

    @unittest.skip("Test is flaky, times-out on CI")
    def test_error_fail(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]],
            train=["...E..", "....F.", "......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            expect="....",
        )

    @unittest.skip("Test is flaky, times-out on CI")
    def test_error_error(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]],
            train=["...E..", ".E....", "......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            connections=5,
            expect=".....",
        )


if __name__ == "__main__":
    testing_common.run_tests()
