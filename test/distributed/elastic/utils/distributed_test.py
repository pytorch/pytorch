#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import multiprocessing as mp
import os
import socket
import sys
import unittest
from contextlib import closing

from torch.distributed import DistNetworkError, DistStoreError
from torch.distributed.elastic import utils as elastic_utils
from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_TSAN,
    TestCase,
)


def _create_c10d_store_mp(is_server, server_addr, port, world_size, wait_for_workers):
    store = create_c10d_store(
        is_server,
        server_addr,
        port,
        world_size,
        wait_for_workers=wait_for_workers,
        timeout=2,
    )
    if store is None:
        raise AssertionError

    store.set(f"test_key/{os.getpid()}", b"test_value")


def _can_bind(family, host):
    try:
        with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
            s.bind((host, 0))
        return True
    except OSError:
        return False


if IS_WINDOWS or IS_MACOS:
    print("tests incompatible with tsan or asan", file=sys.stderr)
    sys.exit(0)


class DistributedUtilTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_create_store_single_server(self):
        store = create_c10d_store(is_server=True, server_addr=socket.gethostname())
        self.assertIsNotNone(store)

    def test_create_store_no_port_multi(self):
        with self.assertRaises(ValueError):
            create_c10d_store(
                is_server=True, server_addr=socket.gethostname(), world_size=2
            )

    @unittest.skipIf(TEST_WITH_TSAN, "test incompatible with tsan")
    def test_create_store_multi(self):
        world_size = 3
        wait_for_workers = False
        localhost = socket.gethostname()

        # start the server on the main process using an available port
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )

        # worker processes will use the port that was assigned to the server
        server_port = store.port

        worker0 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )
        worker1 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )

        worker0.start()
        worker1.start()

        worker0.join()
        worker1.join()

        # check test_key/pid == "test_value"
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker0.pid}").decode("UTF-8")
        )
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker1.pid}").decode("UTF-8")
        )

        self.assertEqual(0, worker0.exitcode)
        self.assertEqual(0, worker1.exitcode)

    def test_create_store_timeout_on_server(self):
        with self.assertRaises(DistStoreError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=True,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_timeout_on_worker(self):
        with self.assertRaises(DistNetworkError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=False,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_with_libuv_support(self):
        world_size = 1
        wait_for_workers = False
        localhost = socket.gethostname()

        os.environ["USE_LIBUV"] = "0"
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertFalse(store.libuvBackend)
        del os.environ["USE_LIBUV"]
        if "USE_LIBUV" in os.environ:
            raise AssertionError("Expected USE_LIBUV to be removed from os.environ")

        # libuv backend is enabled by default
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertTrue(store.libuvBackend)

    def test_port_already_in_use_on_server(self):
        # try to create the TCPStore server twice on the same port
        # the second should fail due to a port conflict
        # first store binds onto a free port
        # try creating the second store on the port that the first store binded to
        server_addr = socket.gethostname()
        pick_free_port = 0
        store1 = create_c10d_store(
            is_server=True,
            server_addr=server_addr,
            server_port=pick_free_port,
            timeout=1,
        )
        with self.assertRaises(DistNetworkError):
            create_c10d_store(
                is_server=True, server_addr=server_addr, server_port=store1.port
            )

    def test_get_socket_with_port_reserves_all_addresses(self):
        # The port is handed to a TCPStore server, which binds the wildcard
        # address, so it must not be in use on any local address. ::1 checks
        # the dual-stack reservation rather than the IPv4-only fallback.
        addrs = [
            (family, host)
            for family, host in (
                (socket.AF_INET, "127.0.0.1"),
                (socket.AF_INET, "127.0.0.2"),
                (socket.AF_INET6, "::1"),
            )
            if _can_bind(family, host)
        ]
        for get_sock in (get_socket_with_port, elastic_utils.get_socket_with_port):
            with closing(get_sock()) as sock:
                port = sock.getsockname()[1]
                for family, host in addrs:
                    with self.subTest(fn=get_sock.__module__, host=host):
                        with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
                            with self.assertRaises(OSError) as cm:
                                s.bind((host, port))
                            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)

    def test_port_already_in_use_on_worker(self):
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            # on the worker port conflict shouldn't matter, it should just timeout
            # since we never created a server
            with self.assertRaises(DistNetworkError):
                create_c10d_store(
                    is_server=False,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )


if __name__ == "__main__":
    run_tests()
