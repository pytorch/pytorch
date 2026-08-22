#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import socket
import sys
import unittest
from contextlib import closing
from unittest import mock

from torch.distributed import DistNetworkError, DistStoreError
from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    MI200_ARCH,
    run_tests,
    skipIfRocmArch,
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


if IS_WINDOWS or IS_MACOS:
    print("tests incompatible with tsan or asan", file=sys.stderr)
    sys.exit(0)


class DistributedUtilTest(TestCase):
    @mock.patch("torch.distributed.elastic.utils.distributed.socket.getaddrinfo")
    @mock.patch("torch.distributed.elastic.utils.distributed.socket.socket")
    def test_get_socket_with_port_continues_after_socket_creation_failure(
        self, socket_mock, getaddrinfo_mock
    ):
        first_addr = (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0))
        second_addr = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
        getaddrinfo_mock.return_value = [first_addr, second_addr]
        usable_socket = mock.Mock()
        socket_mock.side_effect = [OSError("unsupported address family"), usable_socket]

        self.assertIs(get_socket_with_port(), usable_socket)
        usable_socket.bind.assert_called_once_with(("localhost", 0))
        usable_socket.listen.assert_called_once_with(0)

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

    @skipIfRocmArch(MI200_ARCH)
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
