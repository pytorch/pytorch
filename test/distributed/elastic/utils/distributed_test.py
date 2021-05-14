#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import socket
import unittest
from contextlib import closing

from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_free_port,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, IS_MACOS


def _create_c10d_store_mp(is_server, server_addr, port, world_size):
    store = create_c10d_store(is_server, server_addr, port, world_size, timeout=2)
    if store is None:
        raise AssertionError()

    store.set(f"test_key/{os.getpid()}", "test_value".encode("UTF-8"))


@unittest.skipIf(IS_WINDOWS or IS_MACOS, "tests incompatible with tsan or asan")
class DistributedUtilTest(unittest.TestCase):
    def test_create_store_single_server(self):
        store = create_c10d_store(is_server=True, server_addr=socket.gethostname())
        self.assertIsNotNone(store)

    def test_create_store_no_port_multi(self):
        with self.assertRaises(ValueError):
            create_c10d_store(
                is_server=True, server_addr=socket.gethostname(), world_size=2
            )

    def test_create_store_multi(self):
        world_size = 3
        server_port = get_free_port()
        localhost = socket.gethostname()
        worker0 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size),
        )
        worker1 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size),
        )

        worker0.start()
        worker1.start()

        # start the server on the main process
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=server_port,
            world_size=world_size,
            timeout=2,
        )

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
        with self.assertRaises(TimeoutError):
            port = get_free_port()
            create_c10d_store(
                is_server=True,
                server_addr=socket.gethostname(),
                server_port=port,
                world_size=2,
                timeout=1,
            )

    def test_create_store_timeout_on_worker(self):
        with self.assertRaises(TimeoutError):
            port = get_free_port()
            create_c10d_store(
                is_server=False,
                server_addr=socket.gethostname(),
                server_port=port,
                world_size=2,
                timeout=1,
            )

    def test_port_already_in_use_on_server(self):
        sock = get_socket_with_port()
        with closing(sock):
            # try to create a store on the same port without releasing the socket
            # should raise a IOError
            port = sock.getsockname()[1]
            with self.assertRaises(IOError):
                create_c10d_store(
                    is_server=True,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )

    def test_port_already_in_use_on_worker(self):
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            # on the worker port conflict shouldn't matter, it should just timeout
            # since we never created a server
            with self.assertRaises(IOError):
                create_c10d_store(
                    is_server=False,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )


if __name__ == "__main__":
    run_tests()
