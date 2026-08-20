# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import unittest

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import (
    EtcdServer,
    find_free_port,
)


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class EtcdServerTest(unittest.TestCase):
    def test_etcd_server_start_stop(self):
        server = EtcdServer()
        server.start()

        try:
            port = server.get_port()
            host = server.get_host()

            self.assertGreater(port, 0)
            self.assertEqual("localhost", host)
            self.assertEqual(f"{host}:{port}", server.get_endpoint())
            self.assertIsNotNone(server.get_client().version)
        finally:
            server.stop()

    def test_find_free_port_socket_failure_raises_runtime_error(self):
        with unittest.mock.patch(
            "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 0))]
        ), unittest.mock.patch(
            "socket.socket", side_effect=OSError("socket failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to create a socket"):
                find_free_port()

    def test_find_free_port_retries_next_address(self):
        addrs = [
            (2, 1, 6, "", ("127.0.0.1", 0)),
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        second = unittest.mock.Mock()
        with unittest.mock.patch(
            "socket.getaddrinfo", return_value=addrs
        ) as mock_getaddrinfo, unittest.mock.patch("socket.socket", side_effect=[
            OSError("first address failed"),
            second,
        ]) as mock_socket:
            result = find_free_port()
        self.assertEqual(mock_getaddrinfo.call_count, 1)
        self.assertEqual(mock_socket.call_count, 2)
        self.assertIs(result, second)
        second.bind.assert_called_once()
        second.listen.assert_called_once()

    def test_find_free_port_all_addresses_fail_raises_runtime_error(self):
        addrs = [
            (2, 1, 6, "", ("127.0.0.1", 0)),
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        with unittest.mock.patch(
            "socket.getaddrinfo", return_value=addrs
        ), unittest.mock.patch("socket.socket", side_effect=OSError("socket failed")) as mock_socket:
            with self.assertRaisesRegex(RuntimeError, "Failed to create a socket"):
                find_free_port()
        self.assertEqual(mock_socket.call_count, len(addrs))

    def test_etcd_server_with_rendezvous(self):
        server = EtcdServer()
        server.start()

        try:
            endpoint = server.get_endpoint()
            rdzv_params = RendezvousParameters(
                backend="etcd",
                endpoint=endpoint,
                run_id="test_run_1",
                min_nodes=1,
                max_nodes=1,
                timeout=60,
                last_call_timeout=30,
                local_addr="127.0.0.1",
            )
            rdzv_handler = create_rdzv_handler(rdzv_params)
            rdzv_info = rdzv_handler.next_rendezvous()
            self.assertIsNotNone(rdzv_info.store)
            self.assertEqual(0, rdzv_info.rank)
            self.assertEqual(1, rdzv_info.world_size)
        finally:
            server.stop()
