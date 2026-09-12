# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import socket
import sys
import unittest
from unittest import mock

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer, find_free_port


_CIRCLECI = bool(os.getenv("CIRCLECI"))
if _CIRCLECI:
    print("T85992919 temporarily disabling etcd server tests in circle ci", file=sys.stderr)


class FindFreePortTest(unittest.TestCase):
    def test_find_free_port_retries_when_socket_constructor_fails(self) -> None:
        # Regression for pytorch/pytorch#191395: socket() failure must not raise
        # UnboundLocalError from s.close() in the except block.
        good_sock = mock.MagicMock()
        addrs = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

        with mock.patch("socket.getaddrinfo", return_value=addrs), mock.patch(
            "socket.socket", side_effect=[OSError("socket failed"), good_sock]
        ) as socket_mock:
            result = find_free_port()

        self.assertIs(result, good_sock)
        self.assertEqual(2, socket_mock.call_count)
        good_sock.bind.assert_called_once_with(("localhost", 0))
        good_sock.listen.assert_called_once_with(0)
        good_sock.close.assert_not_called()

    def test_find_free_port_raises_runtime_error_without_unbound_local(self) -> None:
        addrs = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

        with mock.patch("socket.getaddrinfo", return_value=addrs), mock.patch(
            "socket.socket", side_effect=OSError("socket failed")
        ):
            with self.assertRaisesRegex(RuntimeError, r"^Failed to create a socket$"):
                find_free_port()


@unittest.skipIf(_CIRCLECI, "T85992919 temporarily disabling in circle ci")
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
