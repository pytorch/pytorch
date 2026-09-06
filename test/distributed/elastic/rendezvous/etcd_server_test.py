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


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class EtcdServerTest(unittest.TestCase):
    @mock.patch(
        "torch.distributed.elastic.rendezvous.etcd_server.socket.getaddrinfo"
    )
    @mock.patch("torch.distributed.elastic.rendezvous.etcd_server.socket.socket")
    def test_find_free_port_skips_socket_creation_error(
        self, mock_socket, mock_getaddrinfo
    ):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("localhost", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("localhost", 0)),
        ]
        expected_socket = mock.Mock()
        mock_socket.side_effect = [OSError("socket failed"), expected_socket]

        self.assertIs(find_free_port(), expected_socket)
        self.assertEqual(2, mock_socket.call_count)
        expected_socket.bind.assert_called_once_with(("localhost", 0))
        expected_socket.listen.assert_called_once_with(0)

    @mock.patch(
        "torch.distributed.elastic.rendezvous.etcd_server.socket.getaddrinfo"
    )
    @mock.patch("torch.distributed.elastic.rendezvous.etcd_server.socket.socket")
    def test_find_free_port_raises_runtime_error_when_all_attempts_fail(
        self, mock_socket, mock_getaddrinfo
    ):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("localhost", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("localhost", 0)),
        ]
        mock_socket.side_effect = OSError("socket failed")

        with self.assertRaisesRegex(RuntimeError, "Failed to create a socket"):
            find_free_port()

        self.assertEqual(2, mock_socket.call_count)

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
