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


class FindFreePortTest(unittest.TestCase):
    ADDRS = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
    ]

    def test_retries_next_address_when_socket_creation_fails(self):
        # socket() raising on the first address must not leak an
        # UnboundLocalError from the cleanup path; the next address is tried.
        good_sock = mock.MagicMock()
        with (
            mock.patch("socket.getaddrinfo", return_value=self.ADDRS),
            mock.patch(
                "socket.socket", side_effect=[OSError("socket failed"), good_sock]
            ) as mock_socket,
        ):
            result = find_free_port()

        self.assertIs(result, good_sock)
        self.assertEqual(mock_socket.call_count, 2)
        good_sock.bind.assert_called_once_with(("localhost", 0))

    def test_closes_socket_when_bind_fails(self):
        # A successfully created socket whose bind() fails must be closed.
        bad_sock = mock.MagicMock()
        bad_sock.bind.side_effect = OSError("bind failed")
        with (
            mock.patch("socket.getaddrinfo", return_value=self.ADDRS[:1]),
            mock.patch("socket.socket", return_value=bad_sock),
        ):
            with self.assertRaises(RuntimeError):
                find_free_port()

        bad_sock.close.assert_called_once()

    def test_raises_runtime_error_when_all_attempts_fail(self):
        # Every address failing at socket() creation ends in the documented
        # RuntimeError rather than an UnboundLocalError.
        with (
            mock.patch("socket.getaddrinfo", return_value=self.ADDRS),
            mock.patch("socket.socket", side_effect=OSError("socket failed")),
        ):
            with self.assertRaises(RuntimeError):
                find_free_port()
