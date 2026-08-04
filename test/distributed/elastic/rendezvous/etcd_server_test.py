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
    @staticmethod
    def _addr():
        return (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))

    @mock.patch(
        "torch.distributed.elastic.rendezvous.etcd_server.socket.getaddrinfo"
    )
    @mock.patch("torch.distributed.elastic.rendezvous.etcd_server.socket.socket")
    def test_find_free_port_falls_back_to_next_addr(
        self, socket_mock, getaddrinfo_mock
    ) -> None:
        # Creating a socket for the first address fails; find_free_port must
        # move on to the next address instead of raising UnboundLocalError from
        # closing a socket that was never created.
        getaddrinfo_mock.return_value = [self._addr(), self._addr()]
        good_socket = mock.MagicMock()
        socket_mock.side_effect = [OSError("first attempt failed"), good_socket]

        self.assertIs(find_free_port(), good_socket)
        good_socket.bind.assert_called_once()
        good_socket.listen.assert_called_once()

    @mock.patch(
        "torch.distributed.elastic.rendezvous.etcd_server.socket.getaddrinfo"
    )
    @mock.patch("torch.distributed.elastic.rendezvous.etcd_server.socket.socket")
    def test_find_free_port_raises_with_cause_when_all_fail(
        self, socket_mock, getaddrinfo_mock
    ) -> None:
        getaddrinfo_mock.return_value = [self._addr()]
        socket_mock.side_effect = OSError("no socket for you")

        with self.assertRaises(RuntimeError) as cm:
            find_free_port()
        self.assertIsInstance(cm.exception.__cause__, OSError)
