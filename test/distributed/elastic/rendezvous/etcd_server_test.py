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
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer


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
    """Tests for find_free_port() error handling."""

    def test_find_free_port_returns_listening_socket(self):
        from torch.distributed.elastic.rendezvous.etcd_server import find_free_port

        s = find_free_port()
        try:
            self.assertIsNotNone(s)
            self.assertGreater(s.getsockname()[1], 0)
        finally:
            s.close()

    @unittest.mock.patch("socket.socket")
    def test_find_free_port_no_unbound_local_error_when_socket_fails(
        self, mock_socket_cls
    ):
        """socket.socket() raising OSError must not cause UnboundLocalError."""
        from torch.distributed.elastic.rendezvous.etcd_server import find_free_port

        mock_socket_cls.side_effect = OSError("Address family not supported")

        with self.assertRaises(RuntimeError):
            find_free_port()
