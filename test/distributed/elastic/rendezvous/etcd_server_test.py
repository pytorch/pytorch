# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import unittest
from unittest.mock import patch

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

    def test_find_free_port_socket_creation_failure(self):
        """
        Test that find_free_port handles socket creation failures gracefully.
        When socket.socket() raises OSError, it should not raise UnboundLocalError
        and should continue trying remaining addresses.
        """
        import socket as socket_module

        # Test 1: First socket creation fails, second succeeds
        call_count = 0
        real_socket = socket_module.socket

        def mock_socket_factory(family, type, proto):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("socket failed on first attempt")
            return real_socket(family, type, proto)

        with patch("socket.getaddrinfo", return_value=[
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            (socket_module.AF_INET6, socket_module.SOCK_STREAM, 0, "", ("::1", 0)),
        ]), patch("socket.socket", side_effect=mock_socket_factory):
            sock = find_free_port()
            self.assertIsNotNone(sock)
            sock.close()

        # Test 2: All socket attempts fail - should raise RuntimeError, not UnboundLocalError
        with patch("socket.getaddrinfo", return_value=[
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        ]), patch("socket.socket", side_effect=OSError("socket failed")):
            with self.assertRaises(RuntimeError) as cm:
                find_free_port()
            self.assertIn("Failed to create a socket", str(cm.exception))

    def test_find_free_port_normal_case(self):
        """Test that find_free_port works normally when sockets can be created."""
        sock = find_free_port()
        self.assertIsNotNone(sock)
        port = sock.getsockname()[1]
        self.assertGreater(port, 0)
        sock.close()
