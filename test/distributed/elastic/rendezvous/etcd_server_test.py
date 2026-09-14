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
    def test_find_free_port_returns_socket(self):
        """find_free_port should succeed under normal conditions."""
        s = find_free_port()
        try:
            self.assertGreater(s.getsockname()[1], 0)
        finally:
            s.close()

    def test_find_free_port_socket_constructor_failure_no_unbound_local_error(self):
        """If socket() raises on the first address, no UnboundLocalError should escape."""
        addr_ipv4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        # Provide two addresses: the first socket() call fails, the second succeeds
        real_socket = socket.socket

        call_count = 0

        def socket_side_effect(family, type, proto):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("simulated socket creation failure")
            return real_socket(family, type, proto)

        with mock.patch("socket.getaddrinfo", return_value=[addr_ipv4, addr_ipv4]), \
             mock.patch("socket.socket", side_effect=socket_side_effect):
            s = find_free_port()
            s.close()

    def test_find_free_port_all_addresses_fail_raises_runtime_error(self):
        """When all addresses fail, RuntimeError (not UnboundLocalError) should be raised."""
        addr_ipv4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))

        with mock.patch("socket.getaddrinfo", return_value=[addr_ipv4]), \
             mock.patch("socket.socket", side_effect=OSError("socket failed")):
            with self.assertRaises(RuntimeError):
                find_free_port()
