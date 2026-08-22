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
from unittest.mock import patch

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer, find_free_port


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class FindFreePortTest(unittest.TestCase):
    # Two IPv4 entries so the fallthrough case does not depend on IPv6
    # being available on the test host.
    _ADDRS = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
    ]

    def test_socket_constructor_failure_tries_next_addr(self):
        real_socket = socket.socket
        calls = []

        def flaky_socket(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise OSError("socket failed")
            return real_socket(*args, **kwargs)

        with (
            patch("socket.getaddrinfo", return_value=self._ADDRS),
            patch("socket.socket", side_effect=flaky_socket),
        ):
            sock = find_free_port()
        try:
            self.assertEqual(2, len(calls))
            self.assertGreater(sock.getsockname()[1], 0)
        finally:
            sock.close()

    def test_all_socket_constructors_fail_raises_runtime_error(self):
        with (
            patch("socket.getaddrinfo", return_value=self._ADDRS),
            patch("socket.socket", side_effect=OSError("socket failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to create a socket"):
                find_free_port()


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
