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
    # A single loopback address so the number of attempts is deterministic.
    _ONE_ADDR = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
    # Two identical loopback addresses so the loop has a second address to try.
    _TWO_ADDRS = _ONE_ADDR * 2

    def test_retries_next_address_when_socket_creation_fails(self) -> None:
        # If creating the socket for the first address fails, find_free_port
        # must move on to the next address instead of raising UnboundLocalError
        # while trying to clean up a socket that was never created.
        good = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with (
                patch("socket.getaddrinfo", return_value=self._TWO_ADDRS),
                patch("socket.socket", side_effect=[OSError("first fails"), good]),
            ):
                result = find_free_port()
            self.assertIs(result, good)
        finally:
            good.close()

    def test_all_attempts_failing_raise_runtime_error(self) -> None:
        # When every address fails to build a socket, the documented
        # RuntimeError must surface -- not an UnboundLocalError from cleanup.
        with (
            patch("socket.getaddrinfo", return_value=self._ONE_ADDR),
            patch("socket.socket", side_effect=OSError("socket failed")),
        ):
            with self.assertRaises(RuntimeError):
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
