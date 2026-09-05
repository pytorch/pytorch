# Owner(s): ["oncall: r2p"]

import socket
from unittest.mock import patch

from torch.distributed.elastic.rendezvous.etcd_server import find_free_port
from torch.testing._internal.common_utils import run_tests, TestCase


class FindFreePortTest(TestCase):
    def test_find_free_port_skips_failed_socket_construction(self):
        """If socket() fails on the first address, continue with remaining addresses."""
        addrs = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]
        call_count = {"n": 0}
        real_socket = socket.socket

        def socket_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("socket failed")
            return real_socket(*args, **kwargs)

        with (
            patch("socket.getaddrinfo", return_value=addrs),
            patch("socket.socket", side_effect=socket_side_effect),
        ):
            sock = find_free_port()
        try:
            self.assertEqual(2, call_count["n"])
            self.assertIsInstance(sock, socket.socket)
        finally:
            sock.close()

    def test_find_free_port_all_socket_failures_raise_runtime_error(self):
        """Socket construction failures must not escape as UnboundLocalError."""
        addrs = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]
        with (
            patch("socket.getaddrinfo", return_value=addrs),
            patch("socket.socket", side_effect=OSError("socket failed")),
        ):
            with self.assertRaises(RuntimeError) as cm:
                find_free_port()
        self.assertIn("Failed to create a socket", str(cm.exception))


if __name__ == "__main__":
    run_tests()
