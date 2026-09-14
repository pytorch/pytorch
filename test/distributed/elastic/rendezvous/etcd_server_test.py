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

    def test_find_free_port_handles_socket_creation_failure(self):
        """
        Test that find_free_port does not raise UnboundLocalError
        when socket creation fails on the first attempt.
        """
        call_count = [0]
        original_socket = __import__('socket').socket
        def failing_socket(family, type, proto=__import__('socket').SOCK_STREAM):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Simulated socket creation failure")
            return original_socket(family, type, proto)

        with patch('socket.socket', side_effect=failing_socket):
            # This should NOT raise UnboundLocalError
            # It should either succeed on a later address or raise RuntimeError
            try:
                sock = find_free_port()
                # If we got here, socket was created successfully
                try:
                    port = sock.getsockname()[1]
                    self.assertGreater(port, 0)
                finally:
                    sock.close()
            except UnboundLocalError:
                self.fail("find_free_port raised UnboundLocalError - bug not fixed!")
            except OSError:
                # Expected if all socket creations fail
                pass
            except RuntimeError:
                # Also expected if all socket creations fail
                pass
