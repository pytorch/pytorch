# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import unittest

import etcd

from torch.distributed.elastic.rendezvous.etcd_rendezvous import (
    EtcdRendezvous,
    EtcdRendezvousHandler,
)
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
            client = etcd.Client(server.get_host(), server.get_port())

            rdzv = EtcdRendezvous(
                client=client,
                prefix="test",
                run_id=1,
                num_min_workers=1,
                num_max_workers=1,
                timeout=60,
                last_call_timeout=30,
            )
            rdzv_handler = EtcdRendezvousHandler(rdzv)
            rdzv_info = rdzv_handler.next_rendezvous()
            self.assertIsNotNone(rdzv_info.store)
            self.assertEqual(0, rdzv_info.rank)
            self.assertEqual(1, rdzv_info.world_size)
        finally:
            server.stop()
