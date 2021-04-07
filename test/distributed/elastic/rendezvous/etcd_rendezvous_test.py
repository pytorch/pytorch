# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
import uuid

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer


@unittest.skipIf(os.getenv("CIRCLECI"), "T85992919 temporarily disabling in circle ci")
class EtcdRendezvousTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def test_etcd_rdzv_basic_params(self):
        """
        Check that we can create the handler with a minimum set of
        params
        """
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=f"{uuid.uuid4()}",
            min_nodes=1,
            max_nodes=1,
        )
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)

    def test_etcd_rdzv_additional_params(self):
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )

        etcd_rdzv = create_rdzv_handler(rdzv_params)

        self.assertIsNotNone(etcd_rdzv)
        self.assertEqual(run_id, etcd_rdzv.get_run_id())

    def test_get_backend(self):
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )

        etcd_rdzv = create_rdzv_handler(rdzv_params)

        self.assertEqual("etcd", etcd_rdzv.get_backend())
