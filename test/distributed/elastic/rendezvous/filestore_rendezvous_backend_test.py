# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from torch.distributed import FileStore
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.filestore_rendezvous_backend import (
    # FileStoreRendezvousBackend,
    create_backend,
)

# Here we will also put the RendezvousBackendTestMixin functionality, once
# the filestore has been implemented.

class CreateBackendTests(TestCase):
    def setUp(self) -> None:
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint="localhost:29300",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
        )

        self._expected_endpoint_host = "localhost"
        self._expected_endpoint_port = 29300
        self._expected_store_type = FileStore

    def test_dummy_test(self):
        with self.assertRaises(NotImplementedError):
            create_backend(self._params)
