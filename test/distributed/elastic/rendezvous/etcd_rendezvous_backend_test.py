# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from base64 import b64encode
from typing import cast, ClassVar
from unittest import TestCase

from etcd import EtcdKeyNotFound  # type: ignore[import]

from rendezvous_backend_test import RendezvousBackendTestMixin

from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import (
    create_backend,
    EtcdRendezvousBackend,
)
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore


class EtcdRendezvousBackendTest(TestCase, RendezvousBackendTestMixin):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.stop()

    def setUp(self) -> None:
        self._client = self._server.get_client()

        # Make sure we have a clean slate.
        try:
            self._client.delete("/dummy_prefix", recursive=True, dir=True)
        except EtcdKeyNotFound:
            pass

        self._backend = EtcdRendezvousBackend(
            self._client, "dummy_run_id", "/dummy_prefix"
        )

    def _corrupt_state(self) -> None:
        self._client.write("/dummy_prefix/dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.stop()

    def setUp(self) -> None:
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint=self._server.get_endpoint(),
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            protocol="hTTp",
            read_timeout="10",
        )

        self._expected_read_timeout = 10

    def test_create_backend_returns_backend(self) -> None:
        backend, store = create_backend(self._params)

        self.assertEqual(backend.name, "etcd-v2")

        self.assertIsInstance(store, EtcdStore)

        etcd_store = cast(EtcdStore, store)

        self.assertEqual(etcd_store.client.read_timeout, self._expected_read_timeout)  # type: ignore[attr-defined]

        client = self._server.get_client()

        backend.set_state(b"dummy_state")

        result = client.get("/torch/elastic/rendezvous/" + self._params.run_id)

        self.assertEqual(result.value, b64encode(b"dummy_state").decode())
        self.assertLessEqual(result.ttl, 7200)

        store.set("dummy_key", "dummy_value")

        result = client.get("/torch/elastic/store/" + b64encode(b"dummy_key").decode())

        self.assertEqual(result.value, b64encode(b"dummy_value").decode())

    def test_create_backend_returns_backend_if_protocol_is_not_specified(self) -> None:
        del self._params.config["protocol"]

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(
        self,
    ) -> None:
        del self._params.config["read_timeout"]

        self._expected_read_timeout = 60

        self.test_create_backend_returns_backend()

    def test_create_backend_raises_error_if_etcd_is_unreachable(self) -> None:
        self._params.endpoint = "dummy:1234"

        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to etcd has failed. See inner exception for details.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_protocol_is_invalid(self) -> None:
        self._params.config["protocol"] = "dummy"

        with self.assertRaisesRegex(
            ValueError, r"^The protocol must be HTTP or HTTPS.$"
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                self._params.config["read_timeout"] = read_timeout

                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    create_backend(self._params)
