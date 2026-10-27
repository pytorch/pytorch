# Owner(s): ["oncall: distributed"]

# Regression tests for https://github.com/pytorch/pytorch/issues/104620
#
# ``register_backend`` recreates ``backend_registry.BackendType``.
# ``torch/distributed/rpc/__init__.py`` re-exports that enum by value
# (``from .backend_registry import BackendType``), so the public
# ``torch.distributed.rpc.BackendType`` must be refreshed when a backend is
# registered. Otherwise it keeps pointing at the stale pre-registration enum and
# ``init_rpc`` rejects ``rpc.BackendType.TENSORPIPE`` once a custom backend has
# been registered.

import os
import tempfile
import unittest
from unittest import mock

import torch.distributed.rpc as rpc
from torch.testing._internal.common_utils import (
    find_free_port,
    HardwareClassification,
    run_tests,
    TestCase,
)


_RPC_AVAILABLE = rpc.is_available()
# ``TensorPipeRpcBackendOptions`` exists even on no-TensorPipe builds (options.py
# falls back to ``object``), so capability must be read from the native flag.
_TENSORPIPE_AVAILABLE = _RPC_AVAILABLE and getattr(
    rpc, "_is_tensorpipe_available", False
)

if _RPC_AVAILABLE:
    from torch.distributed.rpc import backend_registry


def _noop_options_handler(*args, **kwargs):
    return None


def _noop_init_backend_handler(*args, **kwargs):
    return None


def _echo(value):
    return value


class _RegistryTestBase(TestCase):
    _BACKEND = "TEST_BACKEND_104620"

    def setUp(self):
        super().setUp()
        self._registry_backend_type = backend_registry.BackendType
        self._public_backend_type = rpc.BackendType
        self._tmp_files: list[str] = []

    def tearDown(self):
        if rpc._is_current_rpc_agent_set():
            rpc.shutdown()
        # Restore both the registry global and the public re-export so a
        # registration never leaks into other tests regardless of ordering.
        backend_registry.BackendType = self._registry_backend_type
        rpc.BackendType = self._public_backend_type
        for path in self._tmp_files:
            if os.path.exists(path):
                os.unlink(path)
        super().tearDown()

    def _register(self):
        return backend_registry.register_backend(
            self._BACKEND, _noop_options_handler, _noop_init_backend_handler
        )

    def _file_init_method(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        os.unlink(path)
        self._tmp_files.append(path)
        return f"file://{path}"


@unittest.skipIf(not _RPC_AVAILABLE, "RPC is not available on this platform")
class TestBackendRegistryAlignment(_RegistryTestBase):
    hw_classification = HardwareClassification.GENERIC

    def test_public_backend_type_realigned_after_register(self):
        returned = self._register()
        self.assertIs(rpc.BackendType, backend_registry.BackendType)
        self.assertIn(self._BACKEND, rpc.BackendType.__members__)
        self.assertIs(rpc.BackendType[self._BACKEND], returned)
        self.assertIn("TENSORPIPE", rpc.BackendType.__members__)

    def test_repeated_registration_rejected_and_still_aligned(self):
        self._register()
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            self._register()
        self.assertIs(rpc.BackendType, backend_registry.BackendType)
        self.assertIn(self._BACKEND, rpc.BackendType.__members__)


@unittest.skipIf(not _TENSORPIPE_AVAILABLE, "TensorPipe RPC backend is not available")
class TestBackendRegistryTensorPipeRoundtrip(_RegistryTestBase):
    hw_classification = HardwareClassification.GENERIC

    def test_explicit_public_backend_roundtrip_after_register(self):
        self._register()
        options = rpc.TensorPipeRpcBackendOptions(
            init_method=self._file_init_method(), rpc_timeout=60
        )
        rpc.init_rpc(
            "worker0",
            backend=rpc.BackendType.TENSORPIPE,
            rank=0,
            world_size=1,
            rpc_backend_options=options,
        )
        try:
            self.assertEqual(rpc.rpc_sync("worker0", _echo, args=(42,)), 42)
        finally:
            rpc.shutdown()

    def test_inferred_backend_roundtrip_after_register(self):
        self._register()
        options = rpc.TensorPipeRpcBackendOptions(
            init_method=self._file_init_method(), rpc_timeout=60
        )
        rpc.init_rpc(
            "worker0",
            rank=0,
            world_size=1,
            rpc_backend_options=options,  # backend omitted -> inferred from options
        )
        try:
            self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)
            self.assertEqual(rpc.rpc_sync("worker0", _echo, args=(43,)), 43)
        finally:
            rpc.shutdown()

    def test_default_backend_and_options_roundtrip_after_register(self):
        self._register()
        with mock.patch.dict(
            os.environ,
            {"MASTER_ADDR": "localhost", "MASTER_PORT": str(find_free_port())},
        ):
            # backend and options both defaulted -> default BackendType.TENSORPIPE
            rpc.init_rpc("worker0", rank=0, world_size=1)
        try:
            self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)
            self.assertEqual(rpc.rpc_sync("worker0", _echo, args=(44,)), 44)
        finally:
            rpc.shutdown()


if __name__ == "__main__":
    run_tests()
