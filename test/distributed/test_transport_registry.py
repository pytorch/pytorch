# Owner(s): ["oncall: distributed"]

from typing import Any, cast
from unittest.mock import patch

import torch
from torch.distributed._transport import (
    _registry,
    available_transports,
    new_transport,
    register_transport,
    Transport,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _TestTransport(Transport):
    is_supported = True

    def __init__(self, device, *, value=None):
        super().__init__(device)
        self.value = value
        self.closed = False

    @staticmethod
    def supported() -> bool:
        return _TestTransport.is_supported

    def bind(self) -> bytes:
        return b"test://"

    def connect(self, peer_url: bytes) -> int:
        return 0

    def connected(self) -> bool:
        return True

    def register_memory(self, tensor):
        return tensor

    def write(self, local_buffer, remote_buffer) -> int:
        return 0

    def read(self, local_buffer, remote_buffer) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _EntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        return self.value


class TestTransportRegistry(TestCase):
    def setUp(self):
        self.registered = patch.dict(_registry._registered_transports, clear=True)
        self.registered.start()

    def tearDown(self):
        self.registered.stop()

    def test_register_and_create(self):
        register_transport("test", _TestTransport)
        transport = new_transport("TEST", "cpu", value=3)
        self.assertIsInstance(transport, _TestTransport)
        self.assertEqual(transport.device, torch.device("cpu"))
        self.assertEqual(transport.value, 3)

    def test_duplicate_registration(self):
        register_transport("test", _TestTransport)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transport("test", _TestTransport)

    def test_factory_must_be_callable(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            register_transport("test", cast(Any, None))

    def test_entry_point(self):
        entry_point = _EntryPoint("external", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            transport = new_transport("external", "cpu")
        self.assertIsInstance(transport, _TestTransport)

    def test_entry_point_must_return_transport(self):
        entry_point = _EntryPoint("broken", lambda **kwargs: object())
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            with self.assertRaisesRegex(TypeError, "expected a Transport"):
                new_transport("broken", "cpu")

    def test_duplicate_entry_points(self):
        entry_points = [
            _EntryPoint("duplicate", _TestTransport),
            _EntryPoint("duplicate", _TestTransport),
        ]
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter(entry_points)
        ):
            with self.assertRaisesRegex(RuntimeError, "multiple entry points"):
                new_transport("duplicate", "cpu")

    def test_unknown_transport(self):
        with patch.object(_registry, "_iter_entry_points", return_value=iter(())):
            with self.assertRaisesRegex(ValueError, "unknown transport"):
                new_transport("missing", "cpu")

    def test_unsupported_transport_is_closed(self):
        register_transport("test", _TestTransport)
        _TestTransport.is_supported = False
        try:
            with self.assertRaisesRegex(RuntimeError, "is not supported"):
                new_transport("test", "cpu")
        finally:
            _TestTransport.is_supported = True

    def test_available_transports(self):
        entry_point = _EntryPoint("external", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            self.assertEqual(available_transports(), ("external",))


if __name__ == "__main__":
    run_tests()
