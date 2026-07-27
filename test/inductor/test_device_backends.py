# Owner(s): ["module: inductor"]
from unittest import mock

from torch._inductor.codegen import common
from torch.testing._internal.common_utils import run_tests, TestCase


class DeviceBackendLoaderTest(TestCase):
    def setUp(self) -> None:
        # Snapshot the module-level registries so every test starts clean, then
        # restore exactly in tearDown (do not wipe device_codegens wholesale --
        # init_backend_registration() is cached and would not re-populate the
        # built-ins).
        self._orig_loaders = dict(common._device_backend_loaders)
        self._orig_loaded = set(common._loaded_device_backends)
        self._orig_codegen = dict(common.device_codegens)
        common._device_backend_loaders.clear()
        common._loaded_device_backends.clear()
        common._discover_device_backend_entrypoints.cache_clear()

    def tearDown(self) -> None:
        for device in [
            d for d in common.device_codegens if d not in self._orig_codegen
        ]:
            del common.device_codegens[device]
        common._device_backend_loaders.clear()
        common._device_backend_loaders.update(self._orig_loaders)
        common._loaded_device_backends.clear()
        common._loaded_device_backends.update(self._orig_loaded)
        common._discover_device_backend_entrypoints.cache_clear()

    def test_loader_called_once_on_first_resolve(self) -> None:
        device = "fake_loader_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            # A real vendor calls register_backend_for_device(...) here.
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, loader)
        self.assertIsNotNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)
        # Subsequent resolves must not re-invoke the loader.
        common.get_scheduling_for_device(device)
        self.assertEqual(calls["n"], 1)

    def test_loader_fires_on_wrapper_codegen_path(self) -> None:
        device = "fake_wrapper_device"

        common.register_device_backend_loader(
            device, lambda: common.register_backend_for_device(device, object, object)
        )
        self.assertIsNotNone(common.get_wrapper_codegen_for_device(device))

    def test_unknown_device_still_returns_none(self) -> None:
        self.assertIsNone(
            common.get_scheduling_for_device("totally_unknown_device_xyz")
        )

    def test_entrypoint_discovery_picks_up_vendor(self) -> None:
        device = "fake_ep_device"
        ep = mock.MagicMock()
        ep.name = device
        ep.load.return_value = lambda: common.register_backend_for_device(
            device, object, object
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        ep.load.assert_called_once()

    def test_entrypoint_import_is_deferred_until_use(self) -> None:
        device = "fake_deferred_device"
        ep = mock.MagicMock()
        ep.name = device
        ep.load.return_value = lambda: common.register_backend_for_device(
            device, object, object
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            common._discover_device_backend_entrypoints()
            # Discovery must build the loader without importing the vendor module.
            ep.load.assert_not_called()
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        ep.load.assert_called_once()

    def test_imperative_loader_does_not_suppress_entrypoints(self) -> None:
        # Regression guard: an imperative loader must not cause entry-point
        # discovery to be skipped for a different device.
        common.register_device_backend_loader("imperative_device", lambda: None)
        ep_device = "ep_only_device"
        ep = mock.MagicMock()
        ep.name = ep_device
        ep.load.return_value = lambda: common.register_backend_for_device(
            ep_device, object, object
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(ep_device))

    def test_loader_failure_is_raised_and_retryable(self) -> None:
        device = "fake_failing_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        common.register_device_backend_loader(device, loader)
        # The real failure propagates (instead of a generic "device not supported").
        with self.assertRaisesRegex(RuntimeError, "boom"):
            common.get_scheduling_for_device(device)
        # The claim was released, so a later resolve retries the loader.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            common.get_scheduling_for_device(device)
        self.assertEqual(calls["n"], 2)

    def test_loader_succeeds_without_registering_is_not_retried(self) -> None:
        # A loader that returns OK but forgets to register must not busy-loop:
        # the claim is retained, so later resolves skip it.
        device = "fake_noop_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            # Intentionally does NOT call register_backend_for_device.

        common.register_device_backend_loader(device, loader)
        self.assertIsNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)
        # Second resolve must not re-invoke the no-op loader.
        self.assertIsNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)

    def test_imperative_loader_wins_over_entrypoint_for_same_device(self) -> None:
        # setdefault: a loader registered imperatively takes precedence over a
        # same-named entry point.
        device = "contested_device"
        imperative_calls = {"n": 0}

        def imperative_loader() -> None:
            imperative_calls["n"] += 1
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, imperative_loader)
        ep = mock.MagicMock()
        ep.name = device
        ep.load.return_value = lambda: common.register_backend_for_device(
            device, object, object
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        self.assertEqual(imperative_calls["n"], 1)
        ep.load.assert_not_called()


if __name__ == "__main__":
    run_tests()
