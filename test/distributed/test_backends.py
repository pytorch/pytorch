# Owner(s): ["oncall: distributed"]

import os
import warnings

import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    BackendConfig,
    ProcessGroup,
    _get_backend_type,
    _resolve_default_backend,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


"""
common backend API tests
"""


class TestMiscCollectiveUtils(TestCase):
    def test_device_to_backend_mapping(self, device) -> None:
        """
        Test device to backend mapping
        """
        if "cuda" in device:
            assert dist.get_default_backend_for_device(device) == "nccl"
        elif "cpu" in device:
            assert dist.get_default_backend_for_device(device) == "gloo"
        elif "hpu" in device:
            assert dist.get_default_backend_for_device(device) == "hccl"
        else:
            with self.assertRaises(ValueError):
                dist.get_default_backend_for_device(device)

    def test_create_pg(self, device) -> None:
        """
        Test create process group
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        backend = dist.get_default_backend_for_device(device)
        dist.init_process_group(
            backend=backend, rank=0, world_size=1, init_method="env://"
        )
        pg = dist.distributed_c10d._get_default_group()
        backend_pg = pg._get_backend_name()
        assert backend_pg == backend
        dist.destroy_process_group()

    def test_get_backend_type_builtin(self) -> None:
        """Ensure built-in backends map to ProcessGroup.BackendType."""
        self.assertEqual(
            _get_backend_type("gloo"),
            ProcessGroup.BackendType.GLOO,
        )
        self.assertEqual(
            _get_backend_type("NCCL"),
            ProcessGroup.BackendType.NCCL,
        )

    def test_get_backend_type_custom_backend(self) -> None:
        """Ensure registered plugins resolve to CUSTOM."""
        existing_plugin = Backend._plugins.get("TEST_CUSTOM")
        try:
            Backend._plugins["TEST_CUSTOM"] = Backend._BackendPlugin(
                lambda *args, **kwargs: None,
                False,
            )
            self.assertEqual(
                _get_backend_type("test_custom"),
                ProcessGroup.BackendType.CUSTOM,
            )
        finally:
            if existing_plugin is None:
                Backend._plugins.pop("TEST_CUSTOM", None)
            else:
                Backend._plugins["TEST_CUSTOM"] = existing_plugin

    def test_get_backend_type_invalid_backend(self) -> None:
        """Unknown backend names should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            _get_backend_type("unknown_backend")

    def test_init_process_group_invalid_backend(self, device) -> None:
        """init_process_group should fail fast on unknown backend strings."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(ValueError, "Unknown backend"):
                dist.init_process_group(
                    backend="typo-backend",
                    rank=0,
                    world_size=1,
                    init_method="env://",
                )

    def test_resolve_default_backend_prefers_nccl(self) -> None:
        """Multi-backend strings should prefer NCCL when present."""
        backend = "cpu:gloo,cuda:nccl"
        backend_config = BackendConfig(backend)
        result = _resolve_default_backend(backend, backend_config)
        self.assertEqual(result, ProcessGroup.BackendType.NCCL)

    def test_resolve_default_backend_prefers_custom(self) -> None:
        """Multi-backend strings should prefer CUSTOM when plugin specified."""
        backend = "cpu:test_custom"
        existing_plugin = Backend._plugins.get("TEST_CUSTOM")
        try:
            Backend._plugins["TEST_CUSTOM"] = Backend._BackendPlugin(
                lambda *args, **kwargs: None,
                False,
            )
            backend_config = BackendConfig(backend)
            result = _resolve_default_backend(backend, backend_config)
            self.assertEqual(result, ProcessGroup.BackendType.CUSTOM)
        finally:
            if existing_plugin is None:
                Backend._plugins.pop("TEST_CUSTOM", None)
            else:
                Backend._plugins["TEST_CUSTOM"] = existing_plugin

    def test_resolve_default_backend_fallback_gloo(self) -> None:
        """Multi-backend strings fall back to GLOO when no special devices."""
        backend = "cpu:gloo"
        backend_config = BackendConfig(backend)
        result = _resolve_default_backend(backend, backend_config)
        self.assertEqual(result, ProcessGroup.BackendType.GLOO)


devices = ["cpu", "cuda", "hpu"]
instantiate_device_type_tests(TestMiscCollectiveUtils, globals(), only_for=devices)

if __name__ == "__main__":
    run_tests()
