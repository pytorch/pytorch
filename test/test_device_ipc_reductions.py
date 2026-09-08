# Owner(s): ["module: multiprocessing"]
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _rebuild_device_tensor_kwargs():
    return {
        "tensor_cls": torch.Tensor,
        "tensor_size": torch.Size([1]),
        "tensor_stride": (1,),
        "tensor_offset": 0,
        "storage_cls": torch.UntypedStorage,
        "dtype": torch.float32,
        "storage_device": 0,
        "storage_handle": b"fake_device_handle",
        "storage_size_bytes": 4,
        "storage_offset_bytes": 0,
        "requires_grad": False,
        "ref_counter_handle": b"fake_ref_handle",
        "ref_counter_offset": 0,
        "event_handle": b"fake_event_handle",
        "event_sync_required": False,
    }


@instantiate_parametrized_tests
class TestDeviceIpcReductions(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("device_type", ["cpu", "meta", "cuda", "not_a_real_device"])
    def test_device_supports_ipc_false(self, device_type):
        self.assertFalse(mp.reductions._device_supports_ipc(device_type))

    def test_device_supports_ipc_privateuse1_false_without_hooks(self):
        backend = torch._C._get_privateuse1_backend_name()
        self.assertFalse(mp.reductions._device_supports_ipc(backend))

    def test_typed_storage_share_device_delegates(self):
        storage = torch.zeros(2)._typed_storage()
        fake = (0, b"h", 8, 0, b"r", 0, b"e", False)
        with patch.object(
            storage._untyped_storage, "_share_device_", return_value=fake
        ) as mocked:
            result = storage._share_device_(1, foo=2)
        mocked.assert_called_once_with(1, foo=2)
        self.assertEqual(result, fake)

    def test_typed_storage_new_shared_device_delegates(self):
        with patch.object(
            torch.UntypedStorage, "_new_shared_device", return_value="ok"
        ) as mocked:
            result = torch.TypedStorage._new_shared_device(1, 2, foo=3)
        mocked.assert_called_once_with(1, 2, foo=3)
        self.assertEqual(result, "ok")

    def test_typed_storage_release_ipc_counter_device_delegates(self):
        with patch.object(
            torch.UntypedStorage, "_release_ipc_counter_device", return_value="ok"
        ) as mocked:
            result = torch.TypedStorage._release_ipc_counter_device(b"r", 0)
        mocked.assert_called_once_with(b"r", 0)
        self.assertEqual(result, "ok")

    def test_reduce_tensor_routes_to_rebuild_device_tensor_when_supported(self):
        t = torch.zeros(2)
        fake = (0, b"h", 8, 0, b"r", 0, b"e", False)
        with (
            patch.object(mp.reductions, "_device_supports_ipc", return_value=True),
            patch.object(torch.TypedStorage, "_share_device_", return_value=fake),
        ):
            func, args = mp.reductions.reduce_tensor(t)
        try:
            self.assertIs(func, mp.reductions.rebuild_device_tensor)
            self.assertEqual(args[7], b"h")
            self.assertEqual(args[8], 8)
        finally:
            mp.reductions.shared_cache.pop(("cpu", b"h"), None)

    def test_reduce_tensor_cpu_uses_rebuild_tensor(self):
        func, _args = mp.reductions.reduce_tensor(torch.zeros(2))
        self.assertIs(func, mp.reductions.rebuild_tensor)

    def test_reduce_tensor_meta_uses_rebuild_meta_tensor(self):
        func, _args = mp.reductions.reduce_tensor(torch.zeros(2, device="meta"))
        self.assertIs(func, mp.reductions.rebuild_meta_tensor)

    def test_rebuild_device_tensor_cache_miss_calls_new_shared_device(self):
        dummy = torch.UntypedStorage(4)
        kwargs = _rebuild_device_tensor_kwargs()
        cache_key = (
            torch._C._get_privateuse1_backend_name(),
            kwargs["storage_handle"],
            kwargs["storage_offset_bytes"],
        )
        with (
            patch.object(mp.reductions, "storage_from_cache", return_value=None),
            patch.object(
                torch.UntypedStorage, "_new_shared_device", return_value=dummy
            ) as mock_new,
        ):
            mp.reductions.rebuild_device_tensor(**kwargs)
        try:
            mock_new.assert_called_once_with(
                kwargs["storage_device"],
                kwargs["storage_handle"],
                kwargs["storage_size_bytes"],
                kwargs["storage_offset_bytes"],
                kwargs["ref_counter_handle"],
                kwargs["ref_counter_offset"],
                kwargs["event_handle"],
                kwargs["event_sync_required"],
            )
            self.assertIn(cache_key, mp.reductions.shared_cache)
        finally:
            mp.reductions.shared_cache.pop(cache_key, None)

    def test_rebuild_device_tensor_cache_hit_releases_ipc_counter(self):
        dummy = torch.UntypedStorage(4)
        kwargs = _rebuild_device_tensor_kwargs()
        with (
            patch.object(mp.reductions, "storage_from_cache", return_value=dummy),
            patch.object(torch.UntypedStorage, "_new_shared_device") as mock_new,
            patch.object(
                torch.UntypedStorage, "_release_ipc_counter_device"
            ) as mock_release,
        ):
            mp.reductions.rebuild_device_tensor(**kwargs)
        mock_new.assert_not_called()
        mock_release.assert_called_once_with(
            kwargs["ref_counter_handle"],
            kwargs["ref_counter_offset"],
            device=kwargs["storage_device"],
        )


if __name__ == "__main__":
    run_tests()
