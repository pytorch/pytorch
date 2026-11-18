# Owner(s): ["module: serialization"]

import io
import os
import tempfile
from unittest import mock

import torch
from torch.distributed.checkpoint.utils import _create_file_view
from torch.serialization import _load_zip_region
from torch.testing._internal.common_utils import TestCase, run_tests


def _write_concatenated_saves(path: str, tensors: list[torch.Tensor], pad: int = 64):
    offsets = []
    with open(path, "wb") as f:
        f.write(b"\0" * pad)
        for t in tensors:
            start = f.tell()
            torch.save(t, f)
            offsets.append((start, f.tell() - start))
    return offsets


class TestLoadZipRegion(TestCase):
    def test_nonzero_offset_multiple_zips_share_mapping(self):
        t0 = torch.arange(8, dtype=torch.float32)
        t1 = torch.arange(8, 16, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "multi.bin")
            regions = _write_concatenated_saves(path, [t0, t1], pad=96)
            size = os.path.getsize(path)
            mapped = torch.UntypedStorage.from_file(path, False, size)
            base = mapped.data_ptr()
            with open(path, "rb") as fh:
                loaded = []
                for offset, length in regions:
                    view = _create_file_view(fh, offset, length)
                    obj = _load_zip_region(
                        view,
                        overall_storage=mapped,
                        storage_base_offset=offset,
                    )
                    loaded.append(obj)
                    ptr = obj.untyped_storage().data_ptr()
                    self.assertGreaterEqual(ptr, base)
                    self.assertLess(ptr, base + size)
            self.assertEqual(loaded[0], t0)
            self.assertEqual(loaded[1], t1)

    def test_view_position_restored(self):
        t = torch.ones(4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "one.bin")
            regions = _write_concatenated_saves(path, [t], pad=8)
            offset, length = regions[0]
            mapped = torch.UntypedStorage.from_file(path, False, os.path.getsize(path))
            with open(path, "rb") as fh:
                view = _create_file_view(fh, offset, length)
                self.assertEqual(view.tell(), 0)
                _load_zip_region(
                    view,
                    overall_storage=mapped,
                    storage_base_offset=offset,
                )
                self.assertEqual(view.tell(), 0)

    def test_bad_zip_raises_runtime_error(self):
        blob = io.BytesIO(b"not-a-zip-archive")
        mapped = torch.UntypedStorage(64)
        with self.assertRaisesRegex(RuntimeError, "invalid bounded ZIP region"):
            _load_zip_region(blob, overall_storage=mapped, storage_base_offset=0)

    def test_record_size_mismatch_raises(self):
        t = torch.arange(4, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "one.bin")
            regions = _write_concatenated_saves(path, [t], pad=8)
            offset, length = regions[0]
            mapped = torch.UntypedStorage.from_file(path, False, os.path.getsize(path))
            with open(path, "rb") as fh:
                view = _create_file_view(fh, offset, length)
                with mock.patch.object(
                    torch._C.PyTorchFileReader,
                    "get_record_size",
                    return_value=10**9,
                ):
                    with self.assertRaisesRegex(RuntimeError, "does not match"):
                        _load_zip_region(
                            view,
                            overall_storage=mapped,
                            storage_base_offset=offset,
                        )

    def test_mapped_storage_out_of_bounds_raises(self):
        t = torch.arange(4, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "one.bin")
            regions = _write_concatenated_saves(path, [t], pad=8)
            offset, length = regions[0]
            # Intentionally tiny mapping so the storage slice is too short.
            mapped = torch.UntypedStorage(1)
            with open(path, "rb") as fh:
                view = _create_file_view(fh, offset, length)
                with self.assertRaisesRegex(RuntimeError, "out of bounds"):
                    _load_zip_region(
                        view,
                        overall_storage=mapped,
                        storage_base_offset=0,
                    )


if __name__ == "__main__":
    run_tests()
