# Owner(s): ["oncall: distributed"]

import tempfile
from unittest import mock

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.distributed.checkpoint_utils import (
    Rot13Example,
    get_test_extension_registry,
)


class _FileSystemSubclass(FileSystem):
    """Subclass so `type(self.fs) is FileSystem` is false."""


class _SlicePlanner:
    def __init__(self, dest: torch.Tensor):
        self.dest = dest

    def resolve_tensor(self, req):
        return self.dest

    def commit_tensor(self, req, tensor):
        pass


class _ByteBlob:
    def __init__(self, payload: bytes = b""):
        self.payload = payload

    def state_dict(self):
        return {"raw": self.payload}

    def load_state_dict(self, sd):
        self.payload = sd["raw"]


class TestFileSystemReaderMmap(TestCase):
    def _roundtrip(self, mmap: bool, state_dict=None):
        if state_dict is None:
            state_dict = {
                "a": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                "b": torch.ones(8, dtype=torch.float32),
            }
        with tempfile.TemporaryDirectory() as path:
            save(state_dict, storage_writer=FileSystemWriter(path=path), no_dist=True)
            loaded = {k: torch.empty_like(v) for k, v in state_dict.items()}
            reader = FileSystemReader(path=path, mmap=mmap)
            load(loaded, storage_reader=reader, no_dist=True)
            for k in state_dict:
                self.assertEqual(loaded[k], state_dict[k])

    def test_mmap_false_matches_eager_values(self):
        self._roundtrip(mmap=False)

    def test_mmap_true_matches_eager_values(self):
        self._roundtrip(mmap=True)

    def test_mmap_false_does_not_call_from_file(self):
        with mock.patch.object(
            torch.UntypedStorage,
            "from_file",
            wraps=torch.UntypedStorage.from_file,
        ) as wrapped:
            self._roundtrip(mmap=False)
        wrapped.assert_not_called()

    def test_same_file_multiple_requests_maps_once(self):
        with mock.patch.object(
            torch.UntypedStorage,
            "from_file",
            wraps=torch.UntypedStorage.from_file,
        ) as wrapped:
            self._roundtrip(mmap=True)
        self.assertEqual(wrapped.call_count, 1)
        self.assertIs(wrapped.call_args.kwargs["shared"], False)

    def test_replaced_checkpoint_at_same_path_reads_new_data(self):
        with tempfile.TemporaryDirectory() as path:
            first = {"w": torch.zeros(4)}
            second = {"w": torch.ones(4) * 7}
            save(first, storage_writer=FileSystemWriter(path=path), no_dist=True)
            reader = FileSystemReader(path=path, mmap=True)
            loaded = {"w": torch.empty(4)}
            load(loaded, storage_reader=reader, no_dist=True)
            self.assertEqual(loaded["w"], first["w"])

            save(second, storage_writer=FileSystemWriter(path=path), no_dist=True)
            loaded2 = {"w": torch.empty(4)}
            load(loaded2, storage_reader=reader, no_dist=True)
            self.assertEqual(loaded2["w"], second["w"])

    def test_byte_io_does_not_use_from_file(self):
        with tempfile.TemporaryDirectory() as path:
            blob = _ByteBlob(b"hello-bytes")
            save(
                {"blob": blob}, storage_writer=FileSystemWriter(path=path), no_dist=True
            )
            with mock.patch.object(
                torch.UntypedStorage,
                "from_file",
                wraps=torch.UntypedStorage.from_file,
            ) as wrapped:
                loaded_blob = _ByteBlob()
                load(
                    {"blob": loaded_blob},
                    storage_reader=FileSystemReader(path=path, mmap=True),
                    no_dist=True,
                )
            wrapped.assert_not_called()
            self.assertEqual(loaded_blob.payload, b"hello-bytes")

    def test_filesystem_subclass_skips_mmap(self):
        with tempfile.TemporaryDirectory() as path:
            state = {"a": torch.arange(4.0)}
            save(state, storage_writer=FileSystemWriter(path=path), no_dist=True)
            reader = FileSystemReader(path=path, mmap=True)
            reader.fs = _FileSystemSubclass()
            with mock.patch.object(torch.UntypedStorage, "from_file") as from_file:
                loaded = {"a": torch.empty(4)}
                load(loaded, storage_reader=reader, no_dist=True)
            from_file.assert_not_called()
            self.assertEqual(loaded["a"], state["a"])

    def test_transform_descriptor_skips_mmap(self):
        with tempfile.TemporaryDirectory() as path:
            state = {"a": torch.arange(4.0)}
            writer = FileSystemWriter(path=path, _extensions=[Rot13Example()])
            save(state, storage_writer=writer, no_dist=True)
            reader = FileSystemReader(
                path=path,
                mmap=True,
                _extension_registry=get_test_extension_registry(),
            )
            with mock.patch.object(torch.UntypedStorage, "from_file") as from_file:
                loaded = {"a": torch.empty(4)}
                load(loaded, storage_reader=reader, no_dist=True)
            from_file.assert_not_called()
            self.assertEqual(loaded["a"], state["a"])

    def test_unsupported_zip_region_falls_back_to_torch_load(self):
        with mock.patch.object(
            torch.serialization,
            "_load_zip_region",
            side_effect=torch.serialization._UnsupportedZipRegion("unsupported"),
        ):
            self._roundtrip(mmap=True)

    def test_storage_range_outside_file_raises(self):
        with tempfile.TemporaryDirectory() as path:
            state = {"a": torch.arange(4.0)}
            save(state, storage_writer=FileSystemWriter(path=path), no_dist=True)
            reader = FileSystemReader(path=path, mmap=True)
            metadata = reader.read_metadata()
            reader.set_up_storage_reader(metadata, True)
            idx = next(iter(reader.storage_data.keys()))
            md = reader.storage_data[idx]
            md.offset = 10**12
            item = ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=idx,
                dest_offsets=torch.Size([0]),
                storage_index=idx,
                storage_offsets=torch.Size([0]),
                lengths=torch.Size([4]),
            )
            dest = torch.empty(4)
            with self.assertRaisesRegex(RuntimeError, "outside the backing file"):
                reader.read_data(LoadPlan(items=[item]), _SlicePlanner(dest)).wait()

    def test_partial_narrow_matches_native(self):
        src = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        expected = src[1:3, 1:3].contiguous()
        with tempfile.TemporaryDirectory() as path:
            save({"w": src}, storage_writer=FileSystemWriter(path=path), no_dist=True)

            def run(mmap: bool) -> torch.Tensor:
                reader = FileSystemReader(path=path, mmap=mmap)
                metadata = reader.read_metadata()
                reader.set_up_storage_reader(metadata, True)
                idx = next(iter(reader.storage_data.keys()))
                item = ReadItem(
                    type=LoadItemType.TENSOR,
                    dest_index=idx,
                    dest_offsets=torch.Size([0, 0]),
                    storage_index=idx,
                    storage_offsets=torch.Size([1, 1]),
                    lengths=torch.Size([2, 2]),
                )
                dest = torch.empty(2, 2)
                reader.read_data(LoadPlan(items=[item]), _SlicePlanner(dest)).wait()
                return dest

            mmap_out = run(True)
            native_out = run(False)
            self.assertEqual(mmap_out, expected)
            self.assertEqual(native_out, expected)
            self.assertEqual(mmap_out, native_out)


if __name__ == "__main__":
    run_tests()
