# Owner(s): ["oncall: distributed checkpointing"]

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
import uuid

import fsspec
from fsspec.implementations.local import LocalFileSystem

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from torch.distributed.checkpoint._hf_utils import (
    _get_fs_and_path,
    _metadata_fn,
    _open_safetensors,
)
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class _MemoryFsTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.name = f"test-{uuid.uuid4().hex}"
        self.root = f"memory://{self.name}"
        self.memory_fs = fsspec.filesystem("memory")
        self.addCleanup(self._remove_root)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _remove_root(self) -> None:
        if self.memory_fs.exists(f"/{self.name}"):
            self.memory_fs.rm(f"/{self.name}", recursive=True)

    def _write_bytes(self, url: str, data: bytes) -> None:
        with fsspec.open(url, "wb") as f:
            f.write(data)

    def _read_bytes(self, url: str) -> bytes:
        with fsspec.open(url, "rb") as f:
            return f.read()


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
@instantiate_parametrized_tests
class TestFsspecSafetensorsFile(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        from safetensors.torch import save

        self.tensors = {
            "a": torch.arange(24, dtype=torch.float32).view(4, 6),
            "b": torch.arange(8, dtype=torch.bfloat16).view(2, 2, 2),
            "s": torch.tensor(3.0),
            "u8": torch.arange(12, dtype=torch.uint8).view(3, 4),
            "i32": torch.arange(-6, 6, dtype=torch.int32).view(4, 3),
            "f8": torch.arange(6, dtype=torch.float32).to(torch.float8_e4m3fn),
            "bool": torch.tensor([[True, False], [False, True]]),
        }
        data = save(self.tensors, metadata={"k": "v"})
        self.local_path = os.path.join(self.temp_dir, "f.safetensors")
        with open(self.local_path, "wb") as f:
            f.write(data)
        self.remote_path = f"{self.root}/f.safetensors"
        self._write_bytes(self.remote_path, data)
        self.remote_fs, _ = _get_fs_and_path(self.remote_path)

    def test_get_fs_and_path_picks_backend(self) -> None:
        self.assertIsInstance(_get_fs_and_path(self.local_path)[0], FileSystem)
        self.assertIsInstance(self.remote_fs, _FileSystem)

    def test_header_matches_safe_open(self) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            self.assertEqual(sorted(remote.keys()), sorted(local.keys()))
            self.assertEqual(remote.metadata(), local.metadata())
            for key in self.tensors:
                self.assertEqual(
                    remote.get_slice(key).get_shape(), local.get_slice(key).get_shape()
                )
                self.assertEqual(
                    remote.get_slice(key).get_dtype(), local.get_slice(key).get_dtype()
                )

    @parametrize(
        "key,index",
        [
            ("a", ()),
            ("a", (slice(1, 3),)),
            ("a", (slice(None), slice(2, 5))),
            ("a", (slice(1, 3), slice(0, 4))),
            ("a", (slice(3, 3),)),
            ("a", (slice(3, 1),)),
            ("a", (slice(-2, None),)),
            ("a", (slice(0, 4, 2),)),
            ("a", (slice(None), slice(0, 6, 2))),
            ("a", 1),
            ("a", -1),
            ("a", (slice(None), 0)),
            ("a", (1, 2)),
            ("a", Ellipsis),
            ("a", (Ellipsis, slice(1, 3))),
            ("a", (0, Ellipsis)),
            ("b", (slice(1, 2), slice(None), slice(0, 1))),
            ("b", (Ellipsis, 0)),
            ("s", ()),
            ("s", Ellipsis),
            ("u8", (slice(1, 3), slice(1, 4))),
            ("i32", (slice(2, 4),)),
            ("f8", (slice(2, 5),)),
            ("bool", (slice(1, 2),)),
        ],
    )
    def test_index_matches_safe_open(self, key, index) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            result = remote.get_slice(key)[index]
            self.assertEqual(result, local.get_slice(key)[index])
            self.assertEqual(result, self.tensors[key][index])

    @parametrize(
        "key,index",
        [
            ("a", 4),
            ("a", -5),
            ("a", (slice(None),) * 3),
            ("a", (slice(None, None, -1),)),
            ("s", 0),
        ],
    )
    def test_invalid_index_raises_like_safe_open(self, key, index) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            with self.assertRaises(Exception) as expected:
                local.get_slice(key)[index]
            with self.assertRaises(type(expected.exception)):
                remote.get_slice(key)[index]

    def test_get_tensor(self) -> None:
        with _open_safetensors(self.remote_fs, self.remote_path) as remote:
            for key, tensor in self.tensors.items():
                self.assertEqual(remote.get_tensor(key), tensor)


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
@instantiate_parametrized_tests
class TestConsolidateOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        from safetensors.torch import save

        self.w = torch.arange(16, dtype=torch.float32).view(4, 4)
        self.b = torch.arange(4, dtype=torch.float32)

        def sharding(offsets: dict[str, list[int]]) -> dict[str, str]:
            info = {k: {"saved_offsets": v} for k, v in offsets.items()}
            return {"DCP_SHARDING_INFO": json.dumps(info)}

        self.shards = {
            "shard-00001-model-00001-of-00001.safetensors": save(
                {"w": self.w[:2].contiguous(), "b": self.b},
                metadata=sharding({"w": [0, 0], "b": [0]}),
            ),
            "shard-00002-model-00001-of-00001.safetensors": save(
                {"w": self.w[2:].contiguous()}, metadata=sharding({"w": [2, 0]})
            ),
        }
        self.mapping = {"w": 1, "b": 2}

    def _write_remote_shards(self) -> None:
        for name, data in self.shards.items():
            self._write_bytes(f"{self.root}/sharded/{name}", data)

    # num_threads=2 writes the two output files from worker threads.
    @parametrize("num_threads", [1, 2])
    def test_consolidate_to_remote(self, num_threads: int) -> None:
        from safetensors.torch import load

        self._write_remote_shards()
        consolidate_safetensors_files(
            input_dir=f"{self.root}/sharded",
            output_dir=self.root,
            fqn_to_index_mapping=self.mapping,
            num_threads=num_threads,
        )

        w_file = self._read_bytes(f"{self.root}/model-00001-of-00002.safetensors")
        b_file = self._read_bytes(f"{self.root}/model-00002-of-00002.safetensors")
        self.assertEqual(load(w_file), {"w": self.w})
        self.assertEqual(load(b_file), {"b": self.b})
        index = json.loads(self._read_bytes(f"{self.root}/{_metadata_fn}"))
        self.assertEqual(
            index["weight_map"],
            {
                "w": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        )

    def test_consolidate_several_tensors_per_file(self) -> None:
        from safetensors.torch import load

        self._write_remote_shards()
        consolidate_safetensors_files(
            input_dir=f"{self.root}/sharded",
            output_dir=self.root,
            fqn_to_index_mapping={"w": 1, "b": 1},
        )

        data = self._read_bytes(f"{self.root}/model-00001-of-00001.safetensors")
        self.assertEqual(load(data), {"w": self.w, "b": self.b})

    def test_local_and_remote_outputs_are_identical(self) -> None:
        local_in = os.path.join(self.temp_dir, "sharded")
        os.makedirs(local_in)
        for name, data in self.shards.items():
            with open(os.path.join(local_in, name), "wb") as f:
                f.write(data)
        self._write_remote_shards()
        # Inputs and outputs may live on different filesystems.
        local_from_remote = os.path.join(self.temp_dir, "from_remote")
        os.makedirs(local_from_remote)
        for input_dir, output_dir in (
            (local_in, self.temp_dir),
            (f"{self.root}/sharded", self.root),
            (local_in, f"{self.root}/from_local"),
            (f"{self.root}/sharded", local_from_remote),
        ):
            consolidate_safetensors_files(input_dir, output_dir, self.mapping)

        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            _metadata_fn,
        ):
            with open(os.path.join(self.temp_dir, name), "rb") as f:
                expected = f.read()
            with open(os.path.join(local_from_remote, name), "rb") as f:
                self.assertEqual(f.read(), expected)
            self.assertEqual(self._read_bytes(f"{self.root}/{name}"), expected)
            self.assertEqual(
                self._read_bytes(f"{self.root}/from_local/{name}"), expected
            )


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
class TestHfStorageOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        self.path = f"{self.root}/ckpt"
        self.state_dict = {"w": torch.rand(4, 6), "b": torch.rand(6)}

    def _load(self, reader) -> dict[str, torch.Tensor]:
        loaded = {k: torch.zeros_like(v) for k, v in self.state_dict.items()}
        dist_cp.load(loaded, storage_reader=reader)
        return loaded

    def test_save_load_round_trip(self) -> None:
        dist_cp.save(
            self.state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(path=self.path),
        )
        for name in ("model-00001-of-00001.safetensors", _metadata_fn):
            self.assertTrue(self.memory_fs.exists(f"/{self.name}/ckpt/{name}"))
        loaded = self._load(dist_cp.HuggingFaceStorageReader(path=self.path))
        self.assertEqual(loaded, self.state_dict)

    def test_save_distributed_with_consolidation(self) -> None:
        dist_cp.save(
            self.state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.path,
                fqn_to_index_mapping={"w": 1, "b": 2},
                save_distributed=True,
                enable_consolidation=True,
            ),
        )
        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            _metadata_fn,
        ):
            self.assertTrue(self.memory_fs.exists(f"/{self.name}/ckpt/{name}"))
        loaded = self._load(
            dist_cp.HuggingFaceStorageReader(path=self.path, thread_count=2)
        )
        self.assertEqual(loaded, self.state_dict)


class _ListingCachingLocalFileSystem(LocalFileSystem):
    """``file://`` that caches directory listings, as gcsfs, s3fs and adlfs do."""

    def ls(self, path, detail=False, **kwargs):
        listing = self.dircache.get((path, detail))
        if listing is None:
            listing = super().ls(path, detail=detail, **kwargs)
            self.dircache[(path, detail)] = listing
        return listing


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
class TestHfStorageReaderSeesExternalWrites(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        fsspec.register_implementation(
            "file", _ListingCachingLocalFileSystem, clobber=True
        )
        self.addCleanup(
            fsspec.register_implementation, "file", LocalFileSystem, clobber=True
        )
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.url = f"file://{self.folder}"
        self._write_externally(
            "model-00001-of-00002.safetensors", {"w": torch.ones(2, 2)}
        )

    def _write_externally(self, name: str, tensors: dict[str, torch.Tensor]) -> None:
        # Written outside this process's fsspec instances, as another rank, job
        # or upload does.
        from safetensors.torch import save_file

        save_file(tensors, os.path.join(self.folder, name))

    def _load_both(self, reader, checkpoint_id=None) -> dict[str, torch.Tensor]:
        loaded = {"w": torch.zeros(2, 2), "b": torch.zeros(2)}
        dist_cp.load(loaded, storage_reader=reader, checkpoint_id=checkpoint_id)
        return loaded

    def test_new_reader_sees_file_written_after_earlier_listing(self) -> None:
        # A load retried after an upload finished, or a reload of a folder that
        # was exported again, must not be answered from an older listing.
        dist_cp.HuggingFaceStorageReader(path=self.url).read_metadata()
        self._write_externally(
            "model-00002-of-00002.safetensors", {"b": torch.full((2,), 2.0)}
        )

        loaded = self._load_both(dist_cp.HuggingFaceStorageReader(path=self.url))

        self.assertEqual(loaded, {"w": torch.ones(2, 2), "b": torch.full((2,), 2.0)})

    def test_reused_reader_sees_file_written_after_earlier_load(self) -> None:
        reader = dist_cp.HuggingFaceStorageReader(path=self.url)
        dist_cp.load(
            {"w": torch.zeros(2, 2)}, storage_reader=reader, checkpoint_id=self.url
        )
        self._write_externally(
            "model-00002-of-00002.safetensors", {"b": torch.full((2,), 2.0)}
        )

        loaded = self._load_both(reader, checkpoint_id=self.url)

        self.assertEqual(loaded, {"w": torch.ones(2, 2), "b": torch.full((2,), 2.0)})


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
class TestQuantizedHfStorageOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_scale_in_separate_file(self) -> None:
        from safetensors.torch import save

        # Each weight's scale lives in the other file, so with two files the
        # reader's worker threads open scale files while reading weights.
        original1 = torch.randn(256, 128)
        original2 = torch.randn(256, 128)
        file1 = "model-00001-of-00002.safetensors"
        file2 = "model-00002-of-00002.safetensors"
        self._write_bytes(
            f"{self.root}/{file1}",
            save(
                {
                    "linear1.weight": original1 / 2.0,
                    "linear2.weight_scale_inv": torch.full((2, 1), 4.0),
                }
            ),
        )
        self._write_bytes(
            f"{self.root}/{file2}",
            save(
                {
                    "linear2.weight": original2 / 4.0,
                    "linear1.weight_scale_inv": torch.full((2, 1), 2.0),
                }
            ),
        )
        weight_map = {
            "linear1.weight": file1,
            "linear2.weight_scale_inv": file1,
            "linear2.weight": file2,
            "linear1.weight_scale_inv": file2,
        }
        self._write_bytes(
            f"{self.root}/{_metadata_fn}",
            json.dumps({"weight_map": weight_map}).encode(),
        )

        loaded = {
            "linear1.weight": torch.zeros(256, 128),
            "linear2.weight": torch.zeros(256, 128),
        }
        dist_cp.load(
            loaded,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=self.root, block_size=128, thread_count=2
            ),
        )
        self.assertEqual(loaded["linear1.weight"], original1)
        self.assertEqual(loaded["linear2.weight"], original2)


if __name__ == "__main__":
    run_tests()
