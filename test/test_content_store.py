# Owner(s): ["oncall: pt2"]

import os
import stat
import unittest
from unittest import mock

import torch
from torch._prims.debug_prims import load_tensor_reader
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryDirectoryName,
    TestCase,
)
from torch.utils._content_store import (
    ContentStoreReader,
    ContentStoreWriter,
    hash_storage,
    zstd_available,
)


class TestContentStore(TestCase):
    def test_basic(self, device):
        # setup test data
        x = torch.randn(4, device=device)
        y = torch.randn(6, device=device)
        z = x.view(2, 2)
        # start writing
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc)
            writer.write_tensor("x", x)
            writer.write_tensor("y", y)
            writer.write_tensor("z", z)
            # do some mutation that is VC UNTRACKED
            x.data.add_(1)
            writer.write_tensor("x2", x)
            writer.write_tensor("y2", y)
            writer.write_tensor("z2", z)
            del writer

            reader = ContentStoreReader(loc)
            n_x = reader.read_tensor("x")
            n_y = reader.read_tensor("y")
            n_z = reader.read_tensor("z")
            self.assertEqual(n_x + 1, x)
            self.assertEqual(n_y, y)
            self.assertEqual(n_z + 1, z)
            self.assertEqual(
                StorageWeakRef(n_x.untyped_storage()),
                StorageWeakRef(n_z.untyped_storage()),
            )
            n_x2 = reader.read_tensor("x2")
            n_y2 = reader.read_tensor("y2")
            n_z2 = reader.read_tensor("z2")
            self.assertEqual(n_x2, x)
            self.assertEqual(n_y2, y)
            self.assertEqual(n_z2, z)
            self.assertEqual(
                StorageWeakRef(n_y2.untyped_storage()),
                StorageWeakRef(n_y.untyped_storage()),
            )

    def test_scalar(self, device):
        # Should not raise an error
        hash_storage(torch.tensor(2, device=device).untyped_storage())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_repeated_hash(self, device):
        # Test that repeated hashing doesn't trigger a recompile in dynamo
        # If it does, we will execute prims.xor_sum in eager which fails
        for _ in range(4):
            hash_storage(torch.tensor(2, device=device).untyped_storage())

    def test_load_tensor(self, device):
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc)
            x = torch.randn(4, device=device)

            def same_meta_as_x(t):
                self.assertEqual(t.size(), x.size())
                self.assertEqual(t.stride(), x.stride())
                self.assertEqual(t.dtype, x.dtype)
                self.assertEqual(t.device, x.device)

            writer.write_tensor("x", x)

            with load_tensor_reader(loc):
                x2 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x2)
                x3 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x3)
                # Must not alias!
                self.assertNotEqual(
                    StorageWeakRef(x.untyped_storage()),
                    StorageWeakRef(x2.untyped_storage()),
                )
                self.assertNotEqual(
                    StorageWeakRef(x2.untyped_storage()),
                    StorageWeakRef(x3.untyped_storage()),
                )

                # Check fake tensor mode works too
                with FakeTensorMode():
                    x4 = torch.ops.debugprims.load_tensor.default(
                        "x", (4,), (1,), dtype=torch.float32, device=device
                    )
                    self.assertTrue(is_fake_tensor(x4))
                    same_meta_as_x(x4)

                # Check fp64 works on non-MPS platforms, since MPS doesn't currently
                # support fp64.
                if not device.startswith("mps"):
                    x5 = torch.ops.debugprims.load_tensor.default(
                        "x", (4,), (1,), dtype=torch.float64, device=device
                    )
                    self.assertEqual(x5.float(), x)
                    self.assertEqual(x5.dtype, torch.float64)

        x6 = torch.ops.debugprims.load_tensor.default(
            "x", (4,), (1,), dtype=torch.float32, device=device
        )
        same_meta_as_x(x6)


instantiate_device_type_tests(
    TestContentStore, globals(), allow_mps=True, allow_xpu=True
)


class TestContentStoreCodec(TestCase):
    def write_and_read(self, storage, compression=None):
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc, compression=compression)
            h = writer.write_storage(storage)
            reader = ContentStoreReader(loc, storage_metadata=writer.storage_metadata)
            self.assertEqual(reader.read_storage(h).tolist(), storage.tolist())
            return h, writer.storage_metadata[h]

    def test_raw_storage_metadata(self):
        storage = torch.arange(16).untyped_storage()
        h, metadata = self.write_and_read(storage)
        self.assertEqual(metadata["codec"], "none")
        self.assertEqual(metadata["path"], os.path.join("storages", h))
        self.assertEqual(metadata["logical_nbytes"], storage.nbytes())

    @unittest.skipIf(not zstd_available(), "requires zstandard")
    def test_zstd_round_trip(self):
        h, metadata = self.write_and_read(
            torch.zeros(1 << 14).untyped_storage(), compression="zstd"
        )
        self.assertEqual(metadata["codec"], "zstd")
        self.assertEqual(metadata["path"], os.path.join("storages", f"{h}.zst"))
        self.assertLess(metadata["stored_nbytes"], metadata["logical_nbytes"])

    @unittest.skipIf(not zstd_available(), "requires zstandard")
    def test_incompressible_storage_is_stored_raw(self):
        storage = torch.randint(0, 256, (1 << 16,), dtype=torch.uint8)
        _, metadata = self.write_and_read(storage.untyped_storage(), "zstd")
        self.assertEqual(metadata["codec"], "none")

    @unittest.skipIf(not zstd_available(), "requires zstandard")
    @parametrize("compression", [None, "zstd"])
    def test_existing_compressed_storage_is_reused(self, compression):
        storage = torch.zeros(1 << 14).untyped_storage()
        with TemporaryDirectoryName() as loc:
            first = ContentStoreWriter(loc, compression="zstd")
            h = first.write_storage(storage)
            second = ContentStoreWriter(loc, compression=compression)
            self.assertEqual(second.write_storage(storage), h)
            self.assertEqual(
                ContentStoreReader(loc).read_storage(h).tolist(), storage.tolist()
            )
        self.assertGreater(first.bytes_written, 0)
        self.assertEqual(second.bytes_written, 0)
        self.assertEqual(second.storage_metadata[h], first.storage_metadata[h])

    @parametrize("compression", [None, "zstd"])
    def test_published_storage_preserves_default_permissions(self, compression):
        if compression == "zstd" and not zstd_available():
            self.skipTest("requires zstandard")
        storage = torch.zeros(1 << 14).untyped_storage()
        with TemporaryDirectoryName() as loc:
            control = os.path.join(loc, "control")
            torch.save(storage, control)
            writer = ContentStoreWriter(loc, compression=compression)
            h = writer.write_storage(storage)
            metadata = writer.storage_metadata[h]
            self.assertEqual(metadata["codec"], compression or "none")
            self.assertEqual(
                stat.S_IMODE(os.stat(os.path.join(loc, metadata["path"])).st_mode),
                stat.S_IMODE(os.stat(control).st_mode),
            )
            self.assertEqual(os.listdir(os.path.join(loc, "storages")), [os.path.basename(metadata["path"])])

    @unittest.skipIf(not zstd_available(), "requires zstandard")
    def test_compressed_tensor_reads_without_manifest(self):
        expected = torch.zeros(1 << 14)
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc, compression="zstd")
            writer.write_tensor("x", expected)
            self.assertEqual(next(iter(writer.storage_metadata.values()))["codec"], "zstd")
            self.assertEqual(ContentStoreReader(loc).read_tensor("x"), expected)

    def test_zstd_requires_zstandard(self):
        entry = {"path": "h.zst", "codec": "zstd", "logical_nbytes": 0}
        metadata = {"h": {**entry, "stored_nbytes": 0}}
        with (
            TemporaryDirectoryName() as loc,
            mock.patch("torch.utils._content_store._zstandard", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "zstandard"):
                ContentStoreWriter(loc, compression="zstd")
            reader = ContentStoreReader(loc, storage_metadata=metadata)
            with self.assertRaisesRegex(RuntimeError, "zstandard"):
                reader.read_storage("h")

    @parametrize(
        "case,error,message",
        [
            ("missing", FileNotFoundError, "missing from the manifest"),
            ("escapes", ValueError, "escapes the content store"),
            ("missing_file", FileNotFoundError, "missing-storage"),
            ("unknown_codec", ValueError, "unknown content-store codec"),
        ],
    )
    def test_storage_metadata_is_validated(self, case, error, message):
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc)
            h = writer.write_storage(torch.arange(4).untyped_storage())
            metadata = dict(writer.storage_metadata[h])
            if case == "escapes":
                metadata["path"] = os.path.join("..", "outside")
            elif case == "missing_file":
                metadata["path"] = os.path.join("storages", "missing-storage")
            elif case == "unknown_codec":
                metadata["codec"] = "lz4"
            storage_metadata = {} if case == "missing" else {h: metadata}
            reader = ContentStoreReader(loc, storage_metadata=storage_metadata)
            with self.assertRaisesRegex(error, message):
                reader.read_storage(h)


instantiate_parametrized_tests(TestContentStoreCodec)


if __name__ == "__main__":
    run_tests()
