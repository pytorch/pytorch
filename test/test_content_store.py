# Owner(s): ["oncall: pt2"]

import tempfile

import torch
from torch._prims.debug_prims import load_tensor_reader
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase
from torch.utils._content_store import (
    ContentStoreReader,
    ContentStoreWriter,
    hash_storage,
)


class TestContentStore(TestCase):
    def test_basic(self, device):
        # setup test data
        x = torch.randn(4, device=device)
        y = torch.randn(6, device=device)
        z = x.view(2, 2)
        # start writing
        with tempfile.TemporaryDirectory() as loc:
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

    @skipIfRocm
    def test_load_tensor(self, device):
        with tempfile.TemporaryDirectory() as loc:
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
                    self.assertIsInstance(x4, FakeTensor)
                    same_meta_as_x(x4)

                # Check fp64 works
                x5 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float64, device=device
                )
                self.assertEqual(x5.float(), x)
                self.assertEqual(x5.dtype, torch.float64)

        x6 = torch.ops.debugprims.load_tensor.default(
            "x", (4,), (1,), dtype=torch.float32, device=device
        )
        same_meta_as_x(x6)


instantiate_device_type_tests(TestContentStore, globals())


if __name__ == "__main__":
    run_tests()
