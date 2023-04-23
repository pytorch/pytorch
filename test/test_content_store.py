# Owner(s): ["oncall: pt2"]

import tempfile

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter


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


instantiate_device_type_tests(TestContentStore, globals())


if __name__ == "__main__":
    run_tests()
