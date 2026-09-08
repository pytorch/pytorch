import tempfile
import warnings

import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint.filesystem import FileSystemReader


class TestMetadataWarning(TestCase):
    def test_read_metadata_warns_about_pickle(self):
        with tempfile.TemporaryDirectory() as path:
            reader = FileSystemReader(path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    reader.read_metadata()
                except FileNotFoundError:
                    pass

        self.assertTrue(
            any("deserialized with pickle" in str(w.message) for w in caught)
        )


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
