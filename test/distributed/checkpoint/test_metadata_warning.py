import warnings
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import TestCase

from torch.distributed.checkpoint import _metadata_warning
from torch.distributed.checkpoint.filesystem import FileSystemReader


class TestMetadataWarning(TestCase):
    def test_read_metadata_warns_about_pickle_once(self):
        original_warned = _metadata_warning._warned
        original_reader = FileSystemReader.read_metadata
        installed = getattr(
            FileSystemReader, "_native_neo_metadata_warning_installed", False
        )

        def read_metadata(_self, *_args, **_kwargs):
            return object()

        try:
            _metadata_warning._warned = False
            with patch.object(FileSystemReader, "read_metadata", read_metadata):
                FileSystemReader._native_neo_metadata_warning_installed = False
                _metadata_warning.install()
                reader = FileSystemReader(".")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reader.read_metadata()
                    reader.read_metadata()

            pickle_warnings = [
                w for w in caught if "deserialized with pickle" in str(w.message)
            ]
            self.assertEqual(len(pickle_warnings), 1)
        finally:
            _metadata_warning._warned = original_warned
            FileSystemReader.read_metadata = original_reader
            FileSystemReader._native_neo_metadata_warning_installed = installed

    def test_installation_is_idempotent(self):
        installed = FileSystemReader.read_metadata
        _metadata_warning.install()
        self.assertIs(FileSystemReader.read_metadata, installed)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
