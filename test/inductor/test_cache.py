# Owner(s): ["module: inductor"]
import functools
import pkgutil
import shutil
import unittest
import unittest.mock as mock

import torch._inductor.codecache as codecache

requires_fsspec = functools.partial(
    unittest.skipIf, not pkgutil.find_loader("fsspec"), "requires fsspec"
)
import os
import tempfile

from torch.testing._internal.common_utils import TestCase


class CacheTests(TestCase):
    def setUp(self):
        codecache.cache_dir.cache_clear()
        self.cache_dir_function = codecache.cache_dir
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @requires_fsspec()
    def test_cache_hit(self):
        cache_path = os.path.join(self.temp_dir, "my-cache-dir")
        os.makedirs(cache_path)
        with mock.patch.dict("os.environ", {"TORCHINDUCTOR_CACHE_DIR": cache_path}):
            result = self.cache_dir_function()

            # Validate that the directory already exists and matches the expected path
            self.assertTrue(os.path.exists(cache_path))
            self.assertEqual(result, cache_path)

    @requires_fsspec()
    def test_cache_miss(self):
        cache_path = os.path.join(self.temp_dir, "my-cache-dir")
        with mock.patch.dict("os.environ", {"TORCHINDUCTOR_CACHE_DIR": cache_path}):
            result = self.cache_dir_function()

            # Validate that the directory was created and matches the expected path
            self.assertTrue(os.path.exists(cache_path))
            self.assertEqual(result, cache_path)

    @requires_fsspec()
    def test_cache_s3_exists(self):
        s3_path = "s3://my-bucket/my-dir"
        with mock.patch.dict("os.environ", {"TORCHINDUCTOR_CACHE_DIR": s3_path}):
            with mock.patch("fsspec.filesystem") as mocked_filesystem:
                # Mocking the filesystem to simulate the directory exists on S3
                mock_fs = mock.MagicMock()
                mock_fs.exists.return_value = True
                mocked_filesystem.return_value = mock_fs

                result = self.cache_dir_function()

                # Validate the result
                self.assertEqual(result, s3_path)
                mock_fs.exists.assert_called_once_with(s3_path)
                mock_fs.makedirs.assert_not_called()  # It shouldn't try to create it since it exists

    @requires_fsspec()
    def test_cache_s3_does_not_exist(self):
        s3_path = "s3://my-bucket/my-dir"
        with mock.patch.dict("os.environ", {"TORCHINDUCTOR_CACHE_DIR": s3_path}):
            with mock.patch("fsspec.filesystem") as mocked_filesystem:
                # Mocking the filesystem to simulate the directory doesn't exist on S3
                mock_fs = mock.MagicMock()
                mock_fs.exists.return_value = False
                mocked_filesystem.return_value = mock_fs

                result = self.cache_dir_function()

                # Validate the result
                self.assertEqual(result, s3_path)
                mock_fs.exists.assert_called_once_with(s3_path)
                mock_fs.makedirs.assert_called_once_with(s3_path, exist_ok=True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests(needs="fsspec")
