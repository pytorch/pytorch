# Owner(s): ["module: inductor"]

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime import cache_dir_utils
from torch._inductor.test_case import run_tests, TestCase


class TestCacheDirUtils(TestCase):
    def test_default_cache_dir_falls_back_to_uid(self):
        for exception in (
            KeyError("getpwuid(): uid not found: 1001"),
            ModuleNotFoundError("No module named 'pwd'"),
            OSError("getpwuid(): uid not found: 1001"),
        ):
            with (
                self.subTest(exception=type(exception)),
                mock.patch.object(
                    cache_dir_utils.getpass,
                    "getuser",
                    side_effect=exception,
                ),
                mock.patch.object(
                    cache_dir_utils.os, "getuid", return_value=1001, create=True
                ),
                mock.patch.object(cache_dir_utils, "is_fbcode", return_value=False),
            ):
                self.assertEqual(
                    cache_dir_utils.default_cache_dir(),
                    os.path.join(tempfile.gettempdir(), "torchinductor_uid_1001"),
                )

    def test_default_cache_dir_falls_back_without_getuid(self):
        with (
            mock.patch.object(
                cache_dir_utils.getpass,
                "getuser",
                side_effect=OSError("user unavailable"),
            ),
            mock.patch.object(cache_dir_utils, "os", SimpleNamespace(path=os.path)),
            mock.patch.object(cache_dir_utils, "is_fbcode", return_value=False),
        ):
            self.assertEqual(
                cache_dir_utils.default_cache_dir(),
                os.path.join(tempfile.gettempdir(), "torchinductor_unknown_user"),
            )


if __name__ == "__main__":
    run_tests()
