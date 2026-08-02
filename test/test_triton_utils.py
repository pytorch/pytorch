# Owner(s): ["module: inductor"]

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils import _triton as triton_utils


class TestTritonUtils(TestCase):
    def _run_triton_backend_load_failure(self, exc, env):
        driver_mod = importlib.import_module("triton.runtime.driver")
        fake_driver = SimpleNamespace(
            active=SimpleNamespace(get_current_target=mock.Mock(side_effect=exc))
        )

        triton_utils.triton_backend.cache_clear()
        try:
            with (
                mock.patch.object(driver_mod, "driver", fake_driver),
                mock.patch.dict(os.environ, env),
            ):
                if "TRITON_CACHE_DIR" not in env:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                cache_dir = triton_utils._triton_cache_dir_for_error_message()
                with self.assertRaises(ImportError) as cm:
                    triton_utils.triton_backend()
        finally:
            triton_utils.triton_backend.cache_clear()

        return str(cm.exception), cache_dir

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reports_noexec_cache_dir(self):
        for exc_type in (ImportError, OSError):
            with self.subTest(exc_type=exc_type):
                msg, _ = self._run_triton_backend_load_failure(
                    exc_type(
                        "/noexec/triton/hash/cuda_utils.so: "
                        "failed to map segment from shared object"
                    ),
                    {"TRITON_CACHE_DIR": "/noexec/triton"},
                )

                self.assertIn("TRITON_CACHE_DIR=/noexec/triton", msg)
                self.assertIn("noexec", msg)
                self.assertIn("executable filesystem", msg)

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reports_default_cache_dir(self):
        msg, cache_dir = self._run_triton_backend_load_failure(
            ImportError(
                "/default/triton/hash/cuda_utils.so: "
                "failed to map segment from shared object"
            ),
            {},
        )

        self.assertIsNotNone(cache_dir)
        self.assertIn(f"({cache_dir})", msg)
        self.assertIn("noexec", msg)
        self.assertIn("executable filesystem", msg)

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reraises_other_load_errors(self):
        driver_mod = importlib.import_module("triton.runtime.driver")

        for exc_type in (ImportError, OSError):
            fake_driver = SimpleNamespace(
                active=SimpleNamespace(
                    get_current_target=mock.Mock(side_effect=exc_type("other failure"))
                )
            )

            triton_utils.triton_backend.cache_clear()
            try:
                with (
                    self.subTest(exc_type=exc_type),
                    mock.patch.object(driver_mod, "driver", fake_driver),
                ):
                    with self.assertRaisesRegex(exc_type, "other failure"):
                        triton_utils.triton_backend()
            finally:
                triton_utils.triton_backend.cache_clear()


if __name__ == "__main__":
    run_tests()
