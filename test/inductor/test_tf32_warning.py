# Owner(s): ["module: inductor"]
import warnings
from unittest import mock

import torch
from torch._inductor.test_case import run_tests, TestCase


class TestTF32WarningSuppression(TestCase):
    def _collect_warnings(self, warn_fn):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            warn_fn()
        return records

    def test_warn_tf32_disabled_respects_manual_precision(self):
        from torch._inductor import compile_fx
        from torch._inductor.fx_passes import fuse_attention

        for warn_module in (compile_fx, fuse_attention):
            warn_module._warn_tf32_disabled.cache_clear()

        prev_precision = torch.backends.cuda.matmul.fp32_precision
        prev_user_flag = torch._float32_matmul_precision_set_by_user
        try:
            torch.backends.cuda.matmul.fp32_precision = "highest"
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            ):
                with mock.patch(
                    "torch._float32_matmul_precision_set_by_user", False
                ):
                    records = self._collect_warnings(
                        compile_fx._warn_tf32_disabled
                    )
                    self.assertEqual(len(records), 1)

                compile_fx._warn_tf32_disabled.cache_clear()
                with mock.patch("torch._float32_matmul_precision_set_by_user", True):
                    records = self._collect_warnings(
                        compile_fx._warn_tf32_disabled
                    )
                    self.assertEqual(len(records), 0)

                fuse_attention._warn_tf32_disabled.cache_clear()
                with mock.patch("torch._float32_matmul_precision_set_by_user", True):
                    records = self._collect_warnings(
                        fuse_attention._warn_tf32_disabled
                    )
                    self.assertEqual(len(records), 0)
        finally:
            torch.backends.cuda.matmul.fp32_precision = prev_precision
            torch._float32_matmul_precision_set_by_user = prev_user_flag


if __name__ == "__main__":
    run_tests()
