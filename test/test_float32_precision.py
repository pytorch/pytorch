import contextlib
import functools

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

# This is a wrapper that wraps a test to run this test twice, one with
# allow_bf32=True, another with allow_bf32=False. When running with
# allow_bf32=True, it will use reduced precision as specified by the
# argument


def recover_orig_fp32_precision():
    @contextlib.contextmanager
    def recover():
        old_mkldnn_conv_p = torch.backends.mkldnn.conv.fp32_precision
        old_mkldnn_rnn_p = torch.backends.mkldnn.rnn.fp32_precision
        old_mkldnn_matmul_p = torch.backends.mkldnn.matmul.fp32_precision
        old_cudnn_conv_p = torch.backends.cudnn.conv.fp32_precision
        old_cudnn_rnn_p = torch.backends.cudnn.rnn.fp32_precision
        old_cuda_matmul_p = torch.backends.cuda.matmul.fp32_precision
        old_matmul_precision = torch.get_float32_matmul_precision()
        old_cubls_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            pass
            yield
        finally:
            torch.backends.mkldnn.conv.fp32_precision = old_mkldnn_conv_p
            torch.backends.mkldnn.rnn.fp32_precision = old_mkldnn_rnn_p
            torch.backends.mkldnn.matmul.fp32_precision = old_mkldnn_matmul_p
            torch.backends.cudnn.conv.fp32_precision = old_cudnn_conv_p
            torch.backends.cudnn.rnn.fp32_precision = old_cudnn_rnn_p
            torch.backends.cuda.matmul.fp32_precision = old_cuda_matmul_p
            torch.set_float32_matmul_precision(old_matmul_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_cubls_tf32

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            with recover():
                f(*args, **kwargs)

        return wrapped

    return wrapper


class TestFloat32Precision(TestCase):
    @recover_orig_fp32_precision()
    def test_mlkdnn_get_set(self):
        # get/set mkldnn ops
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="bf16"):
            self.assertEqual(torch.backends.mkldnn.fp32_precision, "bf16")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="default"):
            self.assertEqual(torch.backends.mkldnn.fp32_precision, "default")
        # get/set matmul
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
        torch.backends.mkldnn.matmul.fp32_precision = "default"
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "default")
        # get/set conv
        torch.backends.mkldnn.conv.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.conv.fp32_precision, "bf16")
        torch.backends.mkldnn.conv.fp32_precision = "default"
        self.assertEqual(torch.backends.mkldnn.conv.fp32_precision, "default")
        # get/set rnn
        torch.backends.mkldnn.rnn.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.rnn.fp32_precision, "bf16")
        torch.backends.mkldnn.rnn.fp32_precision = "default"
        self.assertEqual(torch.backends.mkldnn.rnn.fp32_precision, "default")

    @recover_orig_fp32_precision()
    def test_cudnn_get_set(self):
        # get/set cudnn ops
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=None,
            fp32_precision="tf32",
        ):
            self.assertEqual(torch.backends.cudnn.fp32_precision, "tf32")
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=None,
            fp32_precision="default",
        ):
            self.assertEqual(torch.backends.cudnn.fp32_precision, "default")
        # get/set conv
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
        torch.backends.cudnn.conv.fp32_precision = "default"
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "default")
        # get/set rnn
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")
        torch.backends.cudnn.rnn.fp32_precision = "default"
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "default")

    @recover_orig_fp32_precision()
    def test_cuda_get_set(self):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        torch.backends.cuda.matmul.fp32_precision = "default"
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "default")

    def test_generic_precision(self):
        with torch.backends.flags(fp32_precision="default"):
            self.assertEqual(torch.backends.fp32_precision, "default")
        with torch.backends.flags(fp32_precision="tf32"):
            self.assertEqual(torch.backends.fp32_precision, "tf32")

    @recover_orig_fp32_precision()
    def test_default_use_parent(self):
        torch.backends.mkldnn.matmul.fp32_precision = "default"
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="bf16"):
            self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="default"):
            with torch.backends.flags(fp32_precision="bf16"):
                self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
            with torch.backends.flags(fp32_precision="tf32"):
                # when parent is a not supported precision, use default
                self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "default")

    @recover_orig_fp32_precision()
    def test_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid precision"):
            torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "Invalid precision"):
            torch.backends.cuda.matmul.fp32_precision = "bf16"

    @recover_orig_fp32_precision()
    def test_fp32_precision_with_tf32(self):
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=True,
            fp32_precision=None,
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")

        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=False,
            fp32_precision=None,
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "default")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "default")

    @recover_orig_fp32_precision()
    def test_fp32_precision_with_float32_matmul_precision(self):
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "default")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "default")
        torch.set_float32_matmul_precision("high")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "default")
        torch.set_float32_matmul_precision("medium")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")

    @recover_orig_fp32_precision()
    def test_invalid_status_for_legacy_api(self):
        torch.backends.cudnn.conv.fp32_precision = "default"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "Invalid status"):
            print(torch.backends.cudnn.allow_tf32)

        torch.set_float32_matmul_precision("highest")
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        with self.assertRaisesRegex(RuntimeError, "Invalid status"):
            print(torch.get_float32_matmul_precision())

        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "Invalid status"):
            print(torch.backends.cuda.matmul.allow_tf32)


if __name__ == "__main__":
    run_tests()
