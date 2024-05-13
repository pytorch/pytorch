import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFloat32Precision(TestCase):
    def test_mlkdnn_get_set(self):
        # get/set matmul
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "ieee")
        # get/set conv
        torch.backends.mkldnn.conv.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.conv.fp32_precision, "bf16")
        torch.backends.mkldnn.conv.fp32_precision = "ieee"
        self.assertEqual(torch.backends.mkldnn.conv.fp32_precision, "ieee")
        # get/set rnn
        torch.backends.mkldnn.rnn.fp32_precision = "bf16"
        self.assertEqual(torch.backends.mkldnn.rnn.fp32_precision, "bf16")
        torch.backends.mkldnn.rnn.fp32_precision = "ieee"
        self.assertEqual(torch.backends.mkldnn.rnn.fp32_precision, "ieee")
        # get/set mkldnn ops
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="bf16"):
            self.assertEqual(torch.backends.mkldnn.fp32_precision, "bf16")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="ieee"):
            self.assertEqual(torch.backends.mkldnn.fp32_precision, "ieee")

    def test_cudnn_get_set(self):
        # get/set conv
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "ieee")
        # get/set rnn
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "ieee")
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
            fp32_precision="ieee",
        ):
            self.assertEqual(torch.backends.cudnn.fp32_precision, "ieee")

    def test_cuda_get_set(self):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")

    def test_generic_precision(self):
        with torch.backends.flags(fp32_precision="ieee"):
            self.assertEqual(torch.backends.fp32_precision, "ieee")
        with torch.backends.flags(fp32_precision="tf32"):
            self.assertEqual(torch.backends.fp32_precision, "tf32")

    def test_default_use_parent(self):
        torch.backends.mkldnn.matmul.fp32_precision = "default"
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="bf16"):
            self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="default"):
            self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "ieee")

    def test_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid precision"):
            torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "Invalid precision"):
            torch.backends.cuda.matmul.fp32_precision = "bf16"

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
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "ieee")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "ieee")

    def test_fp32_precision_with_float32_matmul_precision(self):
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "ieee")
        torch.set_float32_matmul_precision("high")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "ieee")
        torch.set_float32_matmul_precision("medium")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")


if __name__ == "__main__":
    run_tests()
