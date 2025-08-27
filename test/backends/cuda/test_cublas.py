# Owner(s): ["module: cuda"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCublas(TestCase):
    def test_cublas_flags(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False

        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, False)
        self.assertEqual(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, False
        )
        self.assertEqual(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, False
        )
        self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, False)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, True)
        self.assertEqual(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
        )
        self.assertEqual(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
        )
        self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False

        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, False)
        self.assertEqual(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, False
        )
        self.assertEqual(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, False
        )
        self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, False)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, True)
        self.assertEqual(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
        )
        self.assertEqual(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
        )
        self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)

        with torch.backends.cuda.matmul.flags(
            allow_fp16_reduced_precision_reduction=False
        ):
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, False
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, False
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
            )
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)

        with torch.backends.cuda.matmul.flags(
            allow_bf16_reduced_precision_reduction=False
        ):
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, False
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, False
            )
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, True)

        with torch.backends.cuda.matmul.flags(allow_fp16_accumulation=False):
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
            )
            self.assertEqual(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, True
            )
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, False)
            self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation, False)


if __name__ == "__main__":
    run_tests()
