import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

skipIfFBGEMMGenAINotInstalled = unittest.skipIf(
    "FBGEMM_GENAI" not in torch.__config__.show(),
    "Tests that require FBGEMM_GENAI",
)


class TestFBGEMMGenAI(TestCase):
    @skipIfFBGEMMGenAINotInstalled
    def test_ops_available(self):
        M, N, K = 256, 256, 256
        xq = torch.randn(M, K).to(torch.float8_e4m3fn).cuda()
        wq = torch.randn(N, K).to(torch.float8_e4m3fn).cuda()

        row_scale = torch.randn(M).cuda()
        col_scale = torch.randn(N).cuda()

        _ = torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, row_scale, col_scale)

if __name__ == "__main__":
    run_tests()
