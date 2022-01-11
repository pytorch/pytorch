import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestChunkConverter(AccTestCase):
    @parameterized.expand(
        [
            ("chunk", 3, 1),
            ("chunk", 2000, 2),
            ("chunk", 3, -2),
        ]
    )
    def test_chunk(self, _, chunk, dim):
        class Chunk(nn.Module):
            def forward(self, x):
                return x.chunk(chunk, dim)[0]

        inputs = [torch.randn(3, 10, 20)]
        self.run_test(
            Chunk(),
            inputs,
            expected_ops={acc_ops.chunk},
        )

if __name__ == '__main__':
    run_tests()
