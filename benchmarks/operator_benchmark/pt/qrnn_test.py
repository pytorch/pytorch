from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.quantized.dynamic as nnqd

"""
Microbenchmarks for RNNs.
"""

qrnn_configs = op_bench.config_list(
    attrs=[
        [1, 3, 1],
        [5, 7, 4],
    ],
    # names: input_size, hiddent_size, num_layers
    attr_names=["I", "H", "NL"],
    cross_product_configs={
        "B": (False, True),  # Bias
        "D": (False, True),  # Bidirectional
        "dtype": (torch.qint8, torch.qint32)  # dtype
    },
    tags=["short"]
)

class _RNNBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, input_size, hidden_size, dtype):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, input_size, dtype=torch.float32)
        h = torch.randn(N, hidden_size, dtype=torch.float32)
        self.qX = torch.quantize_per_tensor(X, scale=scale,
                                            zero_point=zero_point, dtype=dtype)
        self.qH = torch.quantize_per_tensor(h, scale=scale,
                                            zero_point=zero_point, dtype=dtype)

    def forward(self):
        # Assume the child sets `self.cell`
        return self.cell(self.qX, self.qH)


class LSTMBenchmark(_RNNBenchmarkBase):
    def init(self, I, H, NL, B, D, dtype):
        super(LSTMBenchmark, self).init(I, H, dtype)
        self.cell = nnqd.LSTM(
            input_size=I,
            hidden_size=H,
            num_layers=NL,
            bias=B,
            batch_first=False,
            dropout=0.0,
            bidirectional=D,
            dtype=dtype
        )
        self.set_module_name("QLSTM")

op_bench.generate_pt_test(qrnn_configs, LSTMBenchmark)
