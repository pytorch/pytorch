from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
from torch import nn

"""
Microbenchmarks for RNNs.
"""

qrnn_configs = op_bench.config_list(
    attrs=[
        [1, 3, 1],
        [5, 7, 4],
    ],
    # names: input_size, hidden_size, num_layers
    attr_names=["I", "H", "NL"],
    cross_product_configs={
        "B": (True,),               # Bias always True for quantized
        "D": (False, True),         # Bidirectional
        "dtype": (torch.qint8,)     # Only qint8 dtype works for now
    },
    tags=["short"]
)

class LSTMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, I, H, NL, B, D, dtype):
        sequence_len = 128
        batch_size = 16

        # The quantized.dynamic.LSTM has a bug. That's why we create a regular
        # LSTM, and quantize it later. See issue #31192.
        scale = 1.0 / 256
        zero_point = 0
        cell_nn = nn.LSTM(
            input_size=I,
            hidden_size=H,
            num_layers=NL,
            bias=B,
            batch_first=False,
            dropout=0.0,
            bidirectional=D,
        )
        cell_temp = nn.Sequential(cell_nn)
        self.cell = torch.quantization.quantize_dynamic(cell_temp,
                                                        {nn.LSTM, nn.Linear},
                                                        dtype=dtype)[0]

        self.x = torch.randn(sequence_len,  # sequence length
                             batch_size,    # batch size
                             I)             # Number of features in X
        self.h = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size
        self.c = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size

        self.set_module_name("QLSTM")

    def forward(self):
        return self.cell(self.x, (self.h, self.c))

op_bench.generate_pt_test(qrnn_configs, LSTMBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
