from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
from torch import nn

from .utils import quantize

"""
Microbenchmarks for RNNs.
"""

qrnn_dynamic_configs = op_bench.config_list(
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

qrnn_static_configs = op_bench.config_list(
    attrs=[
        [1, 3, 4],
        [5, 7, 8],
    ],
    # names: input_size, hidden_size, num_layers
    attr_names=["I", "H", "NL"],
    cross_product_configs={
        "B": (True,),               # Bias always True for quantized
        "D": (False, True),         # Bidirectional
    },
    tags=["short"]
)

class DynamicLSTMBenchmark(op_bench.TorchBenchmarkBase):
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
                             I)             # Number of featues in X
        self.h = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size
        self.c = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size

        self.set_module_name("qLSTMDynamic")

    def forward(self):
        return self.cell(self.x, (self.h, self.c))


class StaticLSTMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, I, H, NL, B, D):
        sequence_len = 128
        batch_size = 16

        cell_nn = nn.LSTM(
            input_size=I,
            hidden_size=H,
            num_layers=NL,
            bias=B,
            batch_first=False,
            dropout=0.0,
            bidirectional=D,
        )
        flat_weights_names = cell_nn._flat_weights_names
        flat_weights = cell_nn._flat_weights
        self.cell = nn.quantized.LSTM(
            input_size=I,
            hidden_size=H,
            num_layers=NL,
            bias=B,
            batch_first=False,
            dropout=0.0,
            bidirectional=D,
            # Quantization params
            flat_weights_names=flat_weights_names,
            flat_weights=flat_weights,
            weights_scale=None,
            weights_zero_point=None
        )

        x = torch.randn(sequence_len,  # sequence length
                        batch_size,    # batch size
                        I)             # Number of featues in X
        h = torch.randn(NL * (D + 1),  # layer_num * dir_num
                        batch_size,    # batch size
                        H)             # hidden size
        c = torch.randn(NL * (D + 1),  # layer_num * dir_num
                        batch_size,    # batch size
                        H)             # hidden size

        self.qx = quantize(x, torch.quint8)
        self.qh = quantize(h, torch.quint8)
        self.qc = quantize(c, torch.qint32, scale=1e-6, zero_point=0)

        self.set_module_name("qLSTMStatic")

    def forward(self):
        return self.cell(self.qx, (self.qh, self.qc))


class LSTMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, I, H, NL, B, D):
        sequence_len = 128
        batch_size = 16

        self.cell_nn = nn.LSTM(
            input_size=I,
            hidden_size=H,
            num_layers=NL,
            bias=B,
            batch_first=False,
            dropout=0.0,
            bidirectional=D,
        )

        self.x = torch.randn(sequence_len,  # sequence length
                             batch_size,    # batch size
                             I)             # Number of featues in X
        self.h = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size
        self.c = torch.randn(NL * (D + 1),  # layer_num * dir_num
                             batch_size,    # batch size
                             H)             # hidden size

        self.set_module_name("LSTM")

    def forward(self):
        return self.cell_nn(self.x, (self.h, self.c))

op_bench.generate_pt_test(qrnn_dynamic_configs, DynamicLSTMBenchmark)
op_bench.generate_pt_test(qrnn_static_configs, StaticLSTMBenchmark)
op_bench.generate_pt_test(qrnn_static_configs, LSTMBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
