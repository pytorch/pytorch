from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.quantization.observer as obs


observer_configs_short = op_bench.config_list(
    # mode is used to show the direction of the benchmark:
    # if 'Q', benchmark quantization, else dequantization
    attr_names=['C', 'M', 'N', 'dtype', 'qscheme'],
    attrs=[
        [3, 512, 512, torch.quint8, torch.per_tensor_affine],
    ],
    tags=['short']
)

observer_configs_long = op_bench.cross_product_configs(
    C=[1, 3, 8],
    M=[256, 1024],
    N=[256, 1024],
    dtype=[torch.quint8],  # dtype doesn't change the timing
    qscheme=[torch.per_tensor_affine, torch.per_tensor_symmetric,
             torch.per_channel_affine, torch.per_channel_symmetric],
    tags=['long']
)

observer_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['MinMaxObserver', obs.MinMaxObserver],
        ['MovingAverageMinMaxObserver', obs.MovingAverageMinMaxObserver],
        ['PerChannelMinMaxObserver', obs.PerChannelMinMaxObserver],
        ['MovingAveragePerChannelMinMaxObserver', obs.MovingAveragePerChannelMinMaxObserver],
        ['HistogramObserver', obs.HistogramObserver],
    ]
)


class ObserverBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func):
        self.f_input = torch.rand(C, M, N)
        self.op_func = op_func(dtype=dtype, qscheme=qscheme)

    def forward(self):
        return self.op_func(self.f_input)


op_bench.generate_pt_tests_from_op_list(
    observer_list,
    observer_configs_short + observer_configs_long,
    ObserverBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
