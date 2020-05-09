from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.quantization.observer as obs

qobserver_short_configs_dict = {
    'attr_names': ('C', 'M', 'N', 'dtype'),
    'attrs': (
        (3, 512, 512, torch.quint8),
    ),
    'tags': ('short',),
}

qobserver_long_configs_dict = {
    'C': (1, 3, 8),
    'M': (256, 1024),
    'N': (256, 1024),
    'dtype': (torch.quint8,),  # dtype doesn't change the timing, keep the same
    'tags': ('long',),
}


qobserver_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        'qscheme': (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **qobserver_short_configs_dict,  # noqa
)

qobserver_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **qobserver_long_configs_dict,
)

qobserver_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={
        'qscheme': (torch.per_channel_affine, torch.per_channel_symmetric)
    },
    **qobserver_short_configs_dict,
)

qobserver_per_channel_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_channel_affine, torch.per_channel_symmetric),
    **qobserver_long_configs_dict,
)


qobserver_per_tensor_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['MinMaxObserver', obs.MinMaxObserver],
        ['MovingAverageMinMaxObserver', obs.MovingAverageMinMaxObserver],
        ['HistogramObserver', obs.HistogramObserver],
    ]
)

qobserver_per_channel_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['PerChannelMinMaxObserver', obs.PerChannelMinMaxObserver],
        ['MovingAveragePerChannelMinMaxObserver',
         obs.MovingAveragePerChannelMinMaxObserver],
    ]
)


class QObserverBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func):
        self.f_input = torch.rand(C, M, N)
        self.op_func = op_func(dtype=dtype, qscheme=qscheme)

    def forward(self):
        return self.op_func(self.f_input)


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_tensor_list,
    qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long,
    QObserverBenchmark)

op_bench.generate_pt_tests_from_op_list(
    qobserver_per_channel_list,
    qobserver_per_channel_configs_short + qobserver_per_channel_configs_long,
    QObserverBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
