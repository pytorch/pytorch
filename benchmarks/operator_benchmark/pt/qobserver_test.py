
import operator_benchmark as op_bench
import torch
import torch.quantization.observer as obs

qobserver_short_configs_dict = {
    'attr_names': ('C', 'M', 'N', 'dtype', 'device'),
    'attrs': (
        (3, 512, 512, torch.quint8, 'cpu'),
        (3, 512, 512, torch.quint8, 'cuda'),
    ),
    'tags': ('short',),
}

q_hist_observer_short_configs_dict = {
    'attr_names': ('C', 'M', 'N', 'dtype', 'device'),
    'attrs': (
        (3, 512, 512, torch.quint8, 'cpu'),
    ),
    'tags': ('short',),
}

qobserver_long_configs_dict = {
    'C': (32, 64),
    'M': (256, 1024),
    'N': (256, 1024),
    'device': ('cpu', 'cuda'),
    'dtype': (torch.quint8,),  # dtype doesn't change the timing, keep the same
    'tags': ('long',),
}

q_hist_observer_long_configs_dict = {
    'C': (1, 3, 8),
    'M': (256, 1024),
    'N': (256, 1024),
    'device': ('cpu',),
    'dtype': (torch.quint8,),  # dtype doesn't change the timing, keep the same
    'tags': ('long',),
}


qobserver_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        'qscheme': (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **qobserver_short_configs_dict,
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

q_hist_observer_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        'qscheme': (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **q_hist_observer_short_configs_dict,
)

q_hist_observer_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **q_hist_observer_long_configs_dict,
)


qobserver_per_tensor_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['MinMaxObserver', obs.MinMaxObserver],
        ['MovingAverageMinMaxObserver', obs.MovingAverageMinMaxObserver],
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

q_hist_observer_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['HistogramObserver', obs.HistogramObserver],
        ['HistogramObserverCalculateQparams', obs.HistogramObserver],
    ]
)


class QObserverBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        self.inputs = {
            "f_input": torch.rand(C, M, N, device=device)
        }
        self.op_func = op_func(dtype=dtype, qscheme=qscheme).to(device)

    def forward(self, f_input):
        self.op_func(f_input)
        return self.op_func.calculate_qparams()


class QObserverBenchmarkCalculateQparams(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        self.f_input = torch.rand(C, M, N, device=device)
        self.q_observer = op_func(dtype=dtype, qscheme=qscheme).to(device)
        self.q_observer(self.f_input)
        self.inputs = {}

    def forward(self):
        return self.q_observer.calculate_qparams()


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_tensor_list,
    qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long,
    QObserverBenchmark)

op_bench.generate_pt_tests_from_op_list(
    qobserver_per_channel_list,
    qobserver_per_channel_configs_short + qobserver_per_channel_configs_long,
    QObserverBenchmark)

op_bench.generate_pt_tests_from_op_list(
    q_hist_observer_list,
    q_hist_observer_per_tensor_configs_short + q_hist_observer_per_tensor_configs_long,
    QObserverBenchmarkCalculateQparams)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
