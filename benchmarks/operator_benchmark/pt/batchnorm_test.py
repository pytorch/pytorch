import torch
import torch.nn.functional as F

import operator_benchmark as op_bench


"""Microbenchmarks for batchnorm operator."""

# Benchmark cudnn if available
if torch.backends.cudnn.is_available:

    def cudnn_benchmark_configs(configs):
        result = []
        for config in configs:
            is_cuda = any("cuda" in attr.values() for attr in config)
            if is_cuda:
                result.append((*config, dict(cudnn=True)))
            result.append((*config, dict(cudnn=False)))
        return result

else:

    def cudnn_benchmark_configs(configs):
        return [(*config, dict(cudnn=False)) for config in configs]


batchnorm_configs_short = cudnn_benchmark_configs(
    op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=[
            [1, 256, 3136],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],
            "training": [True, False],
        },
        tags=["short"],
    )
)

batchnorm_configs_long = cudnn_benchmark_configs(
    op_bench.cross_product_configs(
        M=[2, 128],
        N=[8192, 2048],
        K=[1],
        device=["cpu", "cuda"],
        training=[True, False],
        tags=["long"],
    )
)


class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, training, cudnn):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "mean": torch.rand(N, device=device),
            "var": torch.rand(N, device=device),
            "weight": torch.rand(N, device=device),
            "bias": torch.rand(N, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        self.set_module_name("batchnorm")

    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        with torch.backends.cudnn.flags(enabled=cudnn):
            return F.batch_norm(input_one, mean, var, weight, bias, training)


op_bench.generate_pt_test(
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)
op_bench.generate_pt_gradient_test(
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)


batchnorm1d_configs_short = cudnn_benchmark_configs(
    op_bench.config_list(
        attr_names=["N", "C"],
        attrs=[
            [3136, 256],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],
            "training": [True, False],
        },
        tags=["short"],
    )
)

batchnorm1d_configs_long = cudnn_benchmark_configs(
    op_bench.cross_product_configs(
        N=[2, 128],
        C=[8192, 2048],
        device=["cpu", "cuda"],
        training=[True, False],
        tags=["long"],
    )
)


class BatchNorm1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, device, training, cudnn):
        self.inputs = {
            "input_one": torch.rand(N, C, device=device, requires_grad=self.auto_set()),
            "mean": torch.rand(C, device=device),
            "var": torch.rand(C, device=device),
            "weight": torch.rand(C, device=device),
            "bias": torch.rand(C, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        self.set_module_name("batchnorm")

    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        with torch.backends.cudnn.flags(enabled=cudnn):
            return F.batch_norm(input_one, mean, var, weight, bias, training)


op_bench.generate_pt_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)
op_bench.generate_pt_gradient_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
