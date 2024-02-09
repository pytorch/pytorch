import torch

import operator_benchmark as op_bench


"""Microbenchmarks for quantized batchnorm operator."""

batchnorm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 256, 3136],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": (torch.qint8,),
    },
    tags=["short"],
)


class QBatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self._init(M, N, K, device)
        x_scale = 0.1
        x_zero_point = 0
        self.inputs = {
            "q_input_one": torch.quantize_per_tensor(
                self.input_one, scale=x_scale, zero_point=x_zero_point, dtype=dtype
            ),
            "mean": torch.rand(N),
            "var": torch.rand(N),
            "weight": torch.rand(N),
            "bias": torch.rand(N),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def _init(self, M, N, K, device):
        pass

    def forward(self):
        pass


class QBatchNorm1dBenchmark(QBatchNormBenchmark):
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm1d")
        self.input_one = torch.rand(
            M, N, K, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm1d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


class QBatchNorm2dBenchmark(QBatchNormBenchmark):
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm2d")
        # Note: quantized implementation requires rank 4, which is why we
        # add a 1 as the last dimension
        self.input_one = torch.rand(
            M, N, K, 1, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm2d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm1dBenchmark)
op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm2dBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
