from pt import configs

import operator_benchmark as op_bench
import torch
import torch.ao.nn.quantized as nnq


"""
Microbenchmarks for qConv operators.
"""


class QConv1dBenchmark(op_bench.TorchBenchmarkBase):
    # def init(self, N, IC, OC, L, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, L, device):
        G = 1
        pad = 0
        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, L, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv1d = nnq.Conv1d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv1d.set_weight_bias(self.qW, None)
        self.qconv1d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv1d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv1d")

    def forward(self, input):
        return self.qconv1d(input)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    # def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # super().init(N, IC, OC, (H, W), G, (kernel, kernel), stride, pad)

        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.set_weight_bias(self.qW, None)
        self.qconv2d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv2d")

    def forward(self, input):
        return self.qconv2d(input)


op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    QConv1dBenchmark,
)
op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    QConv2dBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
