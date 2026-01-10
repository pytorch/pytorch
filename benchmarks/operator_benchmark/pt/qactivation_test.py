import operator_benchmark as op_bench
import torch
import torch.ao.nn.quantized.functional as qF


r"""Microbenchmarks for the quantized activations."""

qactivation_long_configs = op_bench.cross_product_configs(
    dims=(
        # VGG-16 relu's with original shape: (-1, 3, 224, 224)
        (64, 224, 224),  # ReLU-1   # noqa: E201
        (128, 112, 112),  # ReLU-6
        (256, 56, 56),  # ReLU-11  # noqa: E241
        (512, 28, 28),  # ReLU-18  # noqa: E241
        (512, 14, 14),  # ReLU-25  # noqa: E241
        # Batch = 16
        (16, 64, 224, 224),  # ReLU-1   # noqa: E241
        (16, 128, 112, 112),  # ReLU-6
        (16, 256, 56, 56),  # ReLU-11  # noqa: E241
        (16, 512, 28, 28),  # ReLU-18  # noqa: E241
        (16, 512, 14, 14),  # ReLU-25  # noqa: E241
    ),
    contig=(False, True),
    inplace=(False, True),
    dtype=(torch.quint8,),
    tags=("long",),
)

qactivation_short_configs = op_bench.cross_product_configs(
    dims=(
        (3, 4, 5),  # Rank=3
        (2, 3, 4, 5),  # Rank=4,
        # Dimensions from the floating point benchmarks
        (512, 512),
        (256, 1024),
    ),
    contig=(False,),
    inplace=(False,),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=("short",),
)

qactivation_ops = op_bench.op_list(
    attrs=(
        ("relu", torch.nn.ReLU()),
        ("relu6", torch.ops.quantized.relu6),
        ("functional.hardtanh", qF.hardtanh),
        ("functional.hardsigmoid", qF.hardsigmoid),
        ("functional.leaky_relu", qF.leaky_relu),
        ("functional.sigmoid", torch.nn.functional.sigmoid),
        ("functional.tanh", torch.nn.functional.tanh),
    ),
    attr_names=("op_name", "op_func"),
)


class QActivationBenchmarkBase(op_bench.TorchBenchmarkBase):
    r"""Base class for all the activations."""

    def _setup(self, dims, contig, dtype):
        # Input
        f_input = (torch.rand(*dims) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0

        # Quantize the tensor
        q_input = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )
        if not contig:
            # Make non-contiguous
            new_shape = list(range(q_input.ndim))[::-1]
            q_input = q_input.permute(new_shape)

        self.inputs = {"q_input": q_input}

    def init(self, dims, contig, inplace, dtype, op_func):
        self._setup(dims, contig, dtype)
        self.qop = op_func


class QActivationBenchmark(QActivationBenchmarkBase):
    def forward(self, q_input):
        return self.qop(q_input)


op_bench.generate_pt_tests_from_op_list(
    qactivation_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationBenchmark,
)


qactivation_scale_zero_point_ops = op_bench.op_list(
    attrs=(
        ("functional.hardswish", qF.hardswish),
        ("functional.elu", qF.elu),
        ("functional.celu", qF.celu),
    ),
    attr_names=("op_name", "op_func"),
)


class QActivationScaleZeroPointBenchmark(QActivationBenchmarkBase):
    def forward(self, q_input):
        return self.qop(q_input, scale=self.scale, zero_point=self.zero_point)


op_bench.generate_pt_tests_from_op_list(
    qactivation_scale_zero_point_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationScaleZeroPointBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
