import torch
import torch.ao.nn.quantized as nnq
import torch.ao.quantization as tq
import torch.nn as nn

import operator_benchmark as op_bench

"""Microbenchmarks for general quantization operations."""

# mode is used to show the direction of the benchmark:
# if 'Q', benchmark quantization, else dequantization

quantize_configs_short_dict = {
    "attr_names": ["C", "M", "N", "dtype", "mode"],
    "attrs": [
        [3, 512, 512, torch.quint8, "Q"],
        [3, 512, 512, torch.quint8, "D"],
    ],
    "tags": ["short"],
}

quantize_configs_long_dict = {
    "C": [3, 5, 8],  # this is reused for per-channel: avoid single channel test
    "M": [256, 1024],
    "N": [256, 1024],
    "dtype": [torch.quint8, torch.qint8, torch.qint32],
    "mode": ["D", "Q"],
    "tags": ["long"],
}


quantize_per_tensor_configs_short = op_bench.config_list(**quantize_configs_short_dict)

quantize_per_tensor_configs_long = op_bench.cross_product_configs(
    **quantize_configs_long_dict
)


class QuantizePerTensorBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""

    def init(self, C, M, N, dtype, mode):
        assert mode in ("Q", "D")
        self.input = torch.rand(C, M, N)
        self.dtype = dtype
        self.op = nnq.Quantize(scale=1.0, zero_point=0, dtype=dtype)
        self.set_module_name("QuantizePerTensor")

        if mode == "D":
            self.input = self.op(self.input)
            self.op = nnq.DeQuantize()
            self.set_module_name("DequantizePerTensor")

        self.inputs = {"input": self.input}

    def forward(self, input):
        return self.op(input)


op_bench.generate_pt_test(
    quantize_per_tensor_configs_short + quantize_per_tensor_configs_long,
    QuantizePerTensorBenchmark,
)

# === Per Channel quantization ===

quantize_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={"axis": (0,)}, **quantize_configs_short_dict
)

quantize_per_channel_configs_long = op_bench.cross_product_configs(
    axis=(0, 1, 2), **quantize_configs_long_dict
)


class QuantizePerChannelBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""

    def init(self, C, M, N, dtype, axis, mode):
        assert mode in ("Q", "D")
        self.input = torch.rand(C, M, N)
        self.op = torch.quantize_per_channel

        channel_len = (C, M, N)[axis]

        self.kwargs = {
            "scales": torch.tensor([1.0] * channel_len),
            "zero_points": torch.tensor([0] * channel_len),
            "dtype": dtype,
            "axis": axis,
        }

        self.set_module_name("QuantizePerChannel")

        if mode == "D":
            self.input = self.op(self.input, **self.kwargs)

            def dequant(input, scales, zero_points, axis: int, dtype: int):
                return input.dequantize()

            self.op = dequant
            self.set_module_name("DequantizePerChannel")

        self.inputs = {
            "input": self.input,
            "scales": torch.tensor([1.0] * channel_len),
            "zero_points": torch.tensor([0] * channel_len),
            "axis": axis,
            "dtype": dtype,
        }

    def forward(self, input, scales, zero_points, axis: int, dtype: int):
        return self.op(
            input, scales=scales, zero_points=zero_points, axis=axis, dtype=dtype
        )


op_bench.generate_pt_test(
    quantize_per_channel_configs_short + quantize_per_channel_configs_long,
    QuantizePerChannelBenchmark,
)

# === Fake Quantization ===
# Generated benchmarks names start with 'learnable_kernel' or 'original_kernel',
#    for ex. 'original_kernel_nbits8_cpu_N1_C1_H256_W256_zero_point_dtypetorch.int32_bwdall'

fake_quantize_configs_short_dict = {
    "attr_names": ["N", "C", "H", "W", "zero_point_dtype"],
    "attrs": [
        [1, 3, 512, 512, torch.int32],
    ],
    "tags": ["short"],
}

fake_quantize_configs_long_dict = {
    "N": [1],
    "C": [1, 3, 8, 32],
    "H": [256, 1024],
    "W": [256, 1024],
    "zero_point_dtype": [torch.int32],
    "tags": ["long"],
}

fake_quantize_configs_short = op_bench.config_list(
    cross_product_configs={
        "device": ("cpu", "cuda"),
    },
    **fake_quantize_configs_short_dict,
)

fake_quantize_configs_long = op_bench.cross_product_configs(
    device=("cpu", "cuda"), **fake_quantize_configs_long_dict
)


class FakeQuantizeBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks fake quantization with default parameters."""

    def init(self, N, C, H, W, zero_point_dtype, device):
        self.inputs = {"input": torch.rand(N, C, H, W).to(device)}
        self.op = tq.FakeQuantize().to(device)
        self.set_module_name("FakeQuantize")

    def forward(self, input):
        return self.op(input)


op_bench.generate_pt_test(
    fake_quantize_configs_short + fake_quantize_configs_long, FakeQuantizeBenchmark
)


# op_type is used to describe the type of operator used in benchmarking:
# learnable_kernel represents the c++ kernel that can backpropagate on
# scale and zero point.
# original_kernel represents the original fake quantize c++ kernel.


def fakeQuantizePerTensorLearnableKernel(
    input, scale, zero_point, quant_min: int, quant_max: int
):
    return torch._fake_quantize_learnable_per_tensor_affine(
        input, scale, zero_point, quant_min, quant_max
    )


def fakeQuantizePerTensorOriginalKernel(
    input, scale, zero_point, quant_min: int, quant_max: int
):
    return torch.fake_quantize_per_tensor_affine(input, 1.0, 0, quant_min, quant_max)


fake_quantize_per_tensor_ops = op_bench.op_list(
    attrs=(
        ("learnable_kernel", fakeQuantizePerTensorLearnableKernel),
        ("original_kernel", fakeQuantizePerTensorOriginalKernel),
    ),
    attr_names=("op_name", "op_func"),
)

fake_quantize_operator_configs_short = op_bench.config_list(
    cross_product_configs={
        "nbits": (4, 8),
        "device": ("cpu", "cuda"),
    },
    **fake_quantize_configs_short_dict,
)

fake_quantize_operator_configs_long = op_bench.cross_product_configs(
    nbits=(4, 8), device=("cpu", "cuda"), **fake_quantize_configs_long_dict
)

# TODO(future PR) Combine config for floating point zero_point with other configs, once it is
# fully supported in all fakeQuant operators and devices for
# https://github.com/pytorch/pytorch/issues/61866.
fake_quantize_configs_long_dict_float_zero_point = (
    fake_quantize_configs_long_dict.copy()
)
fake_quantize_configs_long_dict_float_zero_point["zero_point_dtype"] = [
    torch.float32,
    torch.half,
]

fake_quantize_operator_configs_long_float_zero_point = op_bench.cross_product_configs(
    nbits=(8,),
    device=("cpu", "cuda"),
    **fake_quantize_configs_long_dict_float_zero_point,
)


class FakeQuantizePerTensorBaseOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per tensor operators."""

    def init(self, N, C, H, W, zero_point_dtype, nbits, device, op_func):
        self.quant_min = 0
        self.quant_max = 2**nbits - 1
        self.quant_range = 2**nbits
        self.input = nn.Parameter(
            torch.rand(N, C, H, W, dtype=torch.float, device=device),
            requires_grad=self.auto_set(),
        )
        self.scale = nn.Parameter(
            torch.tensor([1.0]).to(device), requires_grad=self.auto_set()
        )
        if op_func.__name__ == "fakeQuantizePerChannelOriginalKernel":
            self.zero_point = nn.Parameter(
                torch.tensor([0.0]).to(device).to(zero_point_dtype),
                requires_grad=self.auto_set(),
            )
        else:
            self.zero_point = nn.Parameter(
                torch.tensor([0.0]).to(device), requires_grad=self.auto_set()
            )

        self.inputs = {
            "input": self.input,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
        }
        self.op_func = op_func

    def forward(self, input, scale, zero_point, quant_min: int, quant_max: int):
        return self.op_func(input, scale, zero_point, quant_min, quant_max)


op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_tensor_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorBaseOpBenchmark,
)

op_bench.generate_pt_gradient_tests_from_op_list(
    fake_quantize_per_tensor_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorBaseOpBenchmark,
)


def fakeQuantizePerChannelLearnableKernel(
    input, scale, zero_point, axis: int, quant_min: int, quant_max: int
):
    return torch._fake_quantize_learnable_per_channel_affine(
        input, scale, zero_point, axis, quant_min, quant_max
    )


def fakeQuantizePerChannelOriginalKernel(
    input, scale, zero_point, axis: int, quant_min: int, quant_max: int
):
    return torch.fake_quantize_per_channel_affine(
        input, scale, zero_point, axis, quant_min, quant_max
    )


fake_quantize_per_channel_ops = op_bench.op_list(
    attrs=(
        ("learnable_kernel", fakeQuantizePerChannelLearnableKernel),
        ("original_kernel", fakeQuantizePerChannelOriginalKernel),
    ),
    attr_names=("op_name", "op_func"),
)

fake_quantize_per_channel_float_zero_point_ops = op_bench.op_list(
    attrs=(("original_kernel", fakeQuantizePerChannelOriginalKernel),),
    attr_names=("op_name", "op_func"),
)


class FakeQuantizePerChannelOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per channel operators."""

    def init(self, N, C, H, W, zero_point_dtype, nbits, device, op_func):
        self.quant_min = 0
        self.quant_max = 2**nbits - 1
        self.quant_range = 2**nbits
        # Axis is chosen with respect to the number of channels: C.
        self.axis = 1
        self.input = nn.Parameter(
            torch.rand(
                N,
                C,
                H,
                W,
                dtype=torch.float,
                device=device,
                requires_grad=self.auto_set(),
            )
        )

        if op_func.__name__ == "fakeQuantizePerChannelOriginalKernel":
            self.scale = torch.ones(
                C, device=device, dtype=torch.float32, requires_grad=False
            )
            self.zero_point = torch.zeros(
                C, device=device, dtype=zero_point_dtype, requires_grad=False
            )
        else:
            self.scale = nn.Parameter(
                torch.ones(C, device=device, dtype=torch.float32),
                requires_grad=self.auto_set(),
            )
            self.zero_point = nn.Parameter(
                torch.zeros(C, device=device, dtype=torch.float32),
                requires_grad=self.auto_set(),
            )

        self.inputs = {
            "input": self.input,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "axis": self.axis,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
        }

        self.op_func = op_func

    def forward(
        self, input, scale, zero_point, axis: int, quant_min: int, quant_max: int
    ):
        return self.op_func(input, scale, zero_point, axis, quant_min, quant_max)


op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_channel_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    fake_quantize_per_channel_float_zero_point_ops,
    fake_quantize_operator_configs_long_float_zero_point,
    FakeQuantizePerChannelOpBenchmark,
)

op_bench.generate_pt_gradient_tests_from_op_list(
    fake_quantize_per_channel_ops,
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
