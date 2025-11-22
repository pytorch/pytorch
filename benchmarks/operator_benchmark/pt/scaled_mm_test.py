from pt import configs

import operator_benchmark as op_bench

import torch
from torch.nn.functional import ScalingType


"""
Microbenchmarks for scaled_mm and scaled_grouped_mm operators.
"""


# Helper function to generate scales
def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype):
    """Convert tensor to FP8 scale."""
    amax = torch.max(torch.abs(x))
    e4m3_max = 448.0  # E4M3_MAX_POS
    scale = e4m3_max / torch.clamp(amax, min=1e-12)
    return scale.float()


class ScaledMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.float8_dtype = torch.float8_e4m3fn
        self.base_dtype = torch.bfloat16
        
        # Create random inputs
        x = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y = torch.randn(N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()).t()
        
        # Compute scales
        x_scale = tensor_to_scale(x, self.float8_dtype)
        y_scale = tensor_to_scale(y, self.float8_dtype)
        
        # Convert to FP8
        x_fp8 = (x * x_scale).to(self.float8_dtype)
        y_fp8 = (y * y_scale).to(self.float8_dtype)
        
        self.inputs = {
            "x": x_fp8,
            "y": y_fp8,
            "scale_a": x_scale.reciprocal(),
            "scale_b": y_scale.reciprocal(),
        }
        self.set_module_name("scaled_mm")

    def forward(self, x, y, scale_a, scale_b):
        return torch.nn.functional.scaled_mm(
            x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
        )


class ScaledGroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, G, device):
        self.float8_dtype = torch.float8_e4m3fn
        self.base_dtype = torch.bfloat16
        
        # Create random inputs for grouped matmul
        x = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y = torch.randn(G, N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()).transpose(-2, -1)
        
        # Compute scales
        x_scale = tensor_to_scale(x, self.float8_dtype)
        y_scale = tensor_to_scale(y, self.float8_dtype)
        
        # Convert to FP8
        x_fp8 = (x * x_scale).to(self.float8_dtype)
        y_fp8 = (y * y_scale).to(self.float8_dtype)
        
        # Generate jagged offsets for grouped matmul
        # Simple uniform distribution of tokens across groups
        group_size = M // G
        offs = torch.tensor([i * group_size for i in range(G + 1)], device=device)
        offs[-1] = M  # Ensure last offset is exactly M
        
        self.inputs = {
            "x": x_fp8,
            "y": y_fp8,
            "offs": offs,
            "scale_a": x_scale.reciprocal(),
            "scale_b": y_scale.reciprocal(),
        }
        self.set_module_name("scaled_grouped_mm")

    def forward(self, x, y, offs, scale_a, scale_b):
        return torch.nn.functional.scaled_grouped_mm(
            x,
            y,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TENSOR,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TENSOR,
            offs=offs,
            output_dtype=torch.bfloat16,
        )


# Configs for scaled_mm - short configs include both DSv3 671B and Llama4
scaled_mm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [16384, 2048, 7168],   # DSv3 671B - small batch
        [16384, 8192, 5120],   # Llama4 16e - small batch
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["short"],
)

# Configs for scaled_mm - long configs include both models with more variations
scaled_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [16384, 2048, 7168],   # DSv3 671B - small batch
        [128000, 2048, 7168],  # DSv3 671B - large batch
        [16384, 8192, 5120],   # Llama4 16e - small batch
        [128000, 8192, 5120],  # Llama4 16e - large batch
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["long"],
)

# Configs for scaled_grouped_mm - short configs include both models
scaled_grouped_mm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [16384, 2048, 7168, 1],   # DSv3 671B - 1 expert per device
        [16384, 2048, 7168, 4],   # DSv3 671B - 4 experts per device
        [16384, 8192, 5120, 1],   # Llama4 16e - 1 expert per device
        [16384, 8192, 5120, 4],   # Llama4 16e - 4 experts per device
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["short"],
)

# Configs for scaled_grouped_mm - long configs include both models with all expert counts
scaled_grouped_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [16384, 2048, 7168, 1],   # DSv3 671B - 1 expert per device
        [16384, 2048, 7168, 2],   # DSv3 671B - 2 experts per device
        [16384, 2048, 7168, 4],   # DSv3 671B - 4 experts per device
        [16384, 2048, 7168, 8],   # DSv3 671B - 8 experts per device
        [128000, 2048, 7168, 1],  # DSv3 671B - large batch, 1 expert
        [128000, 2048, 7168, 2],  # DSv3 671B - large batch, 2 experts
        [128000, 2048, 7168, 4],  # DSv3 671B - large batch, 4 experts
        [128000, 2048, 7168, 8],  # DSv3 671B - large batch, 8 experts
        [16384, 8192, 5120, 1],   # Llama4 16e - 1 expert per device
        [16384, 8192, 5120, 2],   # Llama4 16e - 2 experts per device
        [16384, 8192, 5120, 4],   # Llama4 16e - 4 experts per device
        [16384, 8192, 5120, 8],   # Llama4 16e - 8 experts per device
        [128000, 8192, 5120, 1],  # Llama4 16e - large batch, 1 expert
        [128000, 8192, 5120, 2],  # Llama4 16e - large batch, 2 experts
        [128000, 8192, 5120, 4],  # Llama4 16e - large batch, 4 experts
        [128000, 8192, 5120, 8],  # Llama4 16e - large batch, 8 experts
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["long"],
)

# Generate tests for scaled_mm
op_bench.generate_pt_test(
    scaled_mm_configs_short + scaled_mm_configs_long, ScaledMMBenchmark
)
op_bench.generate_pt_gradient_test(
    scaled_mm_configs_long, ScaledMMBenchmark
)

# Generate tests for scaled_grouped_mm
op_bench.generate_pt_test(
    scaled_grouped_mm_configs_short + scaled_grouped_mm_configs_long,
    ScaledGroupedMMBenchmark,
)
op_bench.generate_pt_gradient_test(
    scaled_grouped_mm_configs_long, ScaledGroupedMMBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

