from pt import configs

import operator_benchmark as op_bench

import torch
from torch.nn.functional import ScalingType


"""
Microbenchmarks for scaled_mm and scaled_grouped_mm operators.
"""


# FP8 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0
EPS = 1e-12


def _get_float8_dtype(float8_dtype):
    """Convert string or torch dtype to torch.float8 dtype."""
    if float8_dtype == "e4m3fn" or float8_dtype == torch.float8_e4m3fn:
        return torch.float8_e4m3fn
    elif float8_dtype == "e5m2" or float8_dtype == torch.float8_e5m2:
        return torch.float8_e5m2
    else:
        return torch.float8_e4m3fn  # default

# Helper function to generate scales
def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype):
    """Convert tensor to FP8 scale."""
    amax = torch.max(torch.abs(x))
    if float8_dtype == torch.float8_e4m3fn:
        max_pos = E4M3_MAX_POS
    elif float8_dtype == torch.float8_e5m2:
        max_pos = E5M2_MAX_POS
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    scale = max_pos / torch.clamp(amax, min=EPS)
    return scale.float()


class ScaledMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, float8_dtype="e4m3fn", output_dtype="bfloat16"):
        self.float8_dtype = _get_float8_dtype(float8_dtype)
        self.base_dtype = torch.bfloat16

        # Convert output_dtype string to torch dtype
        if output_dtype == "bfloat16":
            self.output_dtype = torch.bfloat16
        elif output_dtype == "float32":
            self.output_dtype = torch.float32
        else:
            self.output_dtype = torch.bfloat16  # default

        # Create random inputs
        # For gradient tests, we need requires_grad on the base tensors
        x_base = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y_base = torch.randn(N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()).t()

        # Compute scales
        x_scale = tensor_to_scale(x_base, self.float8_dtype)
        y_scale = tensor_to_scale(y_base, self.float8_dtype)

        # Convert to FP8 - create as leaf tensors
        with torch.no_grad():
            x_fp8 = (x_base * x_scale).to(self.float8_dtype).detach().requires_grad_(self.auto_set())
            y_fp8 = (y_base * y_scale).to(self.float8_dtype).detach().requires_grad_(self.auto_set())

        # Scales need to be scalar tensors for TensorWise scaling
        # Remove .contiguous() as it doesn't apply to scalar tensors
        scale_a_scalar = x_scale.reciprocal().item()
        scale_b_scalar = y_scale.reciprocal().item()

        self.inputs = {
            "x": x_fp8,
            "y": y_fp8,
            "scale_a": torch.tensor(scale_a_scalar, device=device, dtype=torch.float32),
            "scale_b": torch.tensor(scale_b_scalar, device=device, dtype=torch.float32),
        }
        self.set_module_name("scaled_mm")

    def forward(self, x, y, scale_a, scale_b):
        return torch.nn.functional.scaled_mm(
            x,
            y,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TensorWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TensorWise,
            output_dtype=self.output_dtype,
        )


class ScaledGroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, G, device, float8_dtype="e4m3fn", output_dtype="bfloat16"):
        self.float8_dtype = _get_float8_dtype(float8_dtype)
        self.base_dtype = torch.bfloat16

        # Convert output_dtype string to torch dtype
        if output_dtype == "bfloat16":
            self.output_dtype = torch.bfloat16
        elif output_dtype == "float32":
            self.output_dtype = torch.float32
        else:
            self.output_dtype = torch.bfloat16  # default

        # Create random inputs for grouped matmul (2D input, 3D weight)
        # Following the 2D-3D pattern from test_scaled_grouped_gemm_2d_3d
        # x is (M*G, K) - 2D input with groups along M dimension
        # y is (G, N, K) - 3D weight
        total_M = M * G
        x_base = torch.randn(total_M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y_base = torch.randn(G, N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())

        # Convert to FP8
        with torch.no_grad():
            x_fp8 = x_base.to(self.float8_dtype).detach().requires_grad_(self.auto_set())
            y_fp8 = y_base.to(self.float8_dtype).detach().requires_grad_(self.auto_set())

        # Generate offsets along M dimension: [M, 2M, ..., M*G]
        offs = torch.arange(M, G * M + 1, M, device=device, dtype=torch.int32)

        # For RowWise scaling:
        # scale_a has one scale per row of x: shape (G*M,)
        # scale_b has one scale per row per group of y: shape (G, N)
        scale_a = torch.rand(G * M, device=device, dtype=torch.float32)
        scale_b = torch.rand(G * N, device=device, dtype=torch.float32).view(G, N)

        self.inputs = {
            "x": x_fp8,
            "y": y_fp8,
            "offs": offs,
            "scale_a": scale_a,
            "scale_b": scale_b,
        }
        self.set_module_name("scaled_grouped_mm")

    def forward(self, x, y, offs, scale_a, scale_b):
        return torch.nn.functional.scaled_grouped_mm(
            x,
            y.transpose(-2, -1),  # Transpose y from (G, N, K) to (G, K, N)
            scale_a=scale_a,
            scale_recipe_a=ScalingType.RowWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.RowWise,
            offs=offs,
            output_dtype=self.output_dtype,
        )


# Configs for scaled_mm - short configs include both DSv3 671B and Llama4
# Note: E5M2 is not supported for matrix multiplication (only E4M3FN)
scaled_mm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [16384, 2048, 7168],   # DSv3 671B - small batch
        [16384, 8192, 5120],   # Llama4 16e - small batch
    ],
    cross_product_configs={
        "device": ["cuda"],
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16", "float32"],
    },
    tags=["short"],
)

# Configs for scaled_mm - long configs include both models with more variations
# Note: E5M2 is not supported for matrix multiplication (only E4M3FN)
# REDUCED SHAPES FOR FASTER TESTING - keeping all dtypes
scaled_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [2048, 2048, 2048],   # DSv3 671B - small batch (reduced)
        [4096, 2048, 2048],   # DSv3 671B - medium batch (reduced)
        [2048, 2048, 2048],   # Llama4 16e - small batch (reduced)
        [4096, 2048, 2048],   # Llama4 16e - medium batch (reduced)
    ],
    cross_product_configs={
        "device": ["cuda"],
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16", "float32"],  # KEEPING ALL DTYPES
    },
    tags=["long"],
)

# Configs for scaled_grouped_mm - short configs include both models
# Note: E5M2 is not supported for matrix multiplication (only E4M3FN)
# Note: scaled_grouped_mm only supports bfloat16 output (float32 not supported)
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
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16"],  # Only bfloat16 supported for grouped gemm
    },
    tags=["short"],
)

# Configs for scaled_grouped_mm - long configs include both models with all expert counts
# Note: E5M2 is not supported for matrix multiplication (only E4M3FN)
# Note: scaled_grouped_mm only supports bfloat16 output (float32 not supported)
# REDUCED SHAPES FOR FASTER TESTING - keeping all dtypes
scaled_grouped_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [2048, 2048, 2048, 1],   # DSv3 671B - 1 expert per device (reduced)
        [2048, 2048, 2048, 2],   # DSv3 671B - 2 experts per device (reduced)
        [2048, 2048, 2048, 4],   # DSv3 671B - 4 experts per device (reduced)
        [2048, 2048, 2048, 8],   # DSv3 671B - 8 experts per device (reduced)
        [2048, 2048, 2048, 16],  # DSv3 671B - 16 experts per device (reduced)
        [4096, 2048, 2048, 1],   # DSv3 671B - medium batch, 1 expert (reduced)
        [4096, 2048, 2048, 2],   # DSv3 671B - medium batch, 2 experts (reduced)
        [4096, 2048, 2048, 4],   # DSv3 671B - medium batch, 4 experts (reduced)
        [4096, 2048, 2048, 8],   # DSv3 671B - medium batch, 8 experts (reduced)
        [4096, 2048, 2048, 16],  # DSv3 671B - medium batch, 16 experts (reduced)
        [2048, 2048, 2048, 1],   # Llama4 16e - 1 expert per device (reduced)
        [2048, 2048, 2048, 2],   # Llama4 16e - 2 experts per device (reduced)
        [2048, 2048, 2048, 4],   # Llama4 16e - 4 experts per device (reduced)
        [2048, 2048, 2048, 8],   # Llama4 16e - 8 experts per device (reduced)
        [2048, 2048, 2048, 16],  # Llama4 16e - 16 experts per device (reduced)
        [4096, 2048, 2048, 1],   # Llama4 16e - medium batch, 1 expert (reduced)
        [4096, 2048, 2048, 2],   # Llama4 16e - medium batch, 2 experts (reduced)
        [4096, 2048, 2048, 4],   # Llama4 16e - medium batch, 4 experts (reduced)
        [4096, 2048, 2048, 8],   # Llama4 16e - medium batch, 8 experts (reduced)
        [4096, 2048, 2048, 16],  # Llama4 16e - medium batch, 16 experts (reduced)
    ],
    cross_product_configs={
        "device": ["cuda"],
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16"],  # KEEPING ALL DTYPES (bfloat16 only for grouped gemm)
    },
    tags=["long"],
)

# Generate tests for scaled_mm
op_bench.generate_pt_test(
    scaled_mm_configs_short + scaled_mm_configs_long, ScaledMMBenchmark
)

# Generate tests for scaled_grouped_mm
op_bench.generate_pt_test(
    scaled_grouped_mm_configs_short + scaled_grouped_mm_configs_long,
    ScaledGroupedMMBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
