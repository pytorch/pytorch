from pt import configs

import operator_benchmark as op_bench

import torch
from torch.nn.functional import ScalingType, SwizzleType


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


# Platform check for MXFP8 support
def _check_mxfp8_support():
    """Check if platform supports MXFP8 grouped gemm (requires SM100+)."""
    if not torch.cuda.is_available():
        return False

    # Check CUDA compute capability
    capability = torch.cuda.get_device_capability()
    major, minor = capability
    # SM100 = compute capability 10.0
    return major >= 10


# MXFP8 helper functions (from test_scaled_matmul_cuda.py)
def _to_mxfp8(t):
    """Convert tensor to MXFP8 format with block-wise scaling."""
    try:
        from torch.testing._internal.common_quantized import to_mxfp, from_blocked_format
    except ImportError:
        raise ImportError("MXFP8 support requires torch.testing._internal.common_quantized")

    # Convert to MXFP8: returns (scale, quantized_tensor)
    t_scale, t_lp = to_mxfp(t, format="mxfp8")
    # Reconstruct high-precision reference
    t_hp = from_blocked_format(t_lp, t_scale, blocksize=32)

    return t_hp, t_lp, t_scale


def _generate_jagged_offs(G, total_K, multiple_of=32, device="cuda"):
    """Generate jagged offsets for grouped operations."""
    try:
        from torch.testing._internal.common_quantized import generate_jagged_offs
        return generate_jagged_offs(G, total_K, multiple_of=multiple_of, device=device)
    except ImportError:
        # Fallback: equal-sized groups aligned to multiple_of
        group_size = (total_K + G - 1) // G
        group_size = ((group_size + multiple_of - 1) // multiple_of) * multiple_of
        offs = torch.arange(group_size, total_K + 1, group_size, device=device, dtype=torch.int32)
        if len(offs) < G:
            offs = torch.cat([offs, torch.tensor([total_K], device=device, dtype=torch.int32)])
        return offs[:G]


def _to_blocked(scale):
    """Convert scale tensor to blocked format with swizzling."""
    try:
        from torch.testing._internal.common_quantized import to_blocked
        return to_blocked(scale)
    except ImportError:
        return scale  # No swizzling if not available


def _round_up(x: int, y: int) -> int:
    """Round up x to nearest multiple of y."""
    return ((x + y - 1) // y) * y


def _convert_2d_grouped_tensor_to_mxfp8(t, MN, G, offs):
    """Convert 2D grouped tensor to MXFP8 format.

    Args:
        t: Input tensor of shape (MN, K)
        MN: Number of rows (M for input, N for weight)
        G: Number of groups
        offs: Group end offsets along K dimension

    Returns:
        t_hp: High-precision reconstructed tensor
        t_lp: MXFP8 quantized tensor
        t_blocked_scales: Block-wise scales (swizzled if CUDA)
    """
    th_list = []
    t_list = []
    t_blocked_scale_list = []

    for group_idx in range(G):
        # Get group slice along K dimension
        prev_group_end_offset = 0 if group_idx == 0 else offs[group_idx - 1]
        curr_group_end_offset = offs[group_idx]
        group_size = curr_group_end_offset - prev_group_end_offset

        if group_size > 0:
            t_slice = t[:, prev_group_end_offset:curr_group_end_offset].contiguous()

            # Convert slice to MXFP8
            th_slice, tq_slice, t_scale_slice = _to_mxfp8(t_slice)

            # Swizzle scales if on CUDA
            if torch.version.cuda:
                t_scale_slice = _to_blocked(t_scale_slice)

            t_list.append(tq_slice)
            th_list.append(th_slice)
            t_blocked_scale_list.append(t_scale_slice)

    # Concatenate all groups
    tq = torch.cat(t_list, dim=1).contiguous()
    th = torch.cat(th_list, dim=1).contiguous()

    # Combine all blocked scales into one tensor
    t_blocked_scales = torch.cat(t_blocked_scale_list, dim=0)
    MN_rounded = _round_up(MN, 128)
    t_blocked_scales = t_blocked_scales.reshape(MN_rounded, -1)

    return th, tq, t_blocked_scales


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


class MXFP8GroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    """Benchmark for MXFP8 grouped matrix multiplication.

    This benchmark requires SM100+ GPUs (B100/B200/B300) and uses BlockWise1x32 scaling.
    If the platform doesn't support MXFP8, the benchmark will be skipped.
    """

    def init(self, M, N, K, G, device):
        # Check SM100+ support - skip if not available
        if not _check_mxfp8_support():
            # Set empty inputs to skip benchmark gracefully
            self.inputs = {}
            print(f"Skipping MXFP8 benchmark: requires SM100+ GPU (compute capability >= 10.0)")
            return

        self.base_dtype = torch.bfloat16
        self.output_dtype = torch.bfloat16  # MXFP8 grouped gemm only supports bfloat16

        # Create random inputs (2D-2D grouped pattern from test)
        # X is (M, total_K) - 2D input with groups along K dimension
        # W is (N, total_K) - 2D weight with groups along K dimension
        total_K = K  # Total K dimension contains G groups
        X = torch.randn(M, total_K, device=device, dtype=self.base_dtype) * 0.1
        W = torch.randn(N, total_K, device=device, dtype=self.base_dtype) * 0.01

        # Generate jagged offsets for grouped operations
        # These define group boundaries along K dimension
        input_group_end_offsets = _generate_jagged_offs(G, total_K, multiple_of=32, device=device)

        # Convert to MXFP8 format with block-wise scaling
        try:
            xh, xq, x_blocked_scales = _convert_2d_grouped_tensor_to_mxfp8(
                X, M, G, input_group_end_offsets
            )
            wh, wq, w_blocked_scales = _convert_2d_grouped_tensor_to_mxfp8(
                W, N, G, input_group_end_offsets
            )
        except ImportError as e:
            self.inputs = {}
            print(f"Skipping MXFP8 benchmark: {e}")
            return

        # Determine swizzle type
        swizzle = SwizzleType.NO_SWIZZLE
        if torch.version.cuda:
            swizzle = SwizzleType.SWIZZLE_32_4_4

        self.inputs = {
            "x": xq,
            "w": wq,
            "offs": input_group_end_offsets,
            "scale_a": x_blocked_scales,
            "scale_b": w_blocked_scales,
            "swizzle": swizzle,
        }
        self.set_module_name("mxfp8_grouped_mm")

    def forward(self, x, w, offs, scale_a, scale_b, swizzle):
        return torch.nn.functional.scaled_grouped_mm(
            x,
            w.t(),  # Transpose W from (N, K) to (K, N)
            scale_a=scale_a,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            offs=offs,
            output_dtype=self.output_dtype,
            wrap_v2=True,
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
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16", "float32"],
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
scaled_grouped_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [16384, 2048, 7168, 1],   # DSv3 671B - 1 expert per device
        [16384, 2048, 7168, 2],   # DSv3 671B - 2 experts per device
        [16384, 2048, 7168, 4],   # DSv3 671B - 4 experts per device
        [16384, 2048, 7168, 8],   # DSv3 671B - 8 experts per device
        [16384, 2048, 7168, 16],  # DSv3 671B - 16 experts per device
        [128000, 2048, 7168, 1],  # DSv3 671B - large batch, 1 expert
        [128000, 2048, 7168, 2],  # DSv3 671B - large batch, 2 experts
        [128000, 2048, 7168, 4],  # DSv3 671B - large batch, 4 experts
        [128000, 2048, 7168, 8],  # DSv3 671B - large batch, 8 experts
        [128000, 2048, 7168, 16], # DSv3 671B - large batch, 16 experts
        [16384, 8192, 5120, 1],   # Llama4 16e - 1 expert per device
        [16384, 8192, 5120, 2],   # Llama4 16e - 2 experts per device
        [16384, 8192, 5120, 4],   # Llama4 16e - 4 experts per device
        [16384, 8192, 5120, 8],   # Llama4 16e - 8 experts per device
        [16384, 8192, 5120, 16],  # Llama4 16e - 16 experts per device
        [128000, 8192, 5120, 1],  # Llama4 16e - large batch, 1 expert
        [128000, 8192, 5120, 2],  # Llama4 16e - large batch, 2 experts
        [128000, 8192, 5120, 4],  # Llama4 16e - large batch, 4 experts
        [128000, 8192, 5120, 8],  # Llama4 16e - large batch, 8 experts
        [128000, 8192, 5120, 16], # Llama4 16e - large batch, 16 experts
    ],
    cross_product_configs={
        "device": ["cuda"],
        "float8_dtype": ["e4m3fn"],  # Only E4M3FN supported for matmul
        "output_dtype": ["bfloat16"],  # Only bfloat16 supported for grouped gemm
    },
    tags=["long"],
)

# Configs for MXFP8 grouped_mm (2D-2D pattern with BlockWise1x32 scaling)
# Note: Requires SM100+ GPUs (B100/B200/B300)
# Note: Only bfloat16 output supported
# Based on test sizes from test_mxfp8_nvfp4_scaled_grouped_mm_2d_2d
mxfp8_grouped_mm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [2048, 8192, 16640, 1],   # MXFP8 test size - 1 expert
        [2048, 8192, 16640, 4],   # MXFP8 test size - 4 experts
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["short"],
)

# Long configs include more expert counts
mxfp8_grouped_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [2048, 8192, 16640, 1],   # MXFP8 test size - 1 expert
        [2048, 8192, 16640, 4],   # MXFP8 test size - 4 experts
        [2048, 8192, 16640, 16],  # MXFP8 test size - 16 experts
        [2049, 8192, 16640, 1],   # Unaligned M dimension - 1 expert
        [2049, 8192, 16640, 4],   # Unaligned M dimension - 4 experts
        [2049, 8192, 16640, 16],  # Unaligned M dimension - 16 experts
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

# Generate tests for scaled_grouped_mm
op_bench.generate_pt_test(
    scaled_grouped_mm_configs_short + scaled_grouped_mm_configs_long,
    ScaledGroupedMMBenchmark,
)

# Generate tests for MXFP8 grouped_mm
# Note: These will be skipped if platform doesn't support MXFP8 (requires SM100+)
op_bench.generate_pt_test(
    mxfp8_grouped_mm_configs_short + mxfp8_grouped_mm_configs_long,
    MXFP8GroupedMMBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
