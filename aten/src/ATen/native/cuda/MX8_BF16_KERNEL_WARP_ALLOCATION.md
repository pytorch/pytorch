# MX8×MX8 BF16 Kernel: Warp Allocation and Configuration

## Overview

This document details the warp allocation and configuration for PyTorch's MX8×MX8 BF16 (microscaling FP8 to BFloat16) matrix multiplication kernels. These kernels leverage NVIDIA's CUTLASS library and are optimized for SM90+ architectures (Hopper and later).

## Kernel Variants

PyTorch provides two main implementations for MX8×MX8 matrix multiplication:

1. **Standard Scaled GEMM** - Single matrix multiplication
2. **Grouped GEMM** - Batched/grouped matrix multiplications

## Warp Configuration Details

### Thread Block Organization

The kernels use **Warp-Specialized scheduling** with the following CUTLASS dispatch policies:

- **Cooperative Mode** (default):
  - `KernelPtrArrayTmaWarpSpecializedCooperative`
  - `KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum` (fast accumulation)
  
- **Pingpong Mode** (for large tiles with fast accumulation):
  - `KernelPtrArrayTmaWarpSpecializedPingpong`
  - `KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum` (fast accumulation)

### Warp Specialization Strategy

The kernel divides warps into specialized groups:

1. **Data Movement Warps**: Handle TMA (Tensor Memory Accelerator) async copies
2. **Compute Warps**: Execute matrix multiply-accumulate (MMA) operations
3. **Epilogue Warps**: Perform output scaling and stores

This specialization allows overlapping of memory operations with computation for maximum throughput.

### Cluster Configuration

```cpp
using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
```

- **Cluster Dimensions**: 2×2×1 (4 thread blocks per cluster)
- **Thread Blocks per Cluster**: 4
- Each cluster cooperates on a larger tile of the output matrix

## Tile Size Configurations

The kernel dynamically selects tile shapes based on problem size:

### Small Problems (Default)
```cpp
TileShape = <128, 256, 64>  // TB_M=128, TB_N=256, TB_K=64
```
- **M dimension (rows)**: 128 elements per thread block
- **N dimension (cols)**: 256 elements per thread block  
- **K dimension (contraction)**: 64 elements per mainloop iteration
- **Mode**: Cooperative (no pingpong)

### Large Problems with Fast Accumulation
```cpp
TileShape = <256, 128, 128>  // TB_M=256, TB_N=128, TB_K=128
```
- Larger M tile (256) for better data reuse
- **Mode**: Cooperative (no pingpong)

### Large Problems without Fast Accumulation
```cpp
TileShape = <128, 128, 128>  // TB_M=128, TB_N=128, TB_K=128
```
- Balanced tile to avoid register spilling with slower accumulation
- **Mode**: Cooperative (no pingpong)

### Very Large Problems with Fast Accumulation + Pingpong
```cpp
TileShape = <64, 128, 128>  // TB_M=64, TB_N=128, TB_K=128
```
- Smaller M dimension for pingpong buffer management
- **Mode**: Pingpong scheduling
- Enables overlapping of two mainloop iterations

## Warp Count Calculation

For a given tile configuration, the number of warps per thread block is determined by:

```
Warps per TB = (TB_M × TB_N) / (Warp_M × Warp_N)
```

With typical warp tile sizes on SM90:
- **Warp_M**: 64
- **Warp_N**: 64 (for FP8)

### Example: Default Configuration (128×256×64)
```
Warps per TB = (128 × 256) / (64 × 64) = 32768 / 4096 = 8 warps
```

Since each SM90 thread block supports up to 32 warps (1024 threads / 32 threads per warp), the kernel uses **8 warps** with warp specialization:
- **2-4 warps**: TMA data movement
- **4-6 warps**: MMA computation  
- **0-2 warps**: Epilogue operations

## MMA Instruction Configuration

### Core MMA Operation
- **Instruction**: SM90 Tensor Core MMA with FP8 inputs
- **Per-warp MMA shape**: 64×64×32 (M×N×K per warp per iteration)
- **FP8 format**: E4M3 (4-bit exponent, 3-bit mantissa)
- **Accumulator**: FP32 (32-bit floating point)
- **Output**: BF16 (bfloat16) after epilogue scaling and conversion

### Microscaling (MX) Format Details

The MX8 format uses:
- **Data elements**: Float8_e4m3fn (standard FP8)
- **Scale elements**: Float8_e8m0fnu (8-bit exponent-only, no mantissa)
- **Scaling granularity**: BlockWise1x32 (one scale factor per 32 elements)
- **Swizzle pattern**: SWIZZLE_32_4_4 (for optimal memory access)

Scale layout on CUDA:
```
scale_a: [round_up(M, 128) × round_up(ceil_div(K, 32), 4)]
scale_b: [round_up(N, 128) × round_up(ceil_div(K, 32), 4)]
```

## Memory Access Patterns

### TMA (Tensor Memory Accelerator)
- **Alignment requirement**: 16 bytes (128 bits)
- **Access pattern**: Asynchronous bulk transfers
- **Swizzle**: 128-byte swizzle for optimal GMEM throughput

### Shared Memory Layout
- **Stage count**: Auto-carved based on epilogue storage
- **Buffering**: Double/triple buffering with pingpong for large tiles
- **Bank conflicts**: Avoided via CUTLASS swizzling

## Fast Accumulation Mode

When `use_fast_accum=True`:
- **Precision**: FP16 accumulation (instead of FP32)
- **Performance**: ~2× faster on Hopper
- **Accuracy**: Slightly reduced for large K dimensions
- **Tile selection**: Enables larger tiles (256×128×128 or pingpong mode)

## Performance Characteristics

### Theoretical Occupancy

For default configuration (128×256×64):
- **Threads per TB**: 256 threads (8 warps × 32 threads)
- **Shared memory**: ~64-96 KB (depends on stage count)
- **Registers per thread**: ~128-192
- **Occupancy**: Typically 75-100% on SM90

### Expected Throughput

On NVIDIA H100 (SM90):
- **Peak TFLOPS**: ~1000 TFLOPS (FP8 Tensor Core)
- **Achieved**: 70-85% of peak for large matrices
- **Bottleneck**: Memory bandwidth for small K dimensions

## Platform Support

- **CUDA Architecture**: SM89 (Ada), SM90+ (Hopper)
- **ROCm**: Limited support (gfx942, gfx950 on ROCm 7.0+)
- **CUTLASS Version**: 3.x required
- **PyTorch Backend**: Uses MSLK library wrapper when `USE_MSLK` is defined

## Usage Example

```python
import torch
from torch.nn.functional import scaled_mm, ScalingType

# Input tensors in FP8 E4M3 format
mat_a = torch.randn(1024, 512, dtype=torch.float8_e4m3fn, device='cuda')
mat_b = torch.randn(512, 2048, dtype=torch.float8_e4m3fn, device='cuda')

# Microscaling: one scale per 32 elements
scale_a = torch.randn(1024, 16, dtype=torch.float8_e8m0fnu, device='cuda')  # K=512 -> 512/32=16
scale_b = torch.randn(2048, 16, dtype=torch.float8_e8m0fnu, device='cuda')

# Perform MX8×MX8 → BF16 matrix multiplication
result = scaled_mm(
    mat_a, mat_b,
    scale_a=[scale_a],
    scale_recipe_a=[ScalingType.BlockWise1x32],
    scale_b=[scale_b],
    scale_recipe_b=[ScalingType.BlockWise1x32],
    out_dtype=torch.bfloat16,
    use_fast_accum=True
)
```

## Key Source Files

- **Kernel Implementation**: `aten/src/ATen/native/cuda/ScaledGroupMM.cu`
- **Dispatch Logic**: `aten/src/ATen/native/cuda/ScaledBlas.cpp`
- **MSLK Integration**: `aten/src/ATen/cuda/CUDAScaledBlas.cpp`
- **Header Definitions**: `aten/src/ATen/cuda/CUDAScaledBlas.h`
- **Common Utilities**: `aten/src/ATen/native/cuda/GroupMMCommon.cuh`

## References

1. [CUTLASS 3.x Documentation](https://github.com/NVIDIA/cutlass)
2. [NVIDIA Hopper Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
3. [PyTorch Scaled Matrix Multiplication](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html)
4. [Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
