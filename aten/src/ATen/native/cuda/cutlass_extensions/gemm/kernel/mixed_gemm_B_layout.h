/*
  This file exists so that we use the same weight layout for MoE grouped gemm and regular gemm when the weight is
  quantized. The preprocessing code reads this template to know how to organize the quantized weight matrices
  to be consumed by CUTLASS.

  Note that for int4, ThreadBlockK MUST be 64.

 */

#pragma once

#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/platform/platform.h>

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {
};

// Volta specialiations. Volta will dequantize before STS, so we need a different operator
template<typename TypeB>
struct LayoutDetailsB<TypeB, arch::Sm70> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 8;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

// Specializations for Turing+ when B is FP16. These are currently only used for MoE networks.
// TODO - Switch this to column major for weights since gemms should be more performant.
template<typename Arch>
struct LayoutDetailsB<half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

template<typename Arch>
struct LayoutDetailsB<bfloat16_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<bfloat16_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

// Specializations for Turing+ when B is quantized. These can use the operator OpMultiplyAddDequantizeInterleavedBToA,
// which signals that we want to dequantize after loading from smem.
template<typename Arch>
struct LayoutDetailsB<uint8_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK = 64;

private:
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

public:
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint8_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

template<typename Arch>
struct LayoutDetailsB<uint4b_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK = 64;

private:
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint4b_t>::value;
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

public:
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint4b_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
