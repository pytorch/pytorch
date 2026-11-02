#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GroupedMM.h>
#include <ATen/native/mps/operations/Matmul.h>
#include <ATen/ops/bmm.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#endif

#include <algorithm>

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GroupedMM_metallib.h>
#endif

template <typename idx_t>
GroupedMMParams<idx_t> grouped_mm_params(const Tensor& mat_a, const Tensor& mat_b, const Tensor& out, uint32_t groups) {
  const auto& batched = mat_a.dim() == 3 ? mat_a : mat_b.dim() == 3 ? mat_b : out;
  return {
      .mm = matmul_params<idx_t>(mat_a, mat_b, out),
      .groups = groups,
      .batch_stride = static_cast<idx_t>(batched.stride(0)),
  };
}

// Wider output tiles cut per-element DRAM traffic ((BM+BN)*k bytes per tile);
// pick them once mat_b is too big to stay cache-resident, where the MPP
// kernels are otherwise bandwidth-bound streaming it once per row tile. The
// 1024-column / 32 MiB cutoffs approximate the last-level cache of current
// Apple GPUs.
uint32_t grouped_mm_mpp_tile_cols(uint32_t n, uint64_t mat_b_bytes, uint32_t bm) {
  const bool streams_weights_from_dram = n >= 1024 && mat_b_bytes >= (32ull << 20);
  return streams_weights_from_dram ? (bm == 64 ? 256u : 128u) : kMatmulTileN;
}

enum class GroupedMMKernel {
  // One SIMD group per output element, for decode-sized jagged rows.
  FewRows,
  // Tiled MetalPerformancePrimitives matmul2d.
  MPP,
  // Tiled simdgroup_matrix fallback.
  Simdgroup,
};

GroupedMMKernel grouped_mm_pick_kernel(const GroupedMMParams<uint64_t>& params,
                                       bool jagged_rows,
                                       uint32_t bm,
                                       uint32_t mpp_bn) {
  // Decode routes a handful of tokens to top_k experts each, so with no more
  // rows than groups the tiled kernels would launch mostly empty tiles.
  const bool few_rows_per_group = jagged_rows && params.mm.m <= std::min(kGroupedMMFewRowsMax, params.groups);
  // The few rows kernel vector-loads mat_a rows and mat_b columns along k.
  const bool k_contiguous = params.mm.a_stride_k == 1 && params.mm.b_stride_k == 1;
  if (few_rows_per_group && k_contiguous) {
    return GroupedMMKernel::FewRows;
  }
  // matmul2d stores row-major output tiles and indexes within a tile in int32.
  const bool mpp_supported = has_mpp() && params.mm.out_stride_n == 1 && matmul_mpp_indices_fit(params.mm, bm, mpp_bn);
  return mpp_supported ? GroupedMMKernel::MPP : GroupedMMKernel::Simdgroup;
}

// The operand ranks pick the jagged mode: 2d x 3d splits the rows of mat_a,
// 3d x 2d the columns of mat_b, 2d x 2d the shared contraction dim.
void grouped_mm_out_mps(const Tensor& mat_a, const Tensor& mat_b, const Tensor& offsets, const Tensor& out) {
  const auto groups = c10::checked_convert<uint32_t>(offsets.numel(), "number of groups");
  if (groups == 0 || out.numel() == 0) {
    return;
  }

  const bool jagged_rows = mat_a.dim() == 2 && mat_b.dim() == 3;
  const bool jagged_cols = mat_a.dim() == 3 && mat_b.dim() == 2;
  // mode k here is a 2D x 2D matmul but with offsets and that's why we can't use regular mm.
  const auto mode = jagged_rows ? "rows" : jagged_cols ? "cols" : "k";
  const auto params = grouped_mm_params<uint64_t>(mat_a, mat_b, out, groups);
  const auto bm = matmul_tile_rows(jagged_rows ? mat_a.size(0) / groups : mat_a.size(-2));
  const auto mpp_bn = grouped_mm_mpp_tile_cols(params.mm.n, mat_b.nbytes(), bm);
  const auto kernel = grouped_mm_pick_kernel(params, jagged_rows, bm, mpp_bn);
  if (jagged_cols && kernel == GroupedMMKernel::Simdgroup) {
    // Without matmul2d the jagged columns are cheaper to reach through the
    // transpose identity, which the simdgroup rows kernels can store.
    grouped_mm_out_mps(mat_b.transpose(0, 1), mat_a.transpose(-2, -1), offsets, out.transpose(0, 1));
    return;
  }

  // Every operand is either row or col major, because mpp matmul2d wants
  // to know the orientation in advance per operand (`n` row-major, `t` column major)
  // and output is always row-major, so the transposed-output 3d x 2d fallback call lands on the simdgroup kernels.
  const char a_layout = params.mm.a_stride_k == 1 ? 'n' : 't';
  const char b_layout = params.mm.b_stride_k == 1 ? 't' : 'n';
  const auto dtype = scalarToMetalTypeString(out);
  const bool use_u32 = kernel != GroupedMMKernel::MPP && offsetsFitIn<int32_t>(mat_a, mat_b, out);
  // The jagged grid axis covers the worst case of one extra partial tile per
  // group; the kernel discards the excess tiles.
  const auto tile_grid = [&](uint32_t bn) {
    return MTLSizeMake(at::ceil_div<NSUInteger>(params.mm.n, bn) + (jagged_cols ? groups : 0),
                       at::ceil_div<NSUInteger>(params.mm.m, bm) + (jagged_rows ? groups : 0),
                       jagged_rows || jagged_cols ? 1 : groups);
  };
  std::string kernel_name;
  std::string profile_name;
  MTLSize threadgroups;
  uint32_t simdgroups;
  switch (kernel) {
    case GroupedMMKernel::FewRows:
      kernel_name = fmt::format("grouped_mm_few_rows_{}{}", dtype, mtlIdxSuffix(use_u32));
      profile_name = "grouped_mm_few_rows";
      threadgroups = MTLSizeMake(at::ceil_div<NSUInteger>(params.mm.n, kGroupedMMFewRowsSimdgroups), params.mm.m, 1);
      simdgroups = kGroupedMMFewRowsSimdgroups;
      break;
    case GroupedMMKernel::MPP:
      kernel_name = fmt::format("grouped_mm_{}_mpp_{}{}_{}_bm{}_bn{}", mode, a_layout, b_layout, dtype, bm, mpp_bn);
      profile_name = fmt::format("grouped_mm_{}_mpp", mode);
      threadgroups = tile_grid(mpp_bn);
      simdgroups = matmul_simdgroups(bm, mpp_bn);
      break;
    case GroupedMMKernel::Simdgroup:
      kernel_name = fmt::format("grouped_mm_{}_{}_bm{}{}", mode, dtype, bm, mtlIdxSuffix(use_u32));
      profile_name = fmt::format("grouped_mm_{}", mode);
      threadgroups = tile_grid(kMatmulTileN);
      simdgroups = matmul_simdgroups(bm, kMatmulTileN);
      break;
  }
  const auto pipeline = lib.getPipelineStateForFunc(kernel_name);
  const auto threads = MTLSizeMake(simdgroups * c10::metal::simdgroup_size, 1, 1);
  auto stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, profile_name, {mat_a, mat_b, offsets}, stream);
      [encoder setComputePipelineState:pipeline];
      if (use_u32) {
        mtl_setArgs(encoder, mat_a, mat_b, offsets, out, grouped_mm_params<uint32_t>(mat_a, mat_b, out, groups));
      } else {
        mtl_setArgs(encoder, mat_a, mat_b, offsets, out, params);
      }
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

} // namespace

Tensor _grouped_mm_mps(const Tensor& mat_a,
                       const Tensor& mat_b,
                       const std::optional<Tensor>& offs,
                       const std::optional<Tensor>& bias,
                       std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);

  const auto output_dtype = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  auto out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, output_dtype);
  if (mat_a.dim() == 3 && mat_b.dim() == 3) {
    at::bmm_out(out, mat_a, mat_b);
    return out;
  }

  TORCH_INTERNAL_ASSERT(offs.has_value());
  grouped_mm_out_mps(mat_a, mat_b, *offs, out);
  return out;
}

} // namespace at::native
