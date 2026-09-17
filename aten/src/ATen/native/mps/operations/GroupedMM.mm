#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GroupedMM.h>
#include <ATen/ops/bmm.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#endif

#include <algorithm>
#include <limits>

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GroupedMM_metallib.h>
#endif

// Prefer the largest row tile that the typical group height keeps busy; the
// 48/24 cut points keep the picked tile at least 75% occupied.
uint32_t grouped_mm_tile_rows(int64_t rows) {
  if (rows > 48) {
    return 64;
  }
  if (rows > 24) {
    return 32;
  }
  return 16;
}

template <typename idx_t>
GroupedMMParams<idx_t> grouped_mm_params(const Tensor& mat_a, const Tensor& mat_b, const Tensor& out, uint32_t groups) {
  const auto& batched = mat_a.dim() == 3 ? mat_a : mat_b.dim() == 3 ? mat_b : out;
  return {
      .m = c10::checked_convert<uint32_t>(mat_a.size(-2), "mat_a.size(-2)"),
      .n = c10::checked_convert<uint32_t>(mat_b.size(-1), "mat_b.size(-1)"),
      .k = c10::checked_convert<uint32_t>(std::min(mat_a.size(-1), mat_b.size(-2)), "contraction dimension"),
      .groups = groups,
      .a_stride_m = static_cast<idx_t>(mat_a.stride(-2)),
      .a_stride_k = static_cast<idx_t>(mat_a.stride(-1)),
      .b_stride_k = static_cast<idx_t>(mat_b.stride(-2)),
      .b_stride_n = static_cast<idx_t>(mat_b.stride(-1)),
      .out_stride_m = static_cast<idx_t>(out.stride(-2)),
      .out_stride_n = static_cast<idx_t>(out.stride(-1)),
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
  return streams_weights_from_dram ? (bm == 64 ? 256u : 128u) : kGroupedMMTileN;
}

bool grouped_mm_mpp_tensor_offsets_fit(uint64_t rows, uint64_t row_stride, uint64_t cols, uint64_t col_stride) {
  constexpr auto int_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (rows == 0 || cols == 0) {
    return true;
  }

  // The caller bounds dimensions and strides to int32, so the sum fits in uint64.
  return (rows - 1) * row_stride + (cols - 1) * col_stride <= int_max;
}

// matmul2d takes int32 extents and strides and computes offsets relative to
// each tile in int32. The kernels apply batch and tile bases in uint64 first.
bool grouped_mm_mpp_indices_fit(const GroupedMMParams<uint64_t>& params, uint32_t bm, uint32_t bn) {
  constexpr auto int_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (params.m > int_max || params.n > int_max || params.k > int_max || params.a_stride_m > int_max ||
      params.a_stride_k > int_max || params.b_stride_k > int_max || params.b_stride_n > int_max ||
      params.out_stride_m > int_max || params.out_stride_n > int_max) {
    return false;
  }

  const auto tile_rows = std::min(params.m, bm);
  const auto tile_cols = std::min(params.n, bn);
  return grouped_mm_mpp_tensor_offsets_fit(tile_rows, params.a_stride_m, params.k, params.a_stride_k) &&
      grouped_mm_mpp_tensor_offsets_fit(params.k, params.b_stride_k, tile_cols, params.b_stride_n) &&
      grouped_mm_mpp_tensor_offsets_fit(tile_rows, params.out_stride_m, tile_cols, params.out_stride_n);
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
  const auto bm = grouped_mm_tile_rows(jagged_rows ? mat_a.size(0) / groups : mat_a.size(-2));
  const auto mpp_bn = grouped_mm_mpp_tile_cols(params.n, mat_b.nbytes(), bm);
  // Every operand is either row or col major, because mpp matmul2d wants
  // to know the orientation in advance per operand (`n` row-major, `t` column major)
  // and output is always row-major, so the transposed-output 3d x 2d fallback call lands on the simdgroup kernels.
  const char a_layout = params.a_stride_k == 1 ? 'n' : 't';
  const char b_layout = params.b_stride_k == 1 ? 't' : 'n';
  const auto dtype = scalarToMetalTypeString(out);
  const bool use_mpp = has_mpp() && grouped_mm_mpp_indices_fit(params, bm, mpp_bn) && params.out_stride_n == 1;
  if (jagged_cols && !use_mpp) {
    // Without matmul2d the jagged columns are cheaper to reach through the
    // transpose identity, which the simdgroup rows kernels can store.
    grouped_mm_out_mps(mat_b.transpose(0, 1), mat_a.transpose(-2, -1), offsets, out.transpose(0, 1));
    return;
  }

  const bool use_u32 = !use_mpp && offsetsFitIn<int32_t>(mat_a, mat_b, out);
  const auto mpp_kernel_name =
      fmt::format("grouped_mm_{}_mpp_{}{}_{}_bm{}_bn{}", mode, a_layout, b_layout, dtype, bm, mpp_bn);
  const auto simdgroup_kernel_name = fmt::format("grouped_mm_{}_{}_bm{}{}", mode, dtype, bm, mtlIdxSuffix(use_u32));
  const auto kernel_name = use_mpp ? mpp_kernel_name : simdgroup_kernel_name;
  const auto pipeline = lib.getPipelineStateForFunc(kernel_name);
  const auto bn = use_mpp ? mpp_bn : kGroupedMMTileN;
  // The jagged grid axis covers the worst case of one extra partial tile per
  // group; the kernel discards the excess tiles.
  const auto threadgroups = MTLSizeMake(at::ceil_div<NSUInteger>(params.n, bn) + (jagged_cols ? groups : 0),
                                        at::ceil_div<NSUInteger>(params.m, bm) + (jagged_rows ? groups : 0),
                                        jagged_rows || jagged_cols ? 1 : groups);
  const auto threads = MTLSizeMake(grouped_mm_simdgroups(bm, bn) * c10::metal::simdgroup_size, 1, 1);
  const auto profile_name = fmt::format("grouped_mm_{}{}", mode, use_mpp ? "_mpp" : "");
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
