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
#include <string_view>

namespace at::native {
namespace {

using namespace mps;
using namespace std::string_view_literals;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GroupedMM_metallib.h>
#endif

// Prefer the largest row tile that the typical group height keeps busy.
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
  return {
      c10::checked_convert<uint32_t>(mat_a.size(-2), "mat_a.size(-2)"),
      c10::checked_convert<uint32_t>(mat_b.size(-1), "mat_b.size(-1)"),
      c10::checked_convert<uint32_t>(mat_a.size(-1), "mat_a.size(-1)"),
      groups,
      static_cast<idx_t>(mat_a.stride(-2)),
      static_cast<idx_t>(mat_a.stride(-1)),
      mat_a.dim() == 3 ? static_cast<idx_t>(mat_a.stride(0)) : idx_t(0),
      static_cast<idx_t>(mat_b.stride(-2)),
      static_cast<idx_t>(mat_b.stride(-1)),
      mat_b.dim() == 3 ? static_cast<idx_t>(mat_b.stride(0)) : idx_t(0),
      static_cast<idx_t>(out.stride(-2)),
      static_cast<idx_t>(out.stride(-1)),
      out.dim() == 3 ? static_cast<idx_t>(out.stride(0)) : idx_t(0),
  };
}

// Wider output tiles cut per-element DRAM traffic ((BM+BN)*k bytes per tile);
// pick them once mat_b is too big to stay cache-resident, where the MPP
// kernels are otherwise bandwidth-bound streaming it once per row tile.
uint32_t grouped_mm_mpp_tile_cols(uint32_t n, uint64_t mat_b_bytes, uint32_t bm) {
  const bool streams_weights_from_dram = n >= 1024 && mat_b_bytes >= (32ull << 20);
  return streams_weights_from_dram ? (bm == 64 ? 256u : 128u) : kGroupedMMTileN;
}

bool grouped_mm_mpp_tensor_offsets_fit(uint64_t rows, uint64_t row_stride, uint64_t cols, uint64_t col_stride) {
  constexpr auto int_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (rows == 0 || cols == 0) {
    return true;
  }

  const auto row_extent = rows - 1;
  if (row_extent != 0 && row_stride > int_max / row_extent) {
    return false;
  }
  const auto row_offset = row_extent * row_stride;
  const auto col_extent = cols - 1;
  return col_extent == 0 || col_stride <= (int_max - row_offset) / col_extent;
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

// Runs one of the jagged modes; jagged_rows selects whether the offsets split
// the rows of mat_a (2d x 3d) or the shared contraction dim (2d x 2d).
void grouped_mm_out_mps(const Tensor& mat_a,
                        const Tensor& mat_b,
                        const Tensor& offsets,
                        const Tensor& out,
                        bool jagged_rows) {
  const auto groups = c10::checked_convert<uint32_t>(offsets.numel(), "number of groups");
  if (groups == 0 || out.numel() == 0) {
    return;
  }

  const auto mode = jagged_rows ? "rows"sv : "k"sv;
  const auto bm = grouped_mm_tile_rows(jagged_rows ? mat_a.size(0) / groups : mat_a.size(0));
  const auto params = grouped_mm_params<uint64_t>(mat_a, mat_b, out, groups);
  // Every validated operand is row- or column-major; matmul2d wants that
  // orientation spelled out per operand ('n' row-major, 't' column-major) and
  // a row-major output, so the transposed-output 3d x 2d fallback call lands
  // on the simdgroup kernels.
  const char a_layout = params.a_stride_k == 1 ? 'n' : 't';
  const char b_layout = params.b_stride_k == 1 ? 't' : 'n';
  const auto dtype = scalarToMetalTypeString(out);
  const auto mpp_bn = grouped_mm_mpp_tile_cols(params.n, mat_b.nbytes(), bm);
  const auto mpp_kernel_name =
      fmt::format("grouped_mm_{}_mpp_{}{}_{}_bm{}_bn{}", mode, a_layout, b_layout, dtype, bm, mpp_bn);
  const bool use_mpp = has_mpp() && grouped_mm_mpp_indices_fit(params, bm, mpp_bn) && params.out_stride_n == 1 &&
      lib.hasFunction(mpp_kernel_name);
  const bool use_u32 = !use_mpp && offsetsFitIn<int32_t>(mat_a, mat_b, out);
  const auto kernel_name =
      use_mpp ? mpp_kernel_name : fmt::format("grouped_mm_{}_{}_bm{}{}", mode, dtype, bm, mtlIdxSuffix(use_u32));
  const auto pipeline = lib.getPipelineStateForFunc(kernel_name);
  const auto bn = use_mpp ? mpp_bn : kGroupedMMTileN;
  // The rows grid covers the worst case of one extra partial tile per group;
  // the kernel discards the excess tiles.
  const auto threadgroups = MTLSizeMake(at::ceil_div<NSUInteger>(params.n, bn),
                                        at::ceil_div<NSUInteger>(params.m, bm) + (jagged_rows ? groups : 0),
                                        jagged_rows ? 1 : groups);
  const auto threads = MTLSizeMake(grouped_mm_mpp_simdgroups(bm, bn) * c10::metal::simdgroup_size, 1, 1);
  const auto profile_name = fmt::format("grouped_mm_{}{}", mode, use_mpp ? "_mpp"sv : ""sv);
  auto stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, profile_name, {mat_a, mat_b, offsets}, stream);
      [encoder setComputePipelineState:pipeline];
      mtlDispatchByIndexWidth<uint32_t, uint64_t>(use_u32, [&](auto idx_tag) {
        using idx_t = typename decltype(idx_tag)::type;
        mtl_setArgs(encoder, mat_a, mat_b, offsets, out, grouped_mm_params<idx_t>(mat_a, mat_b, out, groups));
      });
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

std::string grouped_mm_cols_mpp_kernel_name(const Tensor& mat_a, const Tensor& mat_b, const Tensor& out) {
  const auto bm = grouped_mm_tile_rows(mat_a.size(-2));
  const auto bn = grouped_mm_mpp_tile_cols(mat_b.size(-1), mat_b.nbytes(), bm);
  const char a_layout = mat_a.stride(-1) == 1 ? 'n' : 't';
  const char b_layout = mat_b.stride(-2) == 1 ? 't' : 'n';
  const auto dtype = scalarToMetalTypeString(out);
  return fmt::format("grouped_mm_cols_mpp_{}{}_{}_bm{}_bn{}", a_layout, b_layout, dtype, bm, bn);
}

bool grouped_mm_cols_mpp_available(const Tensor& mat_a, const Tensor& mat_b, const Tensor& offsets, const Tensor& out) {
  const auto groups = c10::checked_convert<uint32_t>(offsets.numel(), "number of groups");
  const auto bm = grouped_mm_tile_rows(mat_a.size(-2));
  const auto bn = grouped_mm_mpp_tile_cols(mat_b.size(-1), mat_b.nbytes(), bm);
  const auto params = grouped_mm_params<uint64_t>(mat_a, mat_b, out, groups);
  return has_mpp() && grouped_mm_mpp_indices_fit(params, bm, bn) && params.out_stride_n == 1 &&
      lib.hasFunction(grouped_mm_cols_mpp_kernel_name(mat_a, mat_b, out));
}

// 3d x 2d: the offsets split the columns of mat_b and out, so the row-major
// out is written directly instead of routing through the transpose identity
// (which would need a transposed store that matmul2d does not have).
void grouped_mm_cols_mpp_out_mps(const Tensor& mat_a, const Tensor& mat_b, const Tensor& offsets, const Tensor& out) {
  const auto groups = c10::checked_convert<uint32_t>(offsets.numel(), "number of groups");
  if (groups == 0 || out.numel() == 0) {
    return;
  }

  const auto bm = grouped_mm_tile_rows(mat_a.size(-2));
  const auto bn = grouped_mm_mpp_tile_cols(mat_b.size(-1), mat_b.nbytes(), bm);
  const auto params = grouped_mm_params<uint64_t>(mat_a, mat_b, out, groups);
  const auto pipeline = lib.getPipelineStateForFunc(grouped_mm_cols_mpp_kernel_name(mat_a, mat_b, out));
  const auto threadgroups =
      MTLSizeMake(at::ceil_div<NSUInteger>(params.n, bn) + groups, at::ceil_div<NSUInteger>(params.m, bm), 1);
  const auto threads = MTLSizeMake(grouped_mm_mpp_simdgroups(bm, bn) * c10::metal::simdgroup_size, 1, 1);
  auto stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "grouped_mm_cols_mpp", {mat_a, mat_b, offsets}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, mat_a, mat_b, offsets, out, params);
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
  TORCH_CHECK(mat_a.device().is_mps() && mat_b.device().is_mps(),
              "Expected mat_a and mat_b to be MPS tensors, but got ",
              mat_a.device(),
              " and ",
              mat_b.device());
  TORCH_CHECK(mat_a.device() == mat_b.device(),
              "Expected mat_a and mat_b to be on the same device, but got ",
              mat_a.device(),
              " and ",
              mat_b.device());
  TORCH_CHECK(!offs.has_value() || offs->device() == mat_a.device(),
              "Expected offsets to be on the same device as the inputs");
  TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(),
              "Expected mat_a and mat_b to have the same dtype, but got ",
              mat_a.scalar_type(),
              " and ",
              mat_b.scalar_type());
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  // The shared validation skips this check when both operands are 2d (jagged
  // contraction dims); without it the kernel silently truncates to the smaller.
  TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");

  const auto output_dtype = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  auto out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, output_dtype);
  if (mat_a.dim() == 3 && mat_b.dim() == 3) {
    at::bmm_out(out, mat_a, mat_b);
    return out;
  }

  TORCH_INTERNAL_ASSERT(offs.has_value());
  const auto& offsets = *offs;
  if (mat_a.dim() == 2 && mat_b.dim() == 3) {
    grouped_mm_out_mps(mat_a, mat_b, offsets, out, /*jagged_rows=*/true);
  } else if (mat_a.dim() == 3 && mat_b.dim() == 2) {
    if (grouped_mm_cols_mpp_available(mat_a, mat_b, offsets, out)) {
      grouped_mm_cols_mpp_out_mps(mat_a, mat_b, offsets, out);
    } else {
      // Without matmul2d the jagged columns are cheaper to reach through the
      // transpose identity, which the simdgroup rows kernels can store.
      grouped_mm_out_mps(mat_b.transpose(0, 1),
                         mat_a.transpose(-2, -1),
                         offsets,
                         out.transpose(0, 1),
                         /*jagged_rows=*/true);
    }
  } else {
    grouped_mm_out_mps(mat_a, mat_b, offsets, out, /*jagged_rows=*/false);
  }
  return out;
}

} // namespace at::native
