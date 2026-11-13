#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/NormBackwardHeuristics.h>
#include <c10/util/safe_conv.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/norm_except_dim.h>
#endif

namespace at::native {

using namespace at::native::mps;
using namespace std::string_view_literals;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/WeightNorm_metallib.h>
#endif

std::tuple<Tensor, Tensor> weight_norm_mps(const Tensor& v, const Tensor& g, int64_t dim) {
  TORCH_CHECK(dim == 0 || dim == v.dim() - 1, "fused kernels can only be applied for first or last dim")
  if (v.numel() == 0) {
    return {at::empty_like(v), at::zeros_like(g)};
  }
  auto norms = at::norm_except_dim(v, 2, dim);
  return {v.mul(g.div(norms)), norms};
}

std::tuple<Tensor, Tensor> weight_norm_backward_mps(const Tensor& grad_w,
                                                    const Tensor& saved_v,
                                                    const Tensor& saved_g,
                                                    const Tensor& saved_norms,
                                                    int64_t dim) {
  TORCH_CHECK(dim == 0 || dim == saved_v.dim() - 1, "fused kernels can only be applied for first or last dim")
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(saved_v),
                              "weight_norm backward is not implemented for '",
                              toString(saved_v.scalar_type()),
                              "' on MPS");
  TORCH_CHECK_VALUE(
      grad_w.is_contiguous() && saved_v.is_contiguous() && saved_g.is_contiguous() && saved_norms.is_contiguous(),
      "weight_norm backward expects contiguous tensors");
  TORCH_CHECK_VALUE(grad_w.sizes() == saved_v.sizes() && saved_g.numel() == saved_v.size(dim) &&
                        saved_norms.numel() == saved_v.size(dim),
                    "weight_norm backward expects grad_w shaped like v, and one g and one norm per slice");
  // MPS saves norms in v's dtype, unlike CPU and CUDA which keep float for half/bfloat16.
  TORCH_CHECK_TYPE(grad_w.scalar_type() == saved_v.scalar_type() && saved_g.scalar_type() == saved_v.scalar_type() &&
                       saved_norms.scalar_type() == saved_v.scalar_type(),
                   "weight_norm backward expects grad_w, v, g and norms of one dtype, got ",
                   grad_w.scalar_type(),
                   ", ",
                   saved_v.scalar_type(),
                   ", ",
                   saved_g.scalar_type(),
                   ", ",
                   saved_norms.scalar_type());
  auto grad_v = at::empty_like(saved_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_g = at::empty_like(saved_g, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (saved_v.numel() == 0) {
    grad_g.zero_();
    return {std::move(grad_v), std::move(grad_g)};
  }
  const auto slices = c10::safe_conv<uint32_t>(saved_v.size(dim), "uint32_t");
  const auto len = c10::safe_conv<uint32_t>(saved_v.numel() / saved_v.size(dim), "uint32_t");
  const auto dtype = scalarToMetalTypeString(saved_v);
  constexpr auto simd = c10::metal::simdgroup_size;
  const auto row_cfg = select_norm_row_config(slices, len, occupancy_threadgroups());
  const auto tiles = select_norm_column_config(len, slices, occupancy_threadgroups());
  const auto opmath_options = saved_v.options().dtype(at::toOpMathType(saved_v.scalar_type()));
  const auto row_part = dim == 0 && row_cfg.chunks > 1
      ? std::make_optional(at::empty({row_cfg.rows, row_cfg.chunks, 2}, opmath_options))
      : std::nullopt;
  const auto partial = dim == 0 ? Tensor() : at::empty({tiles.splits, slices}, opmath_options);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto run = [&](const std::string& kernel, MTLSize grid, MTLSize threads, const auto&... args) {
        auto pso = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(pso, kernel, {grad_w, saved_v, saved_g, saved_norms}, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, args...);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        getMPSProfiler().endProfileKernel(pso, stream);
      };
      if (dim == 0) {
        const std::array<uint32_t, 4> dims{row_cfg.rows, row_cfg.row_size, row_cfg.lanes, row_cfg.chunk_size};
        for (const auto pass : row_cfg.passes()) {
          run(fmt::format("weight_norm_backward_{}_{}", pass, dtype),
              MTLSizeMake(row_cfg.chunks * row_cfg.threads, row_cfg.row_groups, 1),
              MTLSizeMake(row_cfg.threads, 1, 1),
              grad_w,
              saved_v,
              saved_g,
              saved_norms,
              grad_v,
              row_part,
              dims,
              grad_g);
        }
      } else {
        const auto tile = MTLSizeMake(simd, simd, 1);
        const std::array<uint32_t, 2> rows_cols = {len, slices};
        const std::optional<Tensor> unused;
        run(fmt::format("weight_norm_backward_cols_dot_{}", dtype),
            MTLSizeMake(tiles.col_groups * tiles.tx, tiles.splits * tiles.ty, 1),
            MTLSizeMake(tiles.tx, tiles.ty, 1),
            grad_w,
            saved_v,
            unused,
            unused,
            partial,
            unused,
            unused,
            rows_cols);
        if (tiles.splits > 1) {
          run("weight_norm_backward_cols_sum",
              MTLSizeMake(slices, 1, 1),
              MTLSizeMake(simd, 1, 1),
              partial,
              std::array<uint32_t, 2>{tiles.splits, slices});
        }
        // Every thread of a tile writes simd rows of one contiguous block, so reads and writes stay close in memory.
        constexpr auto rows_per_tile = simd * simd;
        run(fmt::format("weight_norm_backward_cols_{}{}", dtype, mtlIdxSuffix(offsetsFitIn<uint32_t>(saved_v))),
            MTLSizeMake(slices, at::ceil_div(len, rows_per_tile) * simd, 1),
            tile,
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            grad_v,
            grad_g,
            partial,
            rows_cols,
            rows_per_tile);
      }
    }
  });
  return {std::move(grad_v), std::move(grad_g)};
}

} // namespace at::native
