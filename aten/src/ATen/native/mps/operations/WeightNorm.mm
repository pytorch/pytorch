#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/mps/OperationUtils.h>
#include <bit>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/norm_except_dim.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

using namespace at::native::mps;

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
  TORCH_CHECK_NOT_IMPLEMENTED(
      saved_v.scalar_type() == kFloat || saved_v.scalar_type() == kHalf || saved_v.scalar_type() == kBFloat16,
      "weight_norm backward is not implemented for '",
      toString(saved_v.scalar_type()),
      "' on MPS");
  TORCH_CHECK(
      grad_w.is_contiguous() && saved_v.is_contiguous() && saved_g.is_contiguous() && saved_norms.is_contiguous(),
      "weight_norm backward expects contiguous tensors");
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
    return {grad_v, grad_g.zero_()};
  }
  const auto numel = c10::checked_convert<uint32_t>(saved_v.numel(), "uint32_t");
  const auto slices = c10::checked_convert<uint32_t>(saved_v.size(dim), "uint32_t");
  const auto len = numel / slices;
  const auto dtype = scalarToMetalTypeString(saved_v);
  const auto simd = c10::metal::simdgroup_size;
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto max_threads = [&](const std::string& kernel) {
        return static_cast<uint32_t>([lib.getPipelineStateForFunc(kernel) maxTotalThreadsPerThreadgroup]);
      };
      auto run = [&](const std::string& kernel, MTLSize groups, MTLSize threads, auto... args) {
        auto pso = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(pso, kernel, {grad_w, saved_v, saved_g, saved_norms}, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, args...);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        getMPSProfiler().endProfileKernel(pso, stream);
      };
      if (dim == 0) {
        const auto kernel = fmt::format("weight_norm_backward_rows_{}", dtype);
        const auto threads = simd * at::ceil_div(std::min(len, max_threads(kernel)), simd);
        run(kernel,
            MTLSizeMake(slices, 1, 1),
            MTLSizeMake(threads, 1, 1),
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            grad_v,
            grad_g,
            std::array<uint32_t, 2>{slices, len});
      } else {
        // Rows each thread accumulates per block. Fewer rows per thread give narrow
        // weights more threadgroups but more partials to sum; 32 measured best of 4..128.
        constexpr uint32_t rows_per_thread = 32;
        const auto dot_kernel = fmt::format("weight_norm_backward_cols_dot_{}", dtype);
        const auto kernel = fmt::format("weight_norm_backward_cols_{}", dtype);
        const std::string sum_kernel = "weight_norm_backward_cols_sum";
        // Tile width: a power of two up to simdgroup_size so narrow weights do not idle lanes.
        const auto tile_w = std::min(simd, static_cast<uint32_t>(std::bit_ceil(slices)));
        const auto tile_h = std::min({max_threads(dot_kernel), max_threads(sum_kernel), max_threads(kernel)}) / tile_w;
        const auto block = rows_per_thread * tile_h;
        const auto blocks = at::ceil_div(len, block);
        const std::array<uint32_t, 4> dims = {slices, len, block, blocks};
        auto partial = at::empty({blocks, slices}, saved_v.options().dtype(at::toOpMathType(saved_v.scalar_type())));
        const auto groups = MTLSizeMake(at::ceil_div(slices, tile_w), blocks, 1);
        const auto threads = MTLSizeMake(tile_w, tile_h, 1);
        run(dot_kernel, groups, threads, grad_w, saved_v, partial, dims);
        run(sum_kernel, MTLSizeMake(groups.width, 1, 1), threads, partial, dims);
        run(kernel, groups, threads, grad_w, saved_v, saved_g, saved_norms, grad_v, grad_g, partial, dims);
      }
    }
  });
  return {grad_v, grad_g};
}

} // namespace at::native
