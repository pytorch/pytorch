#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/roll_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Roll_metallib.h>
#endif

Tensor roll_mps(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  TORCH_CHECK(!shifts.empty(), "`shifts` required");

  // Match the CPU/CUDA contract: empty dims with single shift means
  // flatten + roll along the flat axis, then restore shape.
  if (dims.empty() && shifts.size() == 1) {
    auto flat = self.contiguous().view({self.numel()});
    return roll_mps(flat, shifts, IntArrayRef{0}).view(self.sizes());
  }

  TORCH_CHECK(shifts.size() == dims.size(),
              "shifts and dimensions must align. shifts: ",
              shifts.size(),
              ", dims:",
              dims.size());

  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Preserve);
  }

  Tensor input = self.contiguous();
  // Output is contiguous in the same layout as `input`. Allocate fresh; we
  // never write through a view here so memory_format=Contiguous is safe.
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  const int64_t ndim_i64 = input.dim();
  TORCH_CHECK(ndim_i64 <= 16, "roll on MPS supports tensors with at most 16 dimensions, got ", ndim_i64);
  // Let maybe_wrap_dim below raise the device-consistent IndexError for the
  // 0-d-input + non-empty-dims case (matching CPU/CUDA).
  const uint32_t ndim = static_cast<uint32_t>(ndim_i64);

  // Accumulate per-dim shifts: callers may pass the same dim multiple times
  // (e.g. dims=[0, 0]); semantically those compose modulo the dim size.
  std::vector<int64_t> sizes(ndim);
  std::vector<int64_t> in_strides(ndim);
  std::vector<int64_t> out_strides(ndim);
  std::vector<int64_t> per_dim_shifts(ndim, 0);
  for (uint32_t d = 0; d < ndim; ++d) {
    sizes[d] = input.size(d);
    in_strides[d] = input.stride(d);
    out_strides[d] = output.stride(d);
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    // wrap_scalar=false so 0-d inputs hit the same IndexError as CPU/CUDA.
    int64_t wrapped = maybe_wrap_dim(dims[i], ndim_i64, /*wrap_scalar=*/false);
    int64_t sz = sizes[wrapped];
    int64_t s = shifts[i] % sz;
    if (s < 0)
      s += sz;
    per_dim_shifts[wrapped] = (per_dim_shifts[wrapped] + s) % sz;
  }

  const uint64_t numel = static_cast<uint64_t>(input.numel());
  const std::string type_suffix = scalarToMetalTypeString(input);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();

      if (ndim == 2) {
        // 2D fast path: row-major contiguous, no per-dim loop, 2D grid.
        const std::string key2d = "roll_2d_" + type_suffix;
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key2d);
        getMPSProfiler().beginProfileKernel(pso, "roll_2d:" + type_suffix, false);
        std::array<int64_t, 2> sizes2{sizes[0], sizes[1]};
        std::array<int64_t, 2> shifts2{per_dim_shifts[0], per_dim_shifts[1]};
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, input, output, sizes2, shifts2);
        const auto grid_x = static_cast<NSUInteger>(sizes[1]);
        const auto grid_y = static_cast<NSUInteger>(sizes[0]);
        const auto maxTg = [pso maxTotalThreadsPerThreadgroup];
        const NSUInteger tg_x = std::min(grid_x, maxTg);
        const NSUInteger tg_y = std::min(grid_y, std::max(maxTg / tg_x, NSUInteger{1}));
        [encoder dispatchThreads:MTLSizeMake(grid_x, grid_y, 1) threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
        getMPSProfiler().endProfileKernel(pso);
      } else {
        const std::string key = "roll_" + type_suffix;
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);
        getMPSProfiler().beginProfileKernel(pso, "roll:" + type_suffix, false);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, input, output, sizes, in_strides, out_strides, per_dim_shifts, ndim, numel);
        mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(numel));
        getMPSProfiler().endProfileKernel(pso);
      }
    }
  });

  return output;
}

} // namespace at::native
