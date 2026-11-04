#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_linalg_eigh.h>
#endif

#include <algorithm>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Eigh_metallib.h>
#endif

static void eigh_kernel_mps(const Tensor& eigenvalues,
                            const Tensor& eigenvectors,
                            const Tensor& infos,
                            bool upper,
                            bool compute_eigenvectors) {
  using namespace mps;

  if (eigenvectors.numel() == 0) {
    return;
  }

  const auto dtype = eigenvectors.scalar_type();
  const int64_t n = eigenvectors.size(-1);
  const int64_t batch = c10::multiply_integers(eigenvectors.sizes().slice(0, eigenvectors.dim() - 2));
  const int64_t elem_size = eigenvectors.element_size();
  const int64_t staging_bytes = n * n * elem_size;
  const int64_t tg_limit = static_cast<int64_t>([MPSDevice::getInstance()->device() maxThreadgroupMemoryLength]);
  const bool fits = (2 * staging_bytes) <= tg_limit;
  const bool unsupported_dtype = (dtype != kFloat && dtype != kComplexFloat);
  const bool too_small = (batch * n * n < 12288);

  if (unsupported_dtype || !fits || too_small) {
    if (!fits) {
      TORCH_WARN_ONCE("linalg.eigh: matrix too large to stage in MPS threadgroup memory (",
                      2 * staging_bytes,
                      " > ",
                      tg_limit,
                      " bytes); falling back to CPU.");
    }
    auto [L_cpu, V_cpu] = at::_linalg_eigh(eigenvectors.to(at::kCPU), upper ? "U" : "L", compute_eigenvectors);
    const_cast<Tensor&>(eigenvalues).copy_(L_cpu);
    if (compute_eigenvectors) {
      const_cast<Tensor&>(eigenvectors).copy_(V_cpu);
    }
    const_cast<Tensor&>(infos).zero_();
    return;
  }

  Tensor V = eigenvectors.reshape({batch, n, n});
  const bool W_direct = eigenvalues.is_contiguous() && eigenvalues.scalar_type() == c10::toRealValueType(dtype);
  const bool info_direct = infos.is_contiguous() && infos.scalar_type() == kInt;
  Tensor W = W_direct ? eigenvalues.reshape({batch, n})
                      : at::empty({batch, n}, eigenvectors.options().dtype(c10::toRealValueType(dtype)));
  Tensor info_b = info_direct ? infos.reshape({batch}) : at::empty({batch}, eigenvectors.options().dtype(kInt));

  EighParams params{static_cast<uint32_t>(n),
                    /*max_sweeps=*/80u,
                    /*tol=*/1e-6f,
                    compute_eigenvectors,
                    upper};

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("eigh_jacobi_{}", scalarToMetalTypeString(eigenvectors)));
      getMPSProfiler().beginProfileKernel(pso, "eigh_jacobi", {eigenvectors}, stream);
      [enc setComputePipelineState:pso];
      // V binds to both A (in) and Q (out): safe since all reads stage into
      // threadgroup memory before any device writeback.
      mtl_setArgs(enc, V, W, V, info_b, params);
      [enc setThreadgroupMemoryLength:n * n * elem_size atIndex:0];
      [enc setThreadgroupMemoryLength:(compute_eigenvectors ? n * n * elem_size : elem_size) atIndex:1];
      const NSUInteger simd = 32;
      const NSUInteger kMaxSimdGroups = 16;
      const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
      const NSUInteger nPairs = (n + 1) / 2;
      const NSUInteger wantSG = std::min<NSUInteger>(std::max<NSUInteger>(nPairs, 1), kMaxSimdGroups);
      NSUInteger tgs = std::min<NSUInteger>(maxThreads, wantSG * simd);
      [enc dispatchThreads:MTLSizeMake(tgs * batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  if (!W_direct) {
    const_cast<Tensor&>(eigenvalues).copy_(W.reshape(eigenvalues.sizes()));
  }
  if (!info_direct) {
    const_cast<Tensor&>(infos).copy_(info_b.reshape(infos.sizes()));
  }
}

} // namespace mps

REGISTER_DISPATCH(linalg_eigh_stub, mps::eigh_kernel_mps);

} // namespace at::native
