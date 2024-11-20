#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/renorm_native.h>
#endif

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RenormKernel_metallib.h>
#endif

void renorm_out_mps(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  Tensor norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims, /*keepdim=*/true);
  auto factor = at::empty(norm.sizes(), self.options());

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  id<MTLBuffer> normBuffer = getMTLBufferStorage(norm);
  id<MTLBuffer> factorBuffer = getMTLBufferStorage(factor);

  string key = "renorm_" + scalarToMetalTypeString(self);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
  id<MTLComputePipelineState> renormPSO = lib.getPipelineStateForFunc(key);

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      // this function call is a no-op if MPSProfiler is not enabled
      getMPSProfiler().beginProfileKernel(renormPSO, key, {norm});

      [computeEncoder setComputePipelineState:renormPSO];
      mtl_setBuffer(computeEncoder, norm, 0);
      mtl_setBuffer(computeEncoder, factor, 1);
      mtl_setBytes(computeEncoder, maxnorm.to<float>(), 2);
      mtl_dispatch1DJob(computeEncoder, renormPSO, norm.numel());

      getMPSProfiler().endProfileKernel(renormPSO);
    }
  });
  at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

} // namespace

TORCH_IMPL_FUNC(renorm_out_mps)
(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  renorm_out_mps(self, p, dim, maxnorm, out);
}
} // namespace at::native
