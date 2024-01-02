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

static const char* METAL_RENORM = R"RENORM_METAL(

#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void renorm(constant T* norm [[buffer(0)]],
                   device T* factor [[buffer(1)]],
                   constant float& maxnorm [[buffer(2)]],
                   uint index [[thread_position_in_grid]]) {
  constexpr T eps = 1e-7;
  constexpr T one = 1;
  factor[index] = norm[index] > maxnorm ? maxnorm / (norm[index] + eps) : one;
}

#define REGISTER_RENORM_OP(DTYPE)                                  \
template                                                           \
[[host_name("renorm_" #DTYPE)]]                                    \
kernel void renorm<DTYPE>(constant DTYPE* norm [[buffer(0)]],      \
                          device DTYPE* factor [[buffer(1)]],      \
                          constant float& maxnorm [[buffer(2)]],   \
                          uint index [[thread_position_in_grid]]);

REGISTER_RENORM_OP(float);
REGISTER_RENORM_OP(half);

)RENORM_METAL";

using namespace mps;

static id<MTLLibrary> compileRenormLibrary(id<MTLDevice> device, const std::string& key) {
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
stringWithCString:
  id<MTLLibrary> renormLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:METAL_RENORM]
                                                      options:options
                                                        error:&error];
  TORCH_CHECK(
      renormLibrary, "Failed to to create renorm mps kernel library, error: ", error.localizedDescription.UTF8String);

  libMap[key] = renormLibrary;
  return renormLibrary;
}

static id<MTLComputePipelineState> renormPipelineState(id<MTLDevice> device, const std::string& key) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[key];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> renormLib = compileRenormLibrary(device, key);
  id<MTLFunction> renormFunc = [renormLib newFunctionWithName:[NSString stringWithUTF8String:key.c_str()]];
  TORCH_CHECK(renormFunc, "Failed to create function state object for: ", key);
  pso = [device newComputePipelineStateWithFunction:renormFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[key] = pso;
  return pso;
}

void renorm_out_mps(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  Tensor norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims, /*keepdim=*/true);
  auto factor = at::empty(norm.sizes(), self.options());
  auto maxnorm_f = maxnorm.to<float>();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  id<MTLBuffer> normBuffer = getMTLBufferStorage(norm);
  id<MTLBuffer> factorBuffer = getMTLBufferStorage(factor);

  string key = "renorm_" + scalarToMetalTypeString(self.scalar_type());
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
  id<MTLComputePipelineState> renormPSO = renormPipelineState(device, key);

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      // this function call is a no-op if MPSProfiler is not enabled
      getMPSProfiler().beginProfileKernel(renormPSO, key, {norm});

      [computeEncoder setComputePipelineState:renormPSO];
      [computeEncoder setBuffer:normBuffer offset:norm.storage_offset() * norm.element_size() atIndex:0];
      [computeEncoder setBuffer:factorBuffer offset:factor.storage_offset() * factor.element_size() atIndex:1];
      [computeEncoder setBytes:&maxnorm_f length:sizeof(float) atIndex:2];

      uint32_t numThreads = norm.numel();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      NSUInteger tgSize = renormPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(renormPSO);
    }
  });
  at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

} // namespace mps

TORCH_IMPL_FUNC(renorm_out_mps)
(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  renorm_out_mps(self, p, dim, maxnorm, out);
}
} // namespace at::native
