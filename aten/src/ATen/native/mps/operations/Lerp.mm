#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/lerp_native.h>
#endif

namespace at::native {

namespace mps {

static const char* METAL_LERP_TENSOR = R"LERP_TENSOR_MTL(

#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void lerp(constant T* self       [[buffer(0)]],
                 constant T* end        [[buffer(1)]],
                 constant float* weight     [[buffer(2)]],
                 device T* output       [[buffer(3)]],
                 uint index [[thread_position_in_grid]]) {
    output[index] = abs(weight[index]) < 0.5 ?
        self[index] + weight[index] * (end[index] - self[index]) :
        end[index] - (end[index] - self[index]) * (1 - weight[index]);
}

template
[[host_name("lerp_float")]]
kernel void lerp<float>(constant float* self [[buffer(0)]],
                        constant float* end  [[buffer(1)]],
                        constant float* weight [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        uint index [[thread_position_in_grid]]);

template
[[host_name("lerp_half")]]
kernel void lerp<half>(constant half* self [[buffer(0)]],
                       constant half* end  [[buffer(1)]],
                       constant float* weight [[buffer(2)]],
                       device half* output [[buffer(3)]],
                       uint index [[thread_position_in_grid]]);
)LERP_TENSOR_MTL";

static id<MTLLibrary> compileLerpTensorLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> lerpTensorLibrary = nil;
  if (lerpTensorLibrary) {
    return lerpTensorLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  lerpTensorLibrary = [device newLibraryWithSource:[NSString stringWithCString:METAL_LERP_TENSOR
                                                                      encoding:NSASCIIStringEncoding]
                                           options:options
                                             error:&error];
  TORCH_CHECK(
      lerpTensorLibrary, "Failed to create metal lerp tensor library, error: ", [[error description] UTF8String]);
  return lerpTensorLibrary;
}

static id<MTLComputePipelineState> lerpTensorPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> lerpTensorLib = compileLerpTensorLibrary(device);
  id<MTLFunction> lerpTensorFunc = [lerpTensorLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(lerpTensorFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:lerpTensorFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

void lerp_tensor_mps(TensorIteratorBase& iter) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  Tensor self = iter.input(0);
  Tensor end = iter.input(1);
  Tensor weight = iter.input(2);
  Tensor out = iter.output();

  id<MTLBuffer> selfBuffer = getMTLBufferStorage(self);
  id<MTLBuffer> endBuffer = getMTLBufferStorage(end);
  id<MTLBuffer> weightBuffer = getMTLBufferStorage(weight);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(out);
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  MPSStream* mpsStream = getCurrentMPSStream();

  const uint32_t numThreads = self.numel();

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      const std::string kernel = "lerp_" + scalarToMetalTypeString(self.scalar_type());
      id<MTLComputePipelineState> lerpTensorPSO = lerpTensorPipelineState(device, kernel);
      [computeEncoder setComputePipelineState:lerpTensorPSO];
      [computeEncoder setBuffer:selfBuffer offset:self.storage_offset() * self.element_size() atIndex:0];
      [computeEncoder setBuffer:endBuffer offset:end.storage_offset() * end.element_size() atIndex:1];
      [computeEncoder setBuffer:weightBuffer offset:weight.storage_offset() atIndex:2];
      [computeEncoder setBuffer:outputBuffer offset:out.storage_offset() * out.element_size() atIndex:3];

      NSUInteger tgSize = lerpTensorPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(lerpTensorPSO);
    }
  });
}

} // namespace mps

void lerp_tensor_mps_kernel(TensorIteratorBase& iter) {
  mps::lerp_tensor_mps(iter);
}

REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_mps_kernel);
} // namespace at::native
