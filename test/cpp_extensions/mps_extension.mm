#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>

// this sample custom kernel is taken from:
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
static const char* CUSTOM_KERNEL = R"MPS_ADD_ARRAYS(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
)MPS_ADD_ARRAYS";

at::Tensor get_cpu_add_output(at::Tensor & cpu_input1, at::Tensor & cpu_input2) {
  return cpu_input1 + cpu_input2;
}

at::Tensor get_mps_add_output(at::Tensor & mps_input1, at::Tensor & mps_input2) {

  // smoke tests
  TORCH_CHECK(mps_input1.is_mps());
  TORCH_CHECK(mps_input2.is_mps());
  TORCH_CHECK(mps_input1.sizes() == mps_input2.sizes());

  using namespace at::native::mps;
  at::Tensor mps_output = at::empty_like(mps_input1);

  @autoreleasepool {
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    NSError *error = nil;
    size_t numThreads = mps_output.numel();
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource: [NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                              options: nil
                                                                error: &error];
    TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    id<MTLFunction> customFunction = [customKernelLibrary newFunctionWithName: @"add_arrays"];
    TORCH_CHECK(customFunction, "Failed to create function state object for the kernel");

    id<MTLComputePipelineState> kernelPSO = [device newComputePipelineStateWithFunction: customFunction error: &error];
    TORCH_CHECK(kernelPSO, error.localizedDescription.UTF8String);

    MPSStream* mpsStream = getCurrentMPSStream();

    dispatch_sync(mpsStream->queue(), ^() {
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState: kernelPSO];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input1) offset:0 atIndex:0];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input2) offset:0 atIndex:1];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_output) offset:0 atIndex:2];
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      // Calculate a thread group size.
      NSUInteger threadsPerGroupSize = std::min(kernelPSO.maxTotalThreadsPerThreadgroup, numThreads);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupSize, 1, 1);

      // Encode the compute command.
      [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadGroupSize];

    });
  }
  return mps_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_cpu_add_output", &get_cpu_add_output);
  m.def("get_mps_add_output", &get_mps_add_output);
}