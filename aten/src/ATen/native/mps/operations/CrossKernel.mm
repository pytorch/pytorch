//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Cross.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/CrossKernel_metallib.h>
#endif

void cross_mps_impl(const Tensor& out, const Tensor& input, const Tensor& other, int64_t dim) {
  TORCH_CHECK(input.dtype() != at::kDouble, "float64 is not supported on MPS");

  auto iter = TensorIteratorConfig()
                  .add_output(out)
                  .add_input(input)
                  .add_input(other)
                  .resize_outputs(false)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/dim)
                  .build();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const int64_t out_dim_stride = out.stride(dim);
  const int64_t input_dim_stride = input.stride(dim);
  const int64_t other_dim_stride = other.stride(dim);
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto kernelDataOffsets = generateKernelDataOffsets(computeEncoder, iter);

      auto crossPSO = lib.getPipelineStateForFunc("cross_" + scalarToMetalTypeString(out));

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(crossPSO, "cross", {input, other});

      [computeEncoder setComputePipelineState:crossPSO];
      mtl_setBuffer(computeEncoder, input, 0);
      mtl_setBuffer(computeEncoder, other, 1);
      mtl_setBuffer(computeEncoder, out, 2);
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      mtl_setBytes(computeEncoder, out_dim_stride, 4);
      mtl_setBytes(computeEncoder, input_dim_stride, 5);
      mtl_setBytes(computeEncoder, other_dim_stride, 6);
      mtl_dispatch1DJob(computeEncoder, crossPSO, numThreads);

      getMPSProfiler().endProfileKernel(crossPSO);
    }
  });
}
} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_mps_impl)
} // namespace at::native
