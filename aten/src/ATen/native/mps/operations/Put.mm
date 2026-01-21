#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/put_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Put_metallib.h>
#endif

namespace mps {

static void put_kernel_mps_impl(
    TensorIterator& iter,
    const TensorBase& self,
    const bool accumulate) {
  using namespace mps;

  if (iter.numel() == 0) {
    return;
  }

  // Get the source tensor and index tensor from the iterator
  // iter has 2 inputs: source, index (reshaped)
  const auto& source = iter.tensor(0);
  const auto& index = iter.tensor(1);

  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "put_(): Expected a long tensor for index, but got ",
      index.scalar_type());

  // Make contiguous copies
  auto source_contiguous = source.contiguous();
  auto index_contiguous = index.contiguous();
  
  // Self needs to be flattened conceptually, but we operate on it directly
  auto self_contiguous = self.contiguous();
  
  MPSStream* stream = getCurrentMPSStream();
  
  @autoreleasepool {
    id<MTLComputePipelineState> pso = nil;
    
    std::string scalar_type = scalarToMetalTypeString(source);
    
    if (accumulate) {
      // Use AtomicType-based accumulate kernel for all types
      pso = lib.getPipelineStateForFunc("put_accumulate_" + scalar_type);
    } else {
      pso = lib.getPipelineStateForFunc("put_" + scalar_type);
    }
    
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pso];
    
    // Set buffers
    // Buffer 0: output (self)
    mtl_setBuffer(encoder, self_contiguous, 0);
    // Buffer 1: source
    mtl_setBuffer(encoder, source_contiguous, 1);
    // Buffer 2: indices
    mtl_setBuffer(encoder, index_contiguous, 2);
    // Buffer 3: numel (number of elements to put)
    int64_t numel = iter.numel();
    [encoder setBytes:&numel length:sizeof(int64_t) atIndex:3];
    // Buffer 4: output_numel (size of self)
    int64_t output_numel = self.numel();
    [encoder setBytes:&output_numel length:sizeof(int64_t) atIndex:4];
    
    // Dispatch - parallel execution for all modes (AtomicType handles all types)
    mtl_dispatch1DJob(encoder, pso, numel);
    
    stream->synchronize(SyncType::COMMIT);
  }
}

} // namespace mps

void put_kernel_mps(
    TensorIterator& iter,
    const TensorBase& self,
    const bool accumulate) {
  mps::put_kernel_mps_impl(iter, self, accumulate);
}

REGISTER_DISPATCH(put_stub, &put_kernel_mps)

} // namespace at::native
