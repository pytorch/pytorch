#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSSleep.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/empty.h>

namespace at::mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = native::mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Sleep_metallib.h>
#endif

void sleep(int64_t cycles) {
  auto out = at::empty({1}, TensorOptions().device(kMPS).dtype(kLong));
  auto stream = getCurrentMPSStream();
  auto pso = lib.getPipelineStateForFunc("sleep");
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      native::mps::mtl_setArgs(encoder, out, cycles);
      native::mps::mtl_dispatch1DJob(encoder, pso, 1);
    }
  });
}

} // namespace at::mps
