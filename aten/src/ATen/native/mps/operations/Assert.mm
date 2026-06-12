#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/error.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_assert_async_native.h>
#endif

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Assert_metallib.h>
#endif
} // namespace mps

void _assert_async_msg_mps(const Tensor& self, std::string_view assert_msg) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");

  // The message rides into the kernel as a constant char* and is copied into
  // the error buffer's ErrorMessage::message slot (250 bytes) on failure.
  std::array<char, 250> msg{};
  TORCH_CHECK(assert_msg.length() < msg.size(), "Message length must be smaller than ", msg.size());
  std::copy_n(assert_msg.data(), assert_msg.length(), msg.begin());

  using namespace mps;
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    // Dispatch by element width; the kernel only checks whether the single
    // element's bits are zero (see Assert.metal).
    auto pso = lib.getPipelineStateForFunc("assert_async_" + std::to_string(self.element_size()));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, self, msg, stream->getErrorBuffer());
        mtl_dispatch1DJob(computeEncoder, pso, 1);
      }
    });
  }
}

void _assert_async_mps(const Tensor& self) {
  _assert_async_msg_mps(self, "");
}

} // namespace at::native
