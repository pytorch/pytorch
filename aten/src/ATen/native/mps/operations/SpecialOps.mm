#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <ATen/native/TensorIterator.h>

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SpecialOps_metallib.h>
#endif

static void unary_kernel_mps(TensorIteratorBase& iter, const std::string& name) {
  using namespace mps;
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  auto input = iter.input();
  auto output = iter.output();
  bool needs_copy = !output.is_contiguous();
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }
  if (needs_copy) {
    output = output.contiguous();
  }
  auto i0PSO = lib.getPipelineStateForFunc(
      fmt::format("{}_{}_{}", name, scalarToMetalTypeString(input), scalarToMetalTypeString(output)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:i0PSO];
      mtl_setArgs(computeEncoder, input, output);
      mtl_dispatch1DJob(computeEncoder, i0PSO, output.numel());
    }
  });
  if (needs_copy) {
    iter.output().copy_(output);
  }
}

static void i0_kernel_mps(TensorIteratorBase& iter) {
  unary_kernel_mps(iter, "i0");
}

static void i1_kernel_mps(TensorIteratorBase& iter) {
  unary_kernel_mps(iter, "i1");
}

REGISTER_DISPATCH(i0_stub, &i0_kernel_mps)
REGISTER_DISPATCH(special_i1_stub, &i1_kernel_mps)
} // namespace at::native
