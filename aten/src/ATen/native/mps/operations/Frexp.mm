#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/frexp_native.h>
#endif

namespace at::native {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Frexp_metallib.h>
#endif

static void frexp_kernel_mps(TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 3, "frexp expects 3 tensors (2 outputs + 1 input)");

  Tensor mantissa = iter.output(0);
  Tensor exponent = iter.output(1);
  const Tensor& self = iter.input(0);

  if (self.numel() == 0) {
    return;
  }

  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "frexp_mps: only floating-point types are supported, got ", self.scalar_type());

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      std::string kernel_name = fmt::format("frexp_kernel_{}", scalarToMetalTypeString(self));
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel_name);

      [computeEncoder setComputePipelineState:pso];

      Tensor self_contiguous = self.contiguous();
      Tensor mantissa_contiguous = mantissa.contiguous();
      Tensor exponent_contiguous = exponent.contiguous();

      mtl_setBuffer(computeEncoder, self_contiguous, 0);
      mtl_setBuffer(computeEncoder, mantissa_contiguous, 1);
      mtl_setBuffer(computeEncoder, exponent_contiguous, 2);

      uint64_t numThreads = self.numel();
      mtl_dispatch1DJob(computeEncoder, pso, numThreads);

      if (!mantissa.is_contiguous()) {
        mantissa.copy_(mantissa_contiguous);
      }
      if (!exponent.is_contiguous()) {
        exponent.copy_(exponent_contiguous);
      }
    }
  });
}

REGISTER_DISPATCH(frexp_stub, &frexp_kernel_mps);

} // namespace at::native
