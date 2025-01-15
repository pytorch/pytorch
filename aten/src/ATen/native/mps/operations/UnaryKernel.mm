#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/UnaryConstants.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/exp_native.h>
#include <ATen/ops/tanh_native.h>
#endif

#include <fmt/format.h>

namespace at::native {
static mps::MetalShaderLibrary lib(UNARY_KERNEL_TEMPLATE, 2);

static void exec_unary_kernel(const Tensor& self, const Tensor& output_, const std::string& name) {
  Tensor inputTensor = self.contiguous();
  Tensor outputTensor = output_;
  bool needs_output_copy = false;
  uint32_t length = output_.numel();
  if (length == 0) {
    return;
  }
  using namespace mps;
  @autoreleasepool {
    id<MTLComputePipelineState> cplState = nil;
    if (c10::isComplexType(self.scalar_type())) {
      auto scalarStr = self.scalar_type() == kComplexFloat ? "float" : "half";
      cplState = lib.getPipelineStateForFunc(name + "_complex_kernel", {scalarStr, scalarStr});
    } else {
      cplState = lib.getPipelineStateForFunc(name + "_kernel",
                                             {scalarToMetalTypeString(outputTensor), scalarToMetalTypeString(self)});
    }

    if (!outputTensor.is_contiguous()) {
      outputTensor = outputTensor.contiguous();
      needs_output_copy = true;
    }

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, name, {self});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setArgs(computeEncoder, outputTensor, inputTensor);
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(outputTensor);
  }
}
TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output_) {
  // handle erfinv ops using metal kernel
  // erfinv algorithm ported from aten/src/ATen/native/Math.h
  // https://github.com/pytorch/pytorch/blob/4154c8ea159fdaecc71ee9af820ac956193c875b/aten/src/ATen/native/Math.h#L152

  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support erfinv op with scalar type: Double");
  exec_unary_kernel(self, output_, "erfinv");
}

TORCH_IMPL_FUNC(exp_out_mps)(const Tensor& self, const Tensor& output_) {
  exec_unary_kernel(self, output_, "exp");
}
TORCH_IMPL_FUNC(tanh_out_mps)(const Tensor& self, const Tensor& output_) {
  exec_unary_kernel(self, output_, "tanh");
}
} // namespace at::native
