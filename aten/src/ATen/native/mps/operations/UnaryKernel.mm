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
#endif

#include <fmt/format.h>

namespace at::native {
static const std::string& getMetalType(const c10::ScalarType& t) {
  // Mapping from c10::ScalarType to integral type that can be used for unary ops
  static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
      {c10::ScalarType::Half, "half"},
      {c10::ScalarType::Float, "float"},
      {c10::ScalarType::Long, "long"},
      {c10::ScalarType::Int, "int"},
      {c10::ScalarType::Short, "short"},
      {c10::ScalarType::Bool, "bool"},
      {c10::ScalarType::Char, "int8_t"},
      {c10::ScalarType::Byte, "uint8_t"},
  };

  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

static const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}

static const std::string& getMetalType(const Tensor& t) {
  return getMetalType(t.scalar_type());
}

static mps::MetalShaderLibrary lib(UNARY_KERNEL_TEMPLATE, 2);

TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output_) {
  // handle erfinv ops using metal kernel
  // erfinv algorithm ported from aten/src/ATen/native/Math.h
  // https://github.com/pytorch/pytorch/blob/4154c8ea159fdaecc71ee9af820ac956193c875b/aten/src/ATen/native/Math.h#L152

  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support erfinv op with scalar type: Double");

  Tensor inputTensor = self;
  Tensor outputTensor = output_;
  bool needs_output_copy = false;
  uint32_t length = output_.numel();
  if (length == 0) {
    return;
  }
  using namespace mps;
  @autoreleasepool {
    auto cplState = lib.getPipelineStateForFunc("erfinv_mps_kernel", {getMetalType(outputTensor), getMetalType(self)});

    if (!self.is_contiguous()) {
      inputTensor = inputTensor.contiguous();
      outputTensor = outputTensor.contiguous();
      needs_output_copy = true;
    }

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, "erf_inv", {inputTensor});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setBuffer(computeEncoder, outputTensor, 0);
      mtl_setBuffer(computeEncoder, inputTensor, 1);
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(outputTensor);
  }
}
} // namespace at::native
