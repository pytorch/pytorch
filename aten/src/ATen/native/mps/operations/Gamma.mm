#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#include <ATen/ops/digamma_native.h>
#include <ATen/ops/lgamma_native.h>
#include <ATen/ops/polygamma_native.h>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Gamma_metallib.h>
#endif

static id<MTLComputePipelineState> getCPLState(const Tensor& t1, const Tensor& t2, const std::string& fname) {
  return lib.getPipelineStateForFunc(
      fmt::format("{}_{}_{}", fname, scalarToMetalTypeString(t1), scalarToMetalTypeString(t2)));
}

} // namespace mps

TORCH_IMPL_FUNC(lgamma_out_mps)(const Tensor& self, const Tensor& output_) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support lgamma_out op with scalar type: Double");

  Tensor output = output_;
  bool needs_output_copy = false;
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }

  if (mps::needsGather(output_)) {
    output = output.contiguous();
    needs_output_copy = true;
  }

  using namespace mps;

  @autoreleasepool {
    auto cplState = getCPLState(self, output, "lgamma");

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, "lgamma_out", {self});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setArgs(computeEncoder, self, output);
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(output);
  }
}

TORCH_IMPL_FUNC(digamma_out_mps)(const Tensor& self, const Tensor& output_) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support digamma_out op with scalar type: Double");

  Tensor output = output_;
  bool needs_output_copy = false;
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }

  if (mps::needsGather(output_)) {
    output = output.contiguous();
    needs_output_copy = true;
  }

  using namespace mps;

  @autoreleasepool {
    id<MTLComputePipelineState> cplState = getCPLState(self, output, "digamma");

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, "digamma_out", {self});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setArgs(computeEncoder, self, output);
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(output);
  }
}

TORCH_IMPL_FUNC(polygamma_out_mps)(const int64_t order, const Tensor& self, const Tensor& output_) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Double,
              "MPS does not support polygamma_out op with scalar type: Double");
  TORCH_CHECK(order >= 0, "Polygamma is implemented only for nonnegative real numbers");

  Tensor output = output_;
  bool needs_output_copy = false;
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }

  if (mps::needsGather(output_)) {
    output = output.contiguous();
    needs_output_copy = true;
  }

  using namespace mps;

  std::string func_name;

  if (order == 0) {
    func_name = "digamma";
  } else if (order == 1) {
    func_name = "trigamma";
  } else {
    func_name = "polygamma";
  }

  @autoreleasepool {
    id<MTLComputePipelineState> cplState = getCPLState(self, output, func_name);

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, func_name, {self});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setArgs(computeEncoder, self, output);

      if (func_name == "polygamma") {
        mtl_setBytes(computeEncoder, order, 2);
      }

      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(output);
  }
}

} // namespace at::native
