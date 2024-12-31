#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSonomaOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/complex_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/nextafter_native.h>
#include <ATen/ops/polar_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/BinaryKernel_metallib.h>
#endif

static void binary_mps_impl(TensorIteratorBase& iter, const std::string func_name) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = func_name + "_" + scalarToMetalTypeString(input);
      auto kernelDataOffsets = generateKernelDataOffsets(computeEncoder, iter);

      id<MTLComputePipelineState> binaryPSO = lib.getPipelineStateForFunc(kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input, other});

      [computeEncoder setComputePipelineState:binaryPSO];
      mtl_setArgs(computeEncoder, input, other, out);
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      mtl_dispatch1DJob(computeEncoder, binaryPSO, numThreads);

      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}

void complex_mul_out(const Tensor& input, const Tensor& other, const Tensor& output) {
  TORCH_INTERNAL_ASSERT(c10::isComplexType(input.scalar_type()) || c10::isComplexType(other.scalar_type()));
  auto new_size = at::infer_size(input.sizes(), other.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }
  auto common_dtype = output.scalar_type();
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto input_as_real = at::view_as_real(input.to(kMPS, common_dtype)).select(input.dim(), 0);
  auto other_as_real = at::view_as_real(other.to(kMPS, common_dtype)).select(other.dim(), 0);
  auto iter =
      TensorIteratorConfig().add_output(output_as_real).add_input(input_as_real).add_input(other_as_real).build();

  mps::binary_mps_impl(iter, "complex_mul");
}

} // namespace mps

static void fmax_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    mps::binary_mps_impl(iter, "fmax");
  } else {
    at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}
static void fmin_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    mps::binary_mps_impl(iter, "fmin");
  } else {
    at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}

static void copysign_mps_kernel(TensorIteratorBase& iter) {
  mps::binary_mps_impl(iter, "copysign");
}

static void nextafter_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "nextafter_mps not implemented for non-floating types");
  mps::binary_mps_impl(iter, "nextafter_kernel");
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel)
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel)
REGISTER_DISPATCH(copysign_stub, &copysign_mps_kernel)
REGISTER_DISPATCH(nextafter_stub, &nextafter_mps_kernel)

Tensor& polar_out_mps(const Tensor& abs, const Tensor& angle, Tensor& output) {
  auto new_size = at::infer_size(abs.sizes(), angle.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return output;
  }
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto iter = TensorIteratorConfig().add_output(output_as_real).add_input(abs).add_input(angle).build();

  mps::binary_mps_impl(iter, "polar");
  return output;
}

Tensor& complex_out_mps(const Tensor& real, const Tensor& imag, Tensor& output) {
  auto new_size = at::infer_size(real.sizes(), imag.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return output;
  }
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto iter = TensorIteratorConfig().add_output(output_as_real).add_input(real).add_input(imag).build();

  mps::binary_mps_impl(iter, "complex_kernel");
  return output;
}
} // namespace at::native
