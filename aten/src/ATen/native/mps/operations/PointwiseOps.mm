//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/BinaryKernel_metallib.h>
#endif

// value stays at float precision for reduced-float dtypes (CPU uses
// value.to<float>, CUDA an accscalar); quantizing it to half/bfloat first
// diverges (e.g. sub-denormal values silently become 0) and Scalar::to<Half>
// is a checked conversion that would throw for out-of-range values.
static c10::ScalarType addc_alpha_type(c10::ScalarType common) {
  return c10::isReducedFloatingType(common) ? c10::ScalarType::Float : common;
}

static void addcmul_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  lib.exec_ternary_kernel(iter, "addcmul", value, addc_alpha_type(iter.output().scalar_type()));
}

static void addcdiv_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  lib.exec_ternary_kernel(iter, "addcdiv", value, addc_alpha_type(iter.output().scalar_type()));
}

REGISTER_DISPATCH(addcmul_stub, addcmul_mps_kernel)
REGISTER_DISPATCH(addcdiv_stub, addcdiv_mps_kernel)

} // namespace at::native
