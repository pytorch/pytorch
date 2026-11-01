//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/PointwiseOps_metallib.h>
#endif

// `value` is handed over at opmath precision, matching the CPU kernels: a
// half/bfloat16 alpha would both lose accuracy and be unable to represent the
// values `addcmul` accepts.
static void addcmul_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  TORCH_CHECK_NOT_IMPLEMENTED(iter.common_dtype() != kBool, "addcmul_mps not implemented for 'Bool'");
  lib.exec_ternary_kernel(iter, "addcmul", value, at::toOpMathType(iter.common_dtype()));
}

static void addcdiv_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  lib.exec_ternary_kernel(iter, "addcdiv", value, at::toOpMathType(iter.common_dtype()));
}

REGISTER_DISPATCH(addcmul_stub, &addcmul_mps_kernel)
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_mps_kernel)

} // namespace at::native
