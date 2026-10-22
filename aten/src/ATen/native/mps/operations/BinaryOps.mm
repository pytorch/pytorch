//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#endif

namespace at::native {
// `add.out` is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs
// this entry; the body is the same one the generated `ufunc_add_CPU` has.
TORCH_IMPL_FUNC(add_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  add_stub(device_type(), *this, alpha);
}

TORCH_IMPL_FUNC(pow_Scalar_out_mps)(const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    out.fill_(1);
  } else {
    at::pow_out(const_cast<Tensor&>(out), mps::wrapped_scalar_tensor_mps(base, exp.device()), exp); // redispatch!
  }
}

} // namespace at::native
