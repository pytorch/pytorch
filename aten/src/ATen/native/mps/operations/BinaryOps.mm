//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sub_native.h>
#endif

namespace at::native {
namespace mps {

static void add_sub_lerp_template(const Tensor& self,
                                  const Tensor& other,
                                  const Scalar& alpha,
                                  const Tensor& output,
                                  std::string op_name) {
  if (!alpha.isComplex() && alpha.toDouble() == 0.0) {
    if (!self.is_alias_of(output)) { // if inplace, no-op
      output.copy_(self);
    }
    return;
  }

  const bool alpha_has_value = alpha.isComplex() || alpha.toDouble() != 1.0;
  if (!alpha_has_value && op_name == "lerp") {
    if (!self.is_alias_of(other)) { // if inplace, no-op
      output.copy_(other);
    }
    return;
  }

  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
    mps::binary_op_kernel(op_name + "_alpha", self, other, output, alpha);
  } else {
    mps::binary_op_kernel(op_name, self, other, output);
  }
}

} // namespace mps

TORCH_IMPL_FUNC(add_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_lerp_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_lerp_template(self, other, alpha, output, "sub");
}

TORCH_IMPL_FUNC(pow_Scalar_out_mps)(const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    out.fill_(1);
  } else {
    at::pow_out(const_cast<Tensor&>(out), mps::wrapped_scalar_tensor_mps(base, exp.device()), exp); // redispatch!
  }
}

} // namespace at::native
