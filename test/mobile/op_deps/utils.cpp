#include "utils.h"

#include <c10/core/TensorOptions.h>
#include <ATen/core/op_registration/op_registration.h>

#include "simple_ops.h"
using namespace at;

Tensor global_helper_call_AA_op_1(const Tensor& self) {
  auto lambda = [&]() {
    return call_AA_op(self);
  };
  return lambda();
}

static std::function<Tensor()> helper(const Tensor& self) {
  return [&]() {
    return call_AA_op(self);
  };
}

Tensor global_helper_call_AA_op_2(const Tensor& self) {
  return helper(self)();
}

Tensor global_helper_call_AA_op_3(const Tensor& self) {
  auto lambda = [&]() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("_test::AA", "")
        .typed<Tensor (const Tensor&)>();
    return op.call(self);
  };
  return lambda();
}

namespace torch {
namespace jit {

C10_EXPORT Tensor API_Function(const Tensor& self) {
  return call_AA_op(self);
}

at::Tensor API_Class::API_Method(const at::Tensor& self) {
  return call_BB_op(self);
}

}  // namespace jit
}  // namespace torch
