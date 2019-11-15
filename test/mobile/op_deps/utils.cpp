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
    static c10::OperatorHandle op = c10::Dispatcher::singleton()
        .findSchema({"aten::AA", ""}).value();
    return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor&>(
        op, self, self);
  };
  return lambda();
}
