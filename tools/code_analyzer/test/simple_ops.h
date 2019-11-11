#pragma once

#include <ATen/Tensor.h>

namespace at {

static inline Tensor call_AA_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::AA", "ignored"}).value();
  return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_BB_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::BB", "out"}).value();
  return c10::Dispatcher::singleton().callUnboxed<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_CC_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::CC", ""}).value();
  return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_DD_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::DD", ""}).value();
  return c10::Dispatcher::singleton().callUnboxed<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_EE_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::EE", ""}).value();
  return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_FF_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"aten::FF", ""}).value();
  return c10::Dispatcher::singleton().callUnboxed<Tensor, const Tensor&>(
      op, self, self);
}

} // namespace at
