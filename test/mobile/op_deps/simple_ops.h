#pragma once

#include <ATen/Tensor.h>

namespace at {

static inline Tensor call_AA_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::AA", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_BB_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::BB", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_CC_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::CC", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_DD_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::DD", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_EE_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::EE", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

static inline Tensor call_FF_op(const Tensor& self) {
  static c10::OperatorHandle op = c10::Dispatcher::singleton()
      .findSchema({"_test::FF", ""}).value();
  return c10::Dispatcher::singleton().call<Tensor, const Tensor&>(
      op, self, self);
}

} // namespace at
