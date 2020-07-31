#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/Tensor.h>

namespace at {

static inline Tensor call_AA_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::AA", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

static inline Tensor call_BB_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::BB", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

static inline Tensor call_CC_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::CC", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

static inline Tensor call_DD_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::DD", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

static inline Tensor call_EE_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::EE", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

static inline Tensor call_FF_op(const Tensor& self) {
  static const auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_test::FF", "")
      .typed<Tensor(const Tensor&)>();
  return op.call(self);
}

} // namespace at
