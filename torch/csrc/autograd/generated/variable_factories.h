#pragma once

// @generated from tools/autograd/templates/variable_factories.h

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>

#include <functional>
#include <initializer_list>
#include <utility>

namespace torch {

#define TENSOR(T, S, _1)                                                   \
  inline at::Tensor tensor(                                                \
      at::ArrayRef<T> values, const at::TensorOptions& options) {          \
    at::Tensor result =                                                    \
        at::tensor(values, at::TensorOptions(options).is_variable(false)); \
    return autograd::make_variable(result, options.requires_grad());       \
  }                                                                        \
  inline at::Tensor tensor(                                                \
      std::initializer_list<T> values, const at::TensorOptions& options) { \
    return torch::tensor(at::ArrayRef<T>(values), options);                \
  }                                                                        \
  inline at::Tensor tensor(T value, const at::TensorOptions& options) {    \
    return torch::tensor(at::ArrayRef<T>(value), options);                 \
  }                                                                        \
  inline at::Tensor tensor(at::ArrayRef<T> values) {                       \
    return torch::tensor(std::move(values), at::dtype(at::k##S));          \
  }                                                                        \
  inline at::Tensor tensor(std::initializer_list<T> values) {              \
    return torch::tensor(at::ArrayRef<T>(values));                         \
  }                                                                        \
  inline at::Tensor tensor(T value) {                                      \
    return torch::tensor(at::ArrayRef<T>(value));                          \
  }
AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(TENSOR)
#undef TENSOR

/// A generic deleter function.
using Deleter = std::function<void(void*)>;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    at::IntList strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor =
      at::from_blob(data, sizes, strides, deleter, options.is_variable(false));
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    at::IntList strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(
      data,
      sizes,
      strides,
      /*deleter=*/[](void*) {},
      options);
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The `deleter`
/// (a `std::function<void(void*)>`) function will be called on the `data` when
/// the Tensor data would normally be deallocated. The `TensorOptions` specify
/// additional configuration options for the returned tensor, such as what type
/// to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor =
      at::from_blob(data, sizes, deleter, options.is_variable(false));
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(data, sizes, /*deleter=*/[](void*) {}, options);
}

inline at::Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_init_dropout_state");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "dropout_seed", dropout_seed);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::_cudnn_init_dropout_state(dropout, train, dropout_seed, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor arange(at::Scalar start, at::Scalar end, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::arange(start, end, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor arange(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::arange(start, end, step, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor arange(at::Scalar end, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::arange(end, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor bartlett_window(int64_t window_length, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::bartlett_window(window_length, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor bartlett_window(int64_t window_length, bool periodic, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::bartlett_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor blackman_window(int64_t window_length, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::blackman_window(window_length, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor blackman_window(int64_t window_length, bool periodic, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::blackman_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty(at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::empty(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty_like(const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::empty_like(self, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty_like(const at::Tensor & self, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::empty_like(self, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty_strided(at::IntList size, at::IntList stride, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::empty_strided(size, stride, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor eye(int64_t n, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::eye(n, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor eye(int64_t n, int64_t m, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::eye(n, m, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor full(at::IntList size, at::Scalar fill_value, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::full(size, fill_value, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor full_like(const at::Tensor & self, at::Scalar fill_value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::full_like(self, fill_value, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor full_like(const at::Tensor & self, at::Scalar fill_value, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::full_like(self, fill_value, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hann_window(int64_t window_length, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hann_window(window_length, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hann_window(int64_t window_length, bool periodic, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hann_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hamming_window(int64_t window_length, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hamming_window(window_length, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hamming_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hamming_window(window_length, periodic, alpha, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::hamming_window(window_length, periodic, alpha, beta, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor linspace(at::Scalar start, at::Scalar end, int64_t steps = 100, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::linspace(start, end, steps, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor logspace(at::Scalar start, at::Scalar end, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::logspace(start, end, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor logspace(at::Scalar start, at::Scalar end, int64_t steps, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::logspace(start, end, steps, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor ones(at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::ones(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor ones_like(const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::ones_like(self, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor ones_like(const at::Tensor & self, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::ones_like(self, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor scalar_tensor(at::Scalar s, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scalar_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::scalar_tensor(s, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::rand(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntList size, at::Generator * generator, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::rand(size, generator, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand_like(const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::rand_like(self, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand_like(const at::Tensor & self, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::rand_like(self, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t high, at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint(high, size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t high, at::IntList size, at::Generator * generator, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint(high, size, generator, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint(low, high, size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntList size, at::Generator * generator, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint(low, high, size, generator, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t high) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint_like(self, high, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint_like(self, low, high, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t high, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint_like(self, high, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randint_like(self, low, high, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randn(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntList size, at::Generator * generator, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randn(size, generator, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn_like(const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randn_like(self, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn_like(const at::Tensor & self, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randn_like(self, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randperm(int64_t n, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randperm(n, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randperm(int64_t n, at::Generator * generator, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::randperm(n, generator, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor range(at::Scalar start, at::Scalar end, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::range(start, end, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor range(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::range(start, end, step, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor zeros(at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::zeros(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor zeros_like(const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::zeros_like(self, self.options().is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor zeros_like(const at::Tensor & self, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::zeros_like(self, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor sparse_coo_tensor(at::IntList size, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::sparse_coo_tensor(size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::sparse_coo_tensor(indices, values, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::sparse_coo_tensor(indices, values, size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntList size, const at::TensorOptions & options = {}) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_unsafe");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::_sparse_coo_tensor_unsafe(indices, values, size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntList size, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntList size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor tril_indices(int64_t row, int64_t col, int64_t offset = 0, const at::TensorOptions & options = at::kLong) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tril_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::tril_indices(row, col, offset, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor triu_indices(int64_t row, int64_t col, int64_t offset = 0, const at::TensorOptions & options = at::kLong) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triu_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = at::triu_indices(row, col, offset, at::TensorOptions(options).is_variable(false));
  at::Tensor result =
    autograd::make_variable(tensor, /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

} // namespace torch
