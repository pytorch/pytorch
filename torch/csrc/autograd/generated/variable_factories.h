#pragma once

// @generated from tools/autograd/templates/variable_factories.h

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>

#include <functional>
#include <initializer_list>
#include <utility>

using at::Device;
using at::ScalarType;
using at::Layout;
using at::DimnameList;

namespace torch {

/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
///
/// NOTE: C++ `torch::tensor` by default gives a double tensor, which is
/// different from Python `torch.tensor` that gives a float tensor by default.
/// We are going to fix this discrepancy by making `torch::tensor` give
/// a float tensor by default.
/// Tracking issue: https://github.com/pytorch/pytorch/issues/28902
///
/// NOTE: C++ `torch::tensor` with an integer literal or a braced-init-list of
/// integer literals always produces a tensor of dtype `at::kLong` (aka. int64_t),
/// matching Python `torch.tensor` behavior.
///
/// NOTE: The following dtypes are not supported by `torch::tensor` currently:
/// - `unsigned int`
/// - `unsigned long int`
/// - `unsigned long long int`
/// - `long long int`
inline at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const at::TensorOptions& options = {}) {
  return autograd::make_variable(
    tensor_data_container.convert_to_tensor(options),
    options.requires_grad());
}

/// A generic deleter function.
using Deleter = std::function<void(void*)>;
using at::MemoryFormat;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, strides, deleter, options);
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
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
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, deleter, options);
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(data, sizes, /*deleter=*/[](void*) {}, options);
}

inline at::Tensor __cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__cudnn_init_dropout_state(dropout, train, dropout_seed, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::__cudnn_init_dropout_state(dropout, train, dropout_seed, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _arange(at::Scalar end, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    if (dtype.has_value()) {
      return at::_arange(end, dtype, layout, device, pin_memory);
    }
    return at::_arange(end, c10::nullopt, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor arange(at::Scalar end, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
  if (options.has_dtype()) {
    return torch::_arange(end, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  } else {
    return torch::_arange(end, c10::nullopt, options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  }
}
inline at::Tensor _arange(at::Scalar start, at::Scalar end, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    if (dtype.has_value()) {
      return at::_arange(start, end, dtype, layout, device, pin_memory);
    }
    return at::_arange(start, end, c10::nullopt, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor arange(at::Scalar start, at::Scalar end, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
  if (options.has_dtype()) {
    return torch::_arange(start, end, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  } else {
    return torch::_arange(start, end, c10::nullopt, options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  }
}
inline at::Tensor _arange(at::Scalar start, at::Scalar end, at::Scalar step, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    if (dtype.has_value()) {
      return at::_arange(start, end, step, dtype, layout, device, pin_memory);
    }
    return at::_arange(start, end, step, c10::nullopt, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor arange(at::Scalar start, at::Scalar end, at::Scalar step, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
  if (options.has_dtype()) {
    return torch::_arange(start, end, step, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  } else {
    return torch::_arange(start, end, step, c10::nullopt, options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
  }
}
inline at::Tensor _bartlett_window(int64_t window_length, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_bartlett_window(window_length, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor bartlett_window(int64_t window_length, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_bartlett_window(window_length, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _bartlett_window(int64_t window_length, bool periodic, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_bartlett_window(window_length, periodic, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor bartlett_window(int64_t window_length, bool periodic, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_bartlett_window(window_length, periodic, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _blackman_window(int64_t window_length, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_blackman_window(window_length, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor blackman_window(int64_t window_length, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_blackman_window(window_length, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _blackman_window(int64_t window_length, bool periodic, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_blackman_window(window_length, periodic, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor blackman_window(int64_t window_length, bool periodic, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_blackman_window(window_length, periodic, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _empty(at::IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty(size, names, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor empty(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_empty(size, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _empty(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty(size, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_empty(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor __empty_affine_quantized(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt, double scale = 1, int64_t zero_point = 0, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_empty_affine_quantized");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _empty_affine_quantized(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt, double scale = 1, int64_t zero_point = 0, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
    return torch::__empty_affine_quantized(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), scale, zero_point, memory_format);
}
inline at::Tensor __empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_empty_per_channel_affine_quantized");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
    return torch::__empty_per_channel_affine_quantized(size, scales, zero_points, axis, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor empty_like(const at::Tensor & self, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_like(self, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _empty_like(const at::Tensor & self, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_like(self, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor empty_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_empty_like(self, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_strided(size, stride, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_empty_strided(size, stride, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _eye(int64_t n, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_eye(n, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor eye(int64_t n, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_eye(n, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _eye(int64_t n, int64_t m, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_eye(n, m, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor eye(int64_t n, int64_t m, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_eye(n, m, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _full(at::IntArrayRef size, at::Scalar fill_value, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_full(size, fill_value, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor full(at::IntArrayRef size, at::Scalar fill_value, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_full(size, fill_value, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _full(at::IntArrayRef size, at::Scalar fill_value, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_full(size, fill_value, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor full(at::IntArrayRef size, at::Scalar fill_value, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_full(size, fill_value, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor full_like(const at::Tensor & self, at::Scalar fill_value, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_full_like(self, fill_value, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _full_like(const at::Tensor & self, at::Scalar fill_value, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_full_like(self, fill_value, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor full_like(const at::Tensor & self, at::Scalar fill_value, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_full_like(self, fill_value, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _from_file(std::string filename, c10::optional<bool> shared = c10::nullopt, c10::optional<int64_t> size = 0, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::from_file");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "filename", filename);
    jit::tracer::addInputs(node, "shared", shared);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_from_file(filename, shared, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor from_file(std::string filename, c10::optional<bool> shared = c10::nullopt, c10::optional<int64_t> size = 0, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_from_file(filename, shared, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hann_window(int64_t window_length, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hann_window(window_length, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hann_window(int64_t window_length, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hann_window(window_length, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hann_window(int64_t window_length, bool periodic, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hann_window(window_length, periodic, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hann_window(int64_t window_length, bool periodic, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hann_window(window_length, periodic, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hamming_window(int64_t window_length, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hamming_window(window_length, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hamming_window(int64_t window_length, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hamming_window(window_length, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hamming_window(int64_t window_length, bool periodic, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hamming_window(window_length, periodic, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hamming_window(int64_t window_length, bool periodic, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hamming_window(window_length, periodic, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hamming_window(int64_t window_length, bool periodic, double alpha, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hamming_window(window_length, periodic, alpha, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hamming_window(window_length, periodic, alpha, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _hamming_window(int64_t window_length, bool periodic, double alpha, double beta, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_hamming_window(window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_hamming_window(window_length, periodic, alpha, beta, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _linspace(at::Scalar start, at::Scalar end, int64_t steps = 100, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_linspace(start, end, steps, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor linspace(at::Scalar start, at::Scalar end, int64_t steps = 100, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_linspace(start, end, steps, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _logspace(at::Scalar start, at::Scalar end, int64_t steps = 100, double base = 10.0, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_logspace(start, end, steps, base, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor logspace(at::Scalar start, at::Scalar end, int64_t steps = 100, double base = 10.0, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_logspace(start, end, steps, base, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _ones(at::IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_ones(size, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor ones(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_ones(size, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _ones(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_ones(size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor ones(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_ones(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor ones_like(const at::Tensor & self, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_ones_like(self, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _ones_like(const at::Tensor & self, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_ones_like(self, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor ones_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_ones_like(self, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _scalar_tensor(at::Scalar s, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scalar_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_scalar_tensor(s, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor scalar_tensor(at::Scalar s, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_scalar_tensor(s, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _rand(at::IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand(size, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor rand(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_rand(size, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _rand(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand(size, generator, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor rand(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_rand(size, generator, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _rand(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand(size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor rand(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_rand(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _rand(at::IntArrayRef size, at::Generator * generator, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand(size, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor rand(at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_rand(size, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor rand_like(const at::Tensor & self, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand_like(self, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _rand_like(const at::Tensor & self, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_rand_like(self, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor rand_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_rand_like(self, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _randint(int64_t high, at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint(high, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint(int64_t high, at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randint(high, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randint(int64_t high, at::IntArrayRef size, at::Generator * generator, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint(high, size, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint(int64_t high, at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randint(high, size, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint(low, high, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randint(low, high, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randint(int64_t low, int64_t high, at::IntArrayRef size, at::Generator * generator, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint(low, high, size, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randint(low, high, size, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t high, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint_like(self, high, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint_like(self, low, high, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _randint_like(const at::Tensor & self, int64_t high, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint_like(self, high, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint_like(const at::Tensor & self, int64_t high, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_randint_like(self, high, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _randint_like(const at::Tensor & self, int64_t low, int64_t high, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randint_like(self, low, high, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_randint_like(self, low, high, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _randn(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn(size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randn(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randn(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randn(at::IntArrayRef size, at::Generator * generator, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn(size, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randn(at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randn(size, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randn(at::IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn(size, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randn(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randn(size, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randn(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn(size, generator, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randn(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randn(size, generator, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor randn_like(const at::Tensor & self, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn_like(self, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _randn_like(const at::Tensor & self, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randn_like(self, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randn_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_randn_like(self, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _randperm(int64_t n, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randperm(n, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randperm(int64_t n, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randperm(n, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _randperm(int64_t n, at::Generator * generator, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_randperm(n, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor randperm(int64_t n, at::Generator * generator, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_randperm(n, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _range(at::Scalar start, at::Scalar end, at::Scalar step = 1, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_range(start, end, step, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor range(at::Scalar start, at::Scalar end, at::Scalar step = 1, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_range(start, end, step, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _range(at::Scalar start, at::Scalar end, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_range(start, end, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor range(at::Scalar start, at::Scalar end, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_range(start, end, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _zeros(at::IntArrayRef size, c10::optional<DimnameList> names, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_zeros(size, names, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor zeros(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_zeros(size, names, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _zeros(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_zeros(size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor zeros(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_zeros(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor zeros_like(const at::Tensor & self, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_zeros_like(self, at::typeMetaToScalarType(self.options().dtype()), self.options().layout(), self.options().device(), self.options().pinned_memory(), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _zeros_like(const at::Tensor & self, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_zeros_like(self, dtype, layout, device, pin_memory, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor zeros_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
    return torch::_zeros_like(self, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad(), memory_format);
}
inline at::Tensor _sparse_coo_tensor(at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor(size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor sparse_coo_tensor(at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_sparse_coo_tensor(size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor(indices, values, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_sparse_coo_tensor(indices, values, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor(indices, values, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_sparse_coo_tensor(indices, values, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor __sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__sparse_coo_tensor_unsafe(indices, values, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::__sparse_coo_tensor_unsafe(indices, values, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor __sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::__sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor __sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, at::ScalarType dtype, at::Layout layout, at::Device device, bool pin_memory = false, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::__sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _tril_indices(int64_t row, int64_t col, int64_t offset = 0, c10::optional<ScalarType> dtype = at::kLong, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_tril_indices(row, col, offset, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor tril_indices(int64_t row, int64_t col, int64_t offset = 0, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_tril_indices(row, col, offset, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _triu_indices(int64_t row, int64_t col, int64_t offset = 0, c10::optional<ScalarType> dtype = at::kLong, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_triu_indices(row, col, offset, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor triu_indices(int64_t row, int64_t col, int64_t offset = 0, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_triu_indices(row, col, offset, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}
inline at::Tensor _normal(double mean, double std, at::IntArrayRef size, at::Generator * generator = nullptr, c10::optional<ScalarType> dtype = c10::nullopt, c10::optional<Layout> layout = c10::nullopt, c10::optional<Device> device = c10::nullopt, c10::optional<bool> pin_memory = c10::nullopt, c10::optional<bool> requires_grad = c10::nullopt) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "requires_grad", requires_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_normal(mean, std, size, generator, dtype, layout, device, pin_memory);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

inline at::Tensor normal(double mean, double std, at::IntArrayRef size, at::Generator * generator = nullptr, const at::TensorOptions & options={}, c10::optional<bool> requires_grad = c10::nullopt) {
    return torch::_normal(mean, std, size, generator, at::typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory(), options.requires_grad());
}

} // namespace torch
