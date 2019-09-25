#pragma once

// @generated from tools/autograd/templates/variable_factories.h

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <ATen/core/EnableNamedTensor.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>

#include <functional>
#include <initializer_list>
#include <utility>

#ifdef BUILD_NAMEDTENSOR
using at::DimnameList;
#endif

namespace torch {

namespace detail {
  enum class InitListTensorType { Scalar, InitList };

  // We use `InitListTensor` to support converting an arbitrarily nested braced-init-list
  // (e.g. {{1, 2}, {3, 4}}) into the equivalent Tensor, taking advantage of the fact that
  // the constructor will automatically be called recursively until it reaches all innermost
  // scalar values.
  //
  // At any time, a `InitListTensor` object represents either of the following:
  // 1. A scalar with value `scalar()` and type `scalar_type()`.
  // 2. A Tensor represented in `std::initializer_list<InitListTensor>` form, with value
  //    `init_list()`, Tensor scalar type `scalar_type()`, and Tensor sizes `sizes()`.
  struct InitListTensor {
#define TENSOR(T, S)                   \
    InitListTensor(T scalar) :         \
        scalar_(scalar), init_list_(), \
        sizes_(),                      \
        scalar_type_(at::k##S),        \
        type_(InitListTensorType::Scalar) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR
    InitListTensor(std::initializer_list<InitListTensor> init_list) :
        scalar_(),
        init_list_(init_list),
        sizes_(),
        scalar_type_(),
        type_(InitListTensorType::InitList) {
      TORCH_CHECK(init_list.size() > 0, "Empty init-list is not supported");
      scalar_type_ = init_list.begin()->scalar_type_;
      const InitListTensor& first_elem = *(init_list.begin());
      for (const auto& elem : init_list) {
        TORCH_CHECK(elem.scalar_type_ == first_elem.scalar_type_,
          "Expected all elements of the tensor to have the same scalar type: ",
          first_elem.scalar_type_,
          ", but got element of scalar type: ",
          elem.scalar_type_);
        TORCH_CHECK(elem.sizes_ == first_elem.sizes_,
          "Expected all sub-lists to have sizes: ",
          first_elem.sizes_,
          " (e.g. ", first_elem, "), ",
          "but got sub-list ",
          elem,
          " with sizes: ",
          elem.sizes_);
      }
      sizes_.reserve(first_elem.sizes_.size() + 1);
      sizes_.push_back(init_list.size());
      sizes_.insert(sizes_.end(), first_elem.sizes_.begin(), first_elem.sizes_.end());
    }

    const c10::Scalar& scalar() const {
      return scalar_;
    }

    const std::initializer_list<InitListTensor>& init_list() const {
      return init_list_;
    }

    const std::vector<int64_t>& sizes() const {
      return sizes_;
    }

    const c10::ScalarType& scalar_type() const {
      return scalar_type_;
    }

    const InitListTensorType& type() const {
      return type_;
    }

    at::Tensor to_tensor(const at::TensorOptions& options) const {
      // NOTE: Here we explicitly choose to initialize the tensor on CPU first,
      // fill each element of the tensor, and then move the tensor to the desired
      // device. For CUDA device, this approach only involves 1 CUDA kernel launch,
      // and is much faster than initializing the tensor on CUDA first and then
      // filling each element of it (which involves `N` CUDA kernel launches where
      // `N` is the number of the elements in the tensor).
      at::Tensor tensor = ([&]() {
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        return at::empty(sizes_, at::TensorOptions(options).device(at::kCPU).is_variable(false));
      })();
      fill_tensor(tensor);
      return tensor.to(options.device());
    }

    void pretty_print_recursive(std::ostream& stream) const {
      if (type_ == InitListTensorType::Scalar) {
        AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "InitListTensor_pretty_print_scalar", [&] {
          stream << scalar_.to<scalar_t>();
        });
      } else if (type_ == InitListTensorType::InitList) {
        stream << "{";
        for (const InitListTensor* it = init_list_.begin(); it != init_list_.end(); it++) {
          it->pretty_print_recursive(stream);
          if (std::next(it) != init_list_.end()) stream << ", ";
        }
        stream << "}";
      }
    }

   private:
    void fill_tensor(at::Tensor tensor) const {
      size_t index = 0;
      for (const auto& elem : init_list_) {
        if (elem.type_ == InitListTensorType::Scalar) {
          at::NoGradGuard guard;
          tensor[index].fill_(elem.scalar());
        } else if (elem.type_ == InitListTensorType::InitList) {
          elem.fill_tensor(tensor[index]);
        } else {
          TORCH_INTERNAL_ASSERT(false, "Invalid InitListTensor");
        }
        index++;
      }
    }
    c10::Scalar scalar_;
    std::initializer_list<InitListTensor> init_list_;
    std::vector<int64_t> sizes_;
    c10::ScalarType scalar_type_;
    InitListTensorType type_;
  };

  inline std::ostream& operator<<(std::ostream& stream, const InitListTensor& list_init_tensor) {
    list_init_tensor.pretty_print_recursive(stream);
    return stream;
  }
} // namespace detail

#define TENSOR(T, S)                                                       \
  inline at::Tensor tensor(                                                \
      at::ArrayRef<T> values, const at::TensorOptions& options) {          \
    at::Tensor result = ([&]() {                                           \
      at::AutoNonVariableTypeMode non_var_type_mode(true);                 \
      return at::tensor(values, at::TensorOptions(options).is_variable(false)); \
    })();                                                                  \
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
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

/// NOTE: `torch::tensor({})` doesn't work at the moment because we would need to solve the
/// ambiguous overload problem (see https://github.com/pytorch/pytorch/pull/26210#discussion_r325336686).
/// If the user wants to create an empty tensor, they can use `torch::randn({0})` for now.
///
/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
inline at::Tensor tensor(detail::InitListTensor list_init_tensor, const at::TensorOptions& options) {
  return autograd::make_variable(list_init_tensor.to_tensor(options), options.requires_grad());
}

inline at::Tensor tensor(detail::InitListTensor list_init_tensor) {
  return torch::tensor(list_init_tensor, at::dtype(list_init_tensor.scalar_type()));
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
    return at::from_blob(data, sizes, strides, deleter, options.is_variable(false));
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
    return at::from_blob(data, sizes, deleter, options.is_variable(false));
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cudnn_init_dropout_state(dropout, train, dropout_seed, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::arange(end, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::arange(start, end, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::arange(start, end, step, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::bartlett_window(window_length, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::bartlett_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::blackman_window(window_length, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::blackman_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options = {}, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(size, names, at::TensorOptions(options).is_variable(false), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options = {}, c10::optional<MemoryFormat> memory_format = c10::nullopt) {
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
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(size, at::TensorOptions(options).is_variable(false), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _empty_affine_quantized(at::IntArrayRef size, const at::TensorOptions & options = {}, double scale = 1, int64_t zero_point = 0, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_empty_affine_quantized");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_affine_quantized(size, at::TensorOptions(options).is_variable(false), scale, zero_point, memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, const at::TensorOptions & options = {}, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
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
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, at::TensorOptions(options).is_variable(false), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty_like(self, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty_like(const at::Tensor & self, const at::TensorOptions & options, c10::optional<MemoryFormat> memory_format = MemoryFormat::Contiguous) {
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
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty_like(self, at::TensorOptions(options).is_variable(false), memory_format);
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty_strided(size, stride, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::eye(n, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::eye(n, m, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor full(at::IntArrayRef size, at::Scalar fill_value, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::full(size, fill_value, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor full(at::IntArrayRef size, at::Scalar fill_value, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::full(size, fill_value, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::full_like(self, fill_value, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::full_like(self, fill_value, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor from_file(std::string filename, c10::optional<bool> shared = c10::nullopt, c10::optional<int64_t> size = 0, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_file(filename, shared, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hann_window(window_length, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hann_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hamming_window(window_length, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hamming_window(window_length, periodic, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hamming_window(window_length, periodic, alpha, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hamming_window(window_length, periodic, alpha, beta, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::linspace(start, end, steps, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor logspace(at::Scalar start, at::Scalar end, int64_t steps = 100, double base = 10.0, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::logspace(start, end, steps, base, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor ones(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ones(size, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor ones(at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ones(size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ones_like(self, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ones_like(self, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::scalar_tensor(s, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand(size, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand(size, generator, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand(size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor rand(at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand(size, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand_like(self, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rand_like(self, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t high, at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint(high, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t high, at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint(high, size, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint(low, high, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint(low, high, size, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint_like(self, high, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint_like(self, low, high, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint_like(self, high, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randint_like(self, low, high, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn(size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntArrayRef size, at::Generator * generator, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn(size, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn(size, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor randn(at::IntArrayRef size, at::Generator * generator, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn(size, generator, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn_like(self, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randn_like(self, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randperm(n, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::randperm(n, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor range(at::Scalar start, at::Scalar end, at::Scalar step = 1, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::range(start, end, step, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::range(start, end, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor zeros(at::IntArrayRef size, c10::optional<DimnameList> names, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::zeros(size, names, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor zeros(at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::zeros(size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::zeros_like(self, self.options().is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/false);
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::zeros_like(self, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor sparse_coo_tensor(at::IntArrayRef size, const at::TensorOptions & options) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sparse_coo_tensor(size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sparse_coo_tensor(indices, values, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sparse_coo_tensor(indices, values, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, const at::TensorOptions & options = {}) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_unsafe(indices, values, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::TensorOptions & options) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options) {
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::tril_indices(row, col, offset, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::triu_indices(row, col, offset, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
inline at::Tensor normal(double mean, double std, at::IntArrayRef size, at::Generator * generator = nullptr, const at::TensorOptions & options = {}) {
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::normal(mean, std, size, generator, at::TensorOptions(options).is_variable(false));
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

} // namespace torch
