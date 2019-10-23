#pragma once

// ${generated_comment}

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

#define NESTED_INIT_LIST_MAX_DEPTH 10

enum class TensorDataContainerType { Scalar, InitList, Tensor };

// We use `TensorDataContainer` to support converting the following data container types
// into the equivalent Tensor:
//
// 1. Arbitrarily nested braced-init-list (e.g. `{{1, 2}, {3, 4}}`) up to `NESTED_INIT_LIST_MAX_DEPTH` dimensions.
// 2. `at::ArrayRef` of supported tensor data types.
// 3. `std::vector` of supported tensor data types.
//
// At any time, a `TensorDataContainer` object represents one of the following:
//
// 1. A scalar with value `scalar()` and type `scalar_type()`.
// 2. A Tensor represented in `std::initializer_list<TensorDataContainer>` form,
//    with value `init_list()`, Tensor scalar type `scalar_type()`, and Tensor sizes `sizes()`.
// 3. A Tensor represented in `at::Tensor` form, with value `tensor()`, scalar type `scalar_type()`,
//    and Tensor sizes `sizes()`.
//
// All the infrastructure here is mostly to support converting an arbitrarily nested braced-init-list
// to the equivalent Tensor successfully. Consider the following example:
//
// `torch::tensor({{1}, {2}})`
//
// Here is the code path that it goes through:
//
// `at::Tensor tensor(detail::TensorDataContainer<1> init_list_tensor)`
//
// which calls:
//
// `TensorDataContainer<1>({{1}, {2}})`
//
// which matches to the `TensorDataContainer<1>(std::initializer_list<TensorDataContainer<2>>)` constructor,
// and in an attempt to convert `{1}` and `{2}` to `TensorDataContainer<2>`, it calls the following:
//
// `TensorDataContainer<2>({1})`  (same call path happens for `{2}`, and we'll just focus on `{1}` here)
//
// At this point, theoretically there are two plausible ways for `{1}` to be matched to one of the
// constructors of `TensorDataContainer<2>`:
//
// 1. It can be a list-initialization of a scalar value, thus matching `TensorDataContainer<2>(int value)`.
// 2. It can be converted to `std::initializer_list<TensorDataContainer<3>>`, thus matching
//    `TensorDataContainer<2>(std::initializer_list<TensorDataContainer<3>>)`.
//
// How does the compiler decide which one to choose? According to `https://en.cppreference.com/w/cpp/language/list_initialization`,
// braced-init-list always prefers the constructor that takes `std::initializer_list`. Hence we happily
// move forward with constructor #2, and it calls the following:
//
// `TensorDataContainer<3>(1)`
//
// Now it matches `TensorDataContainer<3>(int value)`, which stores `1` as a scalar value. All is good.
//
// Note that `torch::tensor({{1}, {2}})` can also match another existing function overload:
// `torch::tensor(at::ArrayRef<int> values)`, because `{1}` and `{2}` can be treated as
// a list-initialization of an `int` value. However, this will produce a Tensor with sizes `{2}`,
// but we actually want a Tensor with sizes `{2, 1}`. In order to avoid matching this function overload,
// we move the ability to convert `at::ArrayRef<T>` (and similarly `std::vector<T>`) into `TensorDataContainer<1>`,
// and since for braced-init-list the `TensorDataContainer<1>(std::initializer_list<TensorDataContainer<2>>)`
// constructor is always preferred over all other constructors, it will take the `std::initializer_list` path
// and all is good again.
//
// P.S. You might ask: why do we need to templatize `TensorDataContainer`, and why can't we just provide a
// `TensorDataContainer(std::initializer_list<TensorDataContainer>>)` constructor to reach the innermost
// dimension of an arbitrarily nested braced-init-list? The answer is that for a braced-init-list
// like `{{1}, {2}}`, when the compiler is handling the inner braced-init-list `{1}` and `{2}`, it seems
// to always prefer the scalar constructor `TensorDataContainer(int)` instead of the `std::initializer_list`
// constructor, thus producing a tensor with the wrong sizes. If we templatize `TensorDataContainer` over
// the # of tensor dimensions, such problem doesn't happen.
template <size_t D>
struct TensorDataContainer {
  // NOTE: For tensors with zero-size dimensions (e.g. `torch::tensor({{}, {}})`),
  // the innermost empty braced-init-list `{}` matches the default constructor of
  // the innermost `TensorDataContainer`.
  TensorDataContainer() : sizes_({0}), scalar_type_(at::ScalarType::Undefined), type_(TensorDataContainerType::InitList) {}
#define TENSOR(T, S) \
  TensorDataContainer(T value) : sizes_(), scalar_(value), scalar_type_(at::k##S), type_(TensorDataContainerType::Scalar) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR
  TensorDataContainer(std::initializer_list<TensorDataContainer<D+1>> init_list) :
      sizes_(),
      scalar_(),
      scalar_type_(init_list.begin()->scalar_type()),
      init_list_(init_list),
      type_(TensorDataContainerType::InitList) {
    const TensorDataContainer<D+1>& first_elem = *(init_list.begin());
    for (const auto& elem : init_list) {
      TORCH_CHECK(elem.sizes() == first_elem.sizes(),
        "Expected all sub-lists to have sizes: ",
        first_elem.sizes(),
        " (e.g. ", first_elem, "), ",
        "but got sub-list ",
        elem,
        " with sizes: ",
        elem.sizes());
      TORCH_CHECK(elem.scalar_type() == first_elem.scalar_type(),
        "Expected all elements of the tensor to have the same scalar type: ",
        first_elem.scalar_type(),
        ", but got element of scalar type: ",
        elem.scalar_type());
    }
    sizes_.clear();
    sizes_.reserve(first_elem.sizes().size() + 1);
    sizes_.push_back(init_list.size());
    sizes_.insert(sizes_.end(), first_elem.sizes().begin(), first_elem.sizes().end());
  }

#define TENSOR(T, S) \
  TensorDataContainer(at::ArrayRef<T> values) { \
    at::AutoNonVariableTypeMode non_var_type_mode(true);  \
    type_ = TensorDataContainerType::Tensor; \
    sizes_ = {(int64_t)values.size()}; \
    scalar_type_ = at::k##S; \
    tensor_ = at::tensor(values, at::TensorOptions().device(at::kCPU).is_variable(false));       \
  } \
  TensorDataContainer(std::vector<T> values) : TensorDataContainer(at::ArrayRef<T>(values)) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

  const c10::Scalar& scalar() const {
    return scalar_;
  }

  const std::initializer_list<TensorDataContainer<D+1>>& init_list() const {
    return init_list_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  const c10::ScalarType& scalar_type() const {
    return scalar_type_;
  }

  const TensorDataContainerType& type() const {
    return type_;
  }

  at::Tensor tensor(const at::TensorOptions& options = {}) const {
    if (type_ == TensorDataContainerType::Scalar) {
      at::AutoNonVariableTypeMode non_var_type_mode(true);
      return at::scalar_tensor(scalar_, options.is_variable(false));
    } else if (type_ == TensorDataContainerType::InitList) {
      // NOTE: Here we explicitly choose to initialize the tensor on CPU first,
      // fill each element of the tensor, and then move the tensor to the desired
      // device. For CUDA device, this approach only involves 1 CUDA kernel launch,
      // and is much faster than initializing the tensor on CUDA first and then
      // filling each element of it (which involves `N` CUDA kernel launches where
      // `N` is the number of the elements in the tensor).
      at::Tensor tensor = ([&]() {
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        return at::empty(sizes_, options.device(at::kCPU).is_variable(false));
      })();
      fill_tensor(tensor);
      return tensor.to(options.device());
    } else if (type_ == TensorDataContainerType::Tensor) {
      return tensor_.to(options);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

  void pretty_print_recursive(std::ostream& stream) const {
    if (type_ == TensorDataContainerType::Scalar) {
      AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "TensorDataContainer_pretty_print_scalar", [&] {
        stream << scalar_.template to<scalar_t>();
      });
    } else if (type_ == TensorDataContainerType::InitList) {
      stream << "{";
      for (const TensorDataContainer<D+1>* it = init_list_.begin(); it != init_list_.end(); it++) {
        stream << *it;
        if (std::next(it) != init_list_.end()) stream << ", ";
      }
      stream << "}";
    } else if (type_ == TensorDataContainerType::Tensor) {
      stream << "{";
      for (int64_t i = 0; i < tensor_.sizes()[0]; i++) {
        AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "TensorDataContainer_pretty_print_tensor_item", [&] {
          stream << tensor_[i].template item<scalar_t>();
        });
      }
      stream << "}";
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }
 private:
  void fill_tensor(at::Tensor tensor) const {
    if (type_ == TensorDataContainerType::Scalar) {
      at::NoGradGuard guard;
      tensor.fill_(scalar_);
    } else if (type_ == TensorDataContainerType::InitList) {
      size_t index = 0;
      for (const auto& elem : init_list_) {
        elem.fill_tensor(tensor[index]);
        index++;
      }
    } else if (type_ == TensorDataContainerType::Tensor) {
      TORCH_INTERNAL_ASSERT(
        false,
        "TensorDataContainer is already a Tensor type, `fill_tensor` should not be called");
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

  std::vector<int64_t> sizes_;
  c10::Scalar scalar_;
  c10::ScalarType scalar_type_;
  std::initializer_list<TensorDataContainer<D+1>> init_list_;
  TensorDataContainerType type_;
  at::Tensor tensor_;

  friend struct TensorDataContainer<(D > 1) ? D-1 : D>;
};

template <size_t D>
inline std::ostream& operator<<(std::ostream& stream, const TensorDataContainer<D>& init_list_tensor) {
  init_list_tensor.pretty_print_recursive(stream);
  return stream;
}

// NOTE: The innermost braced-init-list must satisfy one of the following conditions:
//
// 1. It contains no value (i.e. `{}`).
// 2. It contains only scalar values (e.g. `{1.1, 2.2}`).
//
// Case #1 is handled by the upper dimension `TensorDataContainer<D-1>`'s default constructor,
// while case #2 is handled by this dimension `TensorDataContainer<D>`s scalar constructor.
//
// Currently, we only support braced-init-lists that are up to `NESTED_INIT_LIST_MAX_DEPTH` in depth.
// Following the above reasoning, `TensorDataContainer<NESTED_INIT_LIST_MAX_DEPTH+1>` should only
// accept scalars, and we throw "dimension exceeded" error in other constructors.
template<>
struct TensorDataContainer<NESTED_INIT_LIST_MAX_DEPTH+1> {
  TensorDataContainer() {
    TORCH_CHECK(
      false,
      "Tensor with more than ",
      NESTED_INIT_LIST_MAX_DEPTH,
      " dimensions is not supported");
  }
#define TENSOR(T, S) \
  TensorDataContainer(T value) : sizes_(), scalar_(value), scalar_type_(at::k##S), type_(TensorDataContainerType::Scalar) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR
  // NOTE: This constructor accepts a `std::initializer_list` of its own
  // (different from the template version), so that the compiler won't go off
  // and continue to instantiate `TensorDataContainer` of the next dimension,
  // which would cause infinite recursion.
  TensorDataContainer(std::initializer_list<TensorDataContainer<NESTED_INIT_LIST_MAX_DEPTH+1>> init_list) {
    TORCH_CHECK(
      false,
      "Tensor with more than ",
      NESTED_INIT_LIST_MAX_DEPTH,
      " dimensions is not supported");
  }

  const c10::Scalar& scalar() const {
    return scalar_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  const c10::ScalarType& scalar_type() const {
    return scalar_type_;
  }

  const TensorDataContainerType& type() const {
    return type_;
  }

  void fill_tensor(at::Tensor tensor) const {
    if (type_ == TensorDataContainerType::Scalar) {
      at::NoGradGuard guard;
      tensor.fill_(scalar_);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Only scalar type is supported for this TensorDataContainer");
    }
  }

  void pretty_print_recursive(std::ostream& stream) const {
    if (type_ == TensorDataContainerType::Scalar) {
      AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "TensorDataContainer_pretty_print_scalar", [&] {
        stream << scalar_.template to<scalar_t>();
      });
    } else {
      TORCH_INTERNAL_ASSERT(false, "Only scalar type is supported for this TensorDataContainer");
    }
  }

 private:
  std::vector<int64_t> sizes_;
  c10::Scalar scalar_;
  c10::ScalarType scalar_type_;
  TensorDataContainerType type_;
};

} // namespace detail

/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
inline at::Tensor tensor(detail::TensorDataContainer<1> init_list_tensor, const at::TensorOptions& options) {
  return autograd::make_variable(init_list_tensor.tensor(options), options.requires_grad());
}

inline at::Tensor tensor(detail::TensorDataContainer<1> init_list_tensor) {
  return torch::tensor(init_list_tensor, at::dtype(init_list_tensor.scalar_type()));
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

${function_definitions}

} // namespace torch
