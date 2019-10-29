#pragma once

#include <ATen/ATen.h>

#include <initializer_list>

namespace torch {

namespace detail {

enum class TensorDataContainerType { Scalar, InitList, Tensor };

// We use `TensorDataContainer` to support converting the following data container types
// into the equivalent Tensor:
//
// 1. Arbitrarily nested braced-init-list (e.g. `{{1, 2}, {3, 4}}`).
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
// `at::Tensor tensor(detail::TensorDataContainer tensor_data_container)`
//
// which calls:
//
// `TensorDataContainer({{1}, {2}})`
//
// which matches to the `TensorDataContainer(std::initializer_list<TensorDataContainer>)` constructor,
// and in an attempt to convert `{1}` and `{2}` to `TensorDataContainer`, it calls the following:
//
// `TensorDataContainer({1})`  (same call path happens for `{2}`, and we'll just focus on `{1}` here)
//
// At this point, theoretically there are two plausible ways for `{1}` to be matched to one of the
// constructors of `TensorDataContainer`:
//
// 1. It can be a list-initialization of a scalar value, thus matching `TensorDataContainer(int value)`.
// 2. It can be converted to `std::initializer_list<TensorDataContainer>`, thus matching
//    `TensorDataContainer(std::initializer_list<TensorDataContainer>)`.
//
// How does the compiler decide which one to choose? According to
// `https://en.cppreference.com/w/cpp/language/list_initialization`, braced-init-list always prefers
// the constructor that takes `std::initializer_list`. Hence we happily move forward with constructor #2,
// and it calls the following:
//
// `TensorDataContainer(1)`
//
// Now it matches `TensorDataContainer(int value)`, which stores `1` as a scalar value. All is good.
//
// Note that `torch::tensor({{1}, {2}})` can also match another previously existing function overload:
// `torch::tensor(at::ArrayRef<int> values)`, because `{1}` and `{2}` can be treated as
// a list-initialization of an `int` value. However, this will produce a Tensor with sizes `{2}`,
// but we actually want a Tensor with sizes `{2, 1}`. In order to avoid matching this function overload,
// we removed the function overload and moved the ability to convert `at::ArrayRef<T>`
// (and similarly `std::vector<T>`) into `TensorDataContainer`, and since for braced-init-list the
// `TensorDataContainer(std::initializer_list<TensorDataContainer>)` constructor is always preferred
// over all other constructors, it will take the `std::initializer_list` path and all is good again.
struct TensorDataContainer {
  // NOTE: For tensors with zero-size dimensions (e.g. `torch::tensor({{}, {}})`),
  // the innermost empty braced-init-list `{}` matches the default constructor of
  // the innermost `TensorDataContainer`.
  TensorDataContainer();

#define TENSOR(T, S) \
  TensorDataContainer(T value);
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

  TensorDataContainer(std::initializer_list<TensorDataContainer> init_list);

#define TENSOR(T, S) \
  TensorDataContainer(at::ArrayRef<T> values);
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

  // NOTE: We need to handle `std::vector` explicitly instead of relying on an
  // implicit conversion to `at::ArrayRef`, otherwise the following error can be
  // thrown when calling `torch::tensor(std::vector<double>({1.1, 2.2}))`:
  // ```
  // error: no matching function for call to ‘tensor(const std::vector<int>&)’
  // no known conversion for argument 1 from ‘const std::vector<int>’ to
  // ‘torch::detail::TensorDataContainer’
  // ```
  //
  // NOTE: `torch::tensor(std::vector<bool>)` is not supported for now, because
  // `at::ArrayRef<bool>` cannot be constructed from a `std::vector<bool>` bitfield.
#define TENSOR(T, S) \
  TensorDataContainer(const std::vector<T>& values);
AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TENSOR)
#undef TENSOR

  bool is_scalar() const;

  const c10::Scalar& scalar() const;

  bool is_init_list() const;

  const std::initializer_list<TensorDataContainer>& init_list() const;

  bool is_tensor() const;

  const at::Tensor& tensor() const;

  const std::vector<int64_t>& sizes() const;

  const c10::ScalarType& scalar_type() const;

  at::Tensor convert_to_tensor(const at::TensorOptions& options) const;

  void pretty_print_recursive(std::ostream& stream) const;

 private:
  void fill_tensor(at::Tensor tensor) const;

  std::vector<int64_t> sizes_;
  c10::ScalarType scalar_type_;
  TensorDataContainerType type_;
  c10::Scalar scalar_;
  std::initializer_list<TensorDataContainer> init_list_;
  at::Tensor tensor_;
};

std::ostream& operator<<(std::ostream& stream, const TensorDataContainer& tensor_data_container);

} // namespace detail

} // namespace torch
