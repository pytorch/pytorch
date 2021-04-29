#pragma once

#include <ATen/ATen.h>

#include <initializer_list>

namespace torch {

namespace detail {

enum class TensorDataContainerType { Scalar, InitList, Tensor };

struct TensorDataContainer;

inline std::ostream& operator<<(std::ostream& stream, const TensorDataContainer& tensor_data_container);

// FIXME: There is no `operator<<` overload for `at::kBFloat16` type,
// and we need to convert it to `float` type using `operator float()` function
// defined in `c10/util/BFloat16.h`.
// Tracking issue: https://github.com/pytorch/pytorch/issues/28845
inline std::ostream& operator<<(std::ostream& stream, c10::BFloat16 value) {
  stream << static_cast<float>(value);
  return stream;
}

inline c10::ScalarType compute_desired_dtype(c10::ScalarType scalar_type) {
  if (scalar_type == at::kInt || scalar_type == at::kLong) {
    // C++ `torch::tensor` with an integer type or an `at::ArrayRef` / `std::vector` /
    // (nested) braced-init-list of integer types always produces a tensor of dtype `at::kLong`
    // (aka. int64_t), matching Python `torch.tensor` behavior.
    return at::kLong;
  } else if (scalar_type == at::kFloat || scalar_type == at::kDouble) {
    // C++ `torch::tensor` with a floating-point type or an `at::ArrayRef` / `std::vector` /
    // (nested) braced-init-list of floating-point types always produces a tensor of dtype
    // `torch::get_default_dtype()`, matching Python `torch.tensor` behavior.
    return at::typeMetaToScalarType(at::get_default_dtype());
  } else {
    return scalar_type;
  }
}

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
// this will call into the `torch::tensor` function:
//
// `at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const at::TensorOptions& options = {})`
//
// the compiler will first try to convert `{{1}, {2}}` to `TensorDataContainer` type:
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
struct TensorDataContainer {
  // NOTE: For tensors with zero-size dimensions (e.g. `torch::tensor({{}, {}})`),
  // the innermost empty braced-init-list `{}` matches the default constructor of
  // the innermost `TensorDataContainer`.
  TensorDataContainer() :
      sizes_({0}),
      // NOTE: In Python, the dtype of tensors with zero-size dimensions (e.g. `torch.tensor([[], []])`)
      // depends on the value of `torch.get_default_dtype()`, and we should do the same for the C++ equivalent.
      scalar_type_(at::typeMetaToScalarType(at::get_default_dtype())),
      type_(TensorDataContainerType::InitList) {}
#define TENSOR(T, S) \
  TensorDataContainer(T value) : \
      sizes_(), \
      scalar_type_(at::k##S), \
      type_(TensorDataContainerType::Scalar), \
      scalar_(value) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR
  TensorDataContainer(std::initializer_list<TensorDataContainer> init_list) :
      sizes_(),
      scalar_type_(init_list.begin()->scalar_type()),
      type_(TensorDataContainerType::InitList),
      init_list_(init_list) {
    const TensorDataContainer& first_elem = *(init_list.begin());
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
    sizes_.reserve(first_elem.sizes().size() + 1);
    sizes_.push_back(init_list.size());
    sizes_.insert(sizes_.end(), first_elem.sizes().begin(), first_elem.sizes().end());
  }

#define TENSOR(T, S) \
  TensorDataContainer(at::ArrayRef<T> values) : \
      sizes_({(int64_t)values.size()}), \
      scalar_type_(at::k##S), \
      type_(TensorDataContainerType::Tensor) { \
    at::AutoDispatchBelowAutograd mode; \
    if (scalar_type_ == at::kBool) { \
      tensor_ = at::tensor(values, at::TensorOptions().device(at::kCPU)); \
    } else { \
      tensor_ = at::tensor(values, at::dtype(scalar_type_).device(at::kCPU)); \
    } \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

  // NOTE: We need to handle `std::vector` explicitly instead of relying on an implicit conversion
  // to `at::ArrayRef`, otherwise the following error can be thrown when calling
  // `torch::tensor(std::vector<int>({1, 2}))`:
  // ```
  // error: no matching function for call to 'tensor(const std::vector<int>&)'
  // no known conversion for argument 1 from 'const std::vector<int>' to
  // 'torch::detail::TensorDataContainer'
  // ```
  //
  // NOTE: `torch::tensor(std::vector<bool>)` is not supported for now, because
  // ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.
#define TENSOR(T, S) \
  TensorDataContainer(const std::vector<T>& values) : TensorDataContainer(at::ArrayRef<T>(values)) {}
AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

  bool is_scalar() const {
    return type_ == TensorDataContainerType::Scalar;
  }

  const c10::Scalar& scalar() const {
    TORCH_CHECK(
      is_scalar(),
      "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    return scalar_;
  }

  bool is_init_list() const {
    return type_ == TensorDataContainerType::InitList;
  }

  const std::initializer_list<TensorDataContainer>& init_list() const {
    TORCH_CHECK(
      is_init_list(),
      "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
    return init_list_;
  }

  bool is_tensor() const {
    return type_ == TensorDataContainerType::Tensor;
  }

  const at::Tensor& tensor() const {
    TORCH_CHECK(
      is_tensor(),
      "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
    return tensor_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  const c10::ScalarType& scalar_type() const {
    return scalar_type_;
  }

  at::Tensor convert_to_tensor(at::TensorOptions options) const {
    if (!options.has_dtype()) {
      options = options.dtype(compute_desired_dtype(scalar_type_));
    }

    if (is_scalar()) {
      at::AutoDispatchBelowAutograd mode;
      return at::scalar_tensor(scalar_, options);
    } else if (is_init_list()) {
      // NOTE: Here we explicitly choose to initialize the tensor on CPU first,
      // fill each element of the tensor, and then move the tensor to the desired
      // device. For CUDA device, this approach only involves 1 CUDA kernel launch,
      // and is much faster than initializing the tensor on CUDA first and then
      // filling each element of it (which involves `N` CUDA kernel launches where
      // `N` is the number of the elements in the tensor).
      at::Tensor tensor = ([&]() {
        at::AutoDispatchBelowAutograd mode;
        return at::empty(sizes_, options.device(at::kCPU));
      })();
      fill_tensor(tensor);
      return tensor.to(options.device());
    } else if (is_tensor()) {
      auto output = tensor_.to(options);
      TORCH_CHECK(!tensor_.is_complex() || output.is_complex(), "can not do torch::tensor(complex, dtype=non-complex) because complex can not be casted to real number without loss of information");
      return output;
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

  void pretty_print_recursive(std::ostream& stream) const {
    if (is_scalar()) {
      AT_DISPATCH_ALL_TYPES_AND3(
          at::kBool,
          at::kHalf,
          at::kBFloat16,
          scalar_type_,
          "TensorDataContainer_pretty_print_scalar", [&] {
        stream << scalar_.to<scalar_t>();
      });
    } else if (is_init_list()) {
      stream << "{";
      for (const TensorDataContainer* it = init_list_.begin(); it != init_list_.end(); it++) {
        stream << *it;
        if (std::next(it) != init_list_.end()) stream << ", ";
      }
      stream << "}";
    } else if (is_tensor()) {
      stream << "{";
      for (int64_t i = 0; i < tensor_.sizes()[0]; i++) {
        AT_DISPATCH_ALL_TYPES_AND3(
            at::kBool,
            at::kHalf,
            at::kBFloat16,
            scalar_type_,
            "TensorDataContainer_pretty_print_tensor_item", [&] {
          stream << tensor_[i].item<scalar_t>();
        });
        if (i != tensor_.sizes()[0] - 1) stream << ", ";
      }
      stream << "}";
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }
 private:
  void fill_tensor(at::Tensor& tensor) const {
    if (is_scalar()) {
      TORCH_INTERNAL_ASSERT(
        tensor.dim() == 0,
        "Expected a 0-dim Tensor, but got Tensor with dimensions: ", tensor.dim());
      at::NoGradGuard guard;
      tensor.fill_(scalar_);
    } else if (is_init_list()) {
      TORCH_INTERNAL_ASSERT(
        tensor.sizes()[0] == (int64_t)init_list_.size(),
        "Expected a Tensor with size ",
        init_list_.size(),
        " in its first dimension, but got Tensor with size ",
        tensor.sizes()[0],
        " in its first dimension");
      size_t index = 0;
      for (const auto& elem : init_list_) {
        at::Tensor slice = tensor[index];
        elem.fill_tensor(slice);
        index++;
      }
    } else if (is_tensor()) {
      TORCH_INTERNAL_ASSERT(
        false,
        "TensorDataContainer is already a Tensor type, `fill_tensor` should not be called");
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

  std::vector<int64_t> sizes_;
  c10::ScalarType scalar_type_;
  TensorDataContainerType type_;
  c10::Scalar scalar_;
  std::initializer_list<TensorDataContainer> init_list_;
  at::Tensor tensor_;
};

inline std::ostream& operator<<(std::ostream& stream, const TensorDataContainer& tensor_data_container) {
  tensor_data_container.pretty_print_recursive(stream);
  return stream;
}

} // namespace detail

} // namespace torch
