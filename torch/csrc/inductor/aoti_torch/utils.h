#pragma once

#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Logging.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/shim_exception_state.h>
#include <optional>
#include <type_traits>

namespace torch::aot_inductor {
TORCH_API const char* get_last_error();
TORCH_API void set_last_error(const char* msg);
} // namespace torch::aot_inductor

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)                     \
  try {                                                                     \
    __VA_ARGS__                                                             \
  } catch (const c10::Error& e) {                                           \
    torch::csrc::shim::details::set_torch_exception_what(e.what());         \
    torch::csrc::shim::details::set_torch_exception_what_without_backtrace( \
        e.what_without_backtrace());                                        \
    return AOTI_TORCH_FAILURE;                                              \
  } catch (const std::exception& e) {                                       \
    torch::csrc::shim::details::set_torch_exception_what(e.what());         \
    torch::csrc::shim::details::set_torch_exception_what_without_backtrace( \
        torch::csrc::shim::details::get_torch_exception_what());            \
    return AOTI_TORCH_FAILURE;                                              \
  } catch (...) {                                                           \
    torch::csrc::shim::details::set_torch_exception_what("UNKNOWN");        \
    torch::csrc::shim::details::set_torch_exception_what_without_backtrace( \
        torch::csrc::shim::details::get_torch_exception_what());            \
    return AOTI_TORCH_FAILURE;                                              \
  }                                                                         \
  return AOTI_TORCH_SUCCESS;

namespace torch::aot_inductor {

inline at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}

inline AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor) {
  return reinterpret_cast<AtenTensorHandle>(tensor);
}

inline at::Tensor resolve_tensor_dispatch_flags(AtenTensorHandle handle) {
  at::Tensor* tensor{tensor_handle_to_tensor_pointer(handle)};
  if (tensor->is_conj() || tensor->is_neg()) {
    // If the conjugation or negation dispatch flags are set, runtime dispatch
    // handles them by cloning the tensor before passing them to the native ATen
    // function.  Since the C-shim calls the native function directly, we have
    // to handle the flags ourselves, or results will be silently incorrect.
    return tensor->clone();
  }
  return *tensor;
}

inline std::optional<at::Tensor> resolve_tensor_dispatch_flags(
    const AtenTensorHandle* handle) {
  return handle ? std::make_optional(resolve_tensor_dispatch_flags(*handle))
                : std::nullopt;
}

inline std::vector<at::Tensor> resolve_tensor_list_dispatch_flags(
    const AtenTensorHandle* handle,
    int64_t len) {
  std::vector<at::Tensor> ret{};
  ret.reserve(len);
  for (int64_t i{0}; i < len; ++i) {
    ret.emplace_back(resolve_tensor_dispatch_flags(handle[i]));
  }
  return ret;
}

inline std::vector<std::optional<at::Tensor>> resolve_tensor_list_dispatch_flags(
    const AtenTensorHandle** handle,
    int64_t len) {
  std::vector<std::optional<at::Tensor>> ret{};
  ret.reserve(len);
  for (int64_t i{0}; i < len; ++i) {
    ret.emplace_back(resolve_tensor_dispatch_flags(handle[i]));
  }
  return ret;
}

inline at::Generator* generator_handle_to_generator_pointer(
    AtenGeneratorHandle handle) {
  return reinterpret_cast<at::Generator*>(handle);
}

inline AtenGeneratorHandle generator_pointer_to_generator_handle(
    at::Generator* generator) {
  return reinterpret_cast<AtenGeneratorHandle>(generator);
}

inline AtenTensorHandle new_tensor_handle(at::Tensor&& tensor) {
  at::Tensor* new_tensor = new at::Tensor(std::move(tensor));
  return tensor_pointer_to_tensor_handle(new_tensor);
}

inline void assert_inf_and_nan(
    const std::string& tensor_name,
    at::Tensor& check_tensor) {
  auto isnan_tensor = check_tensor.isnan();
  if (isnan_tensor.any().item<bool>()) {
    throw std::runtime_error("At least one NaN in " + tensor_name);
  }
  auto isinf_tensor = check_tensor.isinf();
  if (isinf_tensor.any().item<bool>()) {
    throw std::runtime_error("At least one INF in " + tensor_name);
  }
}

// utility functions to convert a pointer to an optional value
template <class T>
inline std::optional<T> pointer_to_optional(T* ptr) {
  return ptr ? std::make_optional(*ptr) : std::nullopt;
}

template <class T, class U, typename = std::enable_if_t<!std::is_same_v<T, U>>>
inline std::optional<T> pointer_to_optional(U* ptr) {
  return ptr ? std::make_optional<T>(T(*ptr)) : std::nullopt;
}

template <>
inline std::optional<at::Tensor> pointer_to_optional(AtenTensorHandle* ptr) {
  return ptr ? std::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : std::nullopt;
}

template <>
inline std::optional<at::Tensor> pointer_to_optional(
    const AtenTensorHandle* ptr) {
  return ptr ? std::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : std::nullopt;
}

template <>
inline std::optional<at::Generator> pointer_to_optional(
    AtenGeneratorHandle* ptr) {
  return ptr ? std::make_optional(*generator_handle_to_generator_pointer(*ptr))
             : std::nullopt;
}

inline std::optional<c10::Device> pointer_to_optional_device(
    int32_t* device_type,
    int32_t device_index) {
  return device_type ? std::make_optional(c10::Device(
                           static_cast<c10::DeviceType>(*device_type),
                           static_cast<c10::DeviceIndex>(device_index)))
                     : std::nullopt;
}

// utility functions to convert a pointer to a list
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <class T>
inline c10::ArrayRef<T> pointer_to_list(T* ptr, int64_t len) {
  return c10::ArrayRef<T>(ptr, len);
}

template <
    class T,
    class U,
    typename = std::enable_if_t<!std::is_same_v<T, U>>,
    typename = std::enable_if_t<!is_optional<T>::value>>
inline std::vector<T> pointer_to_list(U* ptr, int64_t len) {
  // std::vector<T> will be implicitly converted to c10::ArrayRef<T> at the call
  // site
  std::vector<T> result;
  result.reserve(len);
  for (const auto i : c10::irange(len)) {
    result.emplace_back(T(ptr[i]));
  }
  return result;
}

template <class T, class U, typename = std::enable_if_t<is_optional<T>::value>>
inline std::vector<T> pointer_to_list(U** ptr, int64_t len) {
  // Here U** denotes a list of optional arguments
  // std::vector<T> will be implicitly converted to c10::ArrayRef<T> at the call
  // site
  std::vector<T> result;
  result.reserve(len);
  for (const auto i : c10::irange(len)) {
    result.emplace_back(pointer_to_optional(ptr[i]));
  }
  return result;
}

template <>
inline std::vector<at::Tensor> pointer_to_list(
    const AtenTensorHandle* ptr,
    int64_t len) {
  std::vector<at::Tensor> result;
  result.reserve(len);
  for (const auto i : c10::irange(len)) {
    result.emplace_back(*tensor_handle_to_tensor_pointer(ptr[i]));
  }
  return result;
}

template <>
inline std::vector<std::optional<at::Tensor>> pointer_to_list(
    const AtenTensorHandle** ptr,
    int64_t len) {
  std::vector<std::optional<at::Tensor>> result;
  result.reserve(len);
  for (const auto i : c10::irange(len)) {
    result.emplace_back(pointer_to_optional<at::Tensor>(ptr[i]));
  }
  return result;
}

template <int N>
inline std::array<bool, N> pointer_to_list(const int32_t* ptr) {
  std::array<bool, N> result;
  std::copy(ptr, ptr + N, result.begin());
  return result;
}

// Wrappers returned by pointer_to_optional_list.
//
// Both expose the same two implicit conversions so generated shim call sites
// can pass the result into either c10::OptionalArrayRef<T> or
// std::optional<c10::ArrayRef<T>> parameters uniformly.
//
// Two wrappers (not one) so the dispatch is compile-time: each call site of
// pointer_to_optional_list resolves to exactly one type via `if constexpr`,
// with no runtime tag.

// Used when the wire element type matches T (modulo cv): the wire buffer can
// be viewed directly as an ArrayRef. No allocation, no copy. The wire buffer
// outlives the surrounding op call expression, so the view stays valid.
template <class T>
struct BorrowedOptionalArrayRef {
  c10::OptionalArrayRef<T> storage;

  /* implicit */ operator c10::OptionalArrayRef<T>() const noexcept {
    return storage;
  }
  /* implicit */ operator std::optional<c10::ArrayRef<T>>() const {
    return storage.has_value() ? std::make_optional(*storage) : std::nullopt;
  }
};

// Used when the wire element type differs from T and pointer_to_list has to
// materialize a converted std::vector<T> (e.g. int64_t wire -> c10::SymInt).
// Owning the vector keeps it alive across the receiving op call so an
// ArrayRef projected from it stays valid.
//
// Returning std::optional<c10::ArrayRef<T>> directly here was the original
// bug: the ArrayRef viewed into a temporary vector destroyed at the end of
// pointer_to_optional_list, surfaced by ASAN as a heap-use-after-free in
// c10::SymInt::is_heap_allocated when convolution_backward_symint read freed
// SymInt memory.
template <class T>
struct OwnedOptionalArrayRef {
  std::optional<std::vector<T>> storage;

  /* implicit */ operator c10::OptionalArrayRef<T>() const {
    return storage ? c10::OptionalArrayRef<T>(c10::ArrayRef<T>(*storage))
                   : c10::OptionalArrayRef<T>(std::nullopt);
  }
  /* implicit */ operator std::optional<c10::ArrayRef<T>>() const {
    return storage ? std::make_optional(c10::ArrayRef<T>(*storage))
                   : std::nullopt;
  }
};

// Utility function to convert a pointer to an optional list of values.
//
// Compile-time dispatch on whether T matches the wire element type U (modulo
// cv): same-type returns a borrowed view of the wire buffer (zero-copy);
// different-type returns an owning wrapper around the converted vector.
template <class T, class U>
inline auto pointer_to_optional_list(U** ptr, int64_t len) {
  if constexpr (std::is_same_v<T, std::remove_cv_t<U>>) {
    return BorrowedOptionalArrayRef<T>{
        ptr ? c10::OptionalArrayRef<T>(c10::ArrayRef<T>(*ptr, len))
            : c10::OptionalArrayRef<T>(std::nullopt)};
  } else {
    return OwnedOptionalArrayRef<T>{
        ptr ? std::optional<std::vector<T>>(pointer_to_list<T>(*ptr, len))
            : std::nullopt};
  }
}

template <typename T>
static c10::List<T> convert_to_c10_List(const T* scalars, const int64_t len) {
  c10::List<T> scalars_list;
  scalars_list.reserve(len);
  for (const auto i : c10::irange(len)) {
    scalars_list.emplace_back(scalars[i]);
  }
  return scalars_list;
}

} // namespace torch::aot_inductor
