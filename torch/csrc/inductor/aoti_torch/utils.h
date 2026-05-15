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

// Owning wrapper used as the return type of pointer_to_optional_list.
//
// The previous implementation returned `std::optional<c10::ArrayRef<T>>`
// constructed from a `std::vector<T>` temporary. When the caller's T differs
// from the on-wire type U (e.g. T=c10::SymInt, U=int64_t), pointer_to_list
// materializes a new std::vector<T> by value; that vector is destroyed at the
// end of pointer_to_optional_list, leaving the ArrayRef inside the returned
// optional dangling. The receiving op (e.g. convolution_backward_symint) then
// reads freed memory -- ASAN catches it as heap-use-after-free at
// c10::SymInt::is_heap_allocated.
//
// This wrapper holds the vector itself. The implicit conversion below produces
// a c10::OptionalArrayRef<T> view into that vector. Because the wrapper is the
// temporary in the call site's full expression, its storage stays alive for
// the duration of the receiving function call.
template <class T>
struct OwningOptionalArrayRef {
  std::optional<std::vector<T>> storage;

  OwningOptionalArrayRef() = default;
  explicit OwningOptionalArrayRef(std::vector<T> v) : storage(std::move(v)) {}

  /* implicit */ operator c10::OptionalArrayRef<T>() const {
    return storage ? c10::OptionalArrayRef<T>(c10::ArrayRef<T>(*storage))
                   : c10::OptionalArrayRef<T>(std::nullopt);
  }

  // Some generated shim call sites pass into a parameter of type
  // std::optional<c10::ArrayRef<T>> (rather than c10::OptionalArrayRef<T>),
  // e.g. at::cpu::_histogramdd_from_bin_cts(...,
  // std::optional<ArrayRef<double>>).
  /* implicit */ operator std::optional<c10::ArrayRef<T>>() const {
    return storage ? std::make_optional(c10::ArrayRef<T>(*storage))
                   : std::nullopt;
  }
};

// Utility function to convert a pointer to an optional list of values
template <class T, class U>
inline OwningOptionalArrayRef<T> pointer_to_optional_list(
    U** ptr,
    int64_t len) {
  if (!ptr) {
    return OwningOptionalArrayRef<T>{};
  }
  // pointer_to_list returns a std::vector<T> by value when T != U (typical:
  // T=c10::SymInt, U=int64_t), or a c10::ArrayRef<T> when T == U. Materialize
  // into a vector either way so the wrapper owns the storage uniformly.
  // TODO: skip the copy when T == U (the wire buffer already outlives the
  // call); would require the wrapper to optionally hold an ArrayRef view.
  auto list = pointer_to_list<T>(*ptr, len);
  return OwningOptionalArrayRef<T>(std::vector<T>(list.begin(), list.end()));
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
