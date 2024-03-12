#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/List.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)    \
  try {                                                    \
    __VA_ARGS__                                            \
  } catch (const std::exception& e) {                      \
    LOG(ERROR) << "Exception in aoti_torch: " << e.what(); \
    return AOTI_TORCH_FAILURE;                             \
  } catch (...) {                                          \
    LOG(ERROR) << "Exception in aoti_torch: UNKNOWN";      \
    return AOTI_TORCH_FAILURE;                             \
  }                                                        \
  return AOTI_TORCH_SUCCESS;

namespace torch::aot_inductor {

// utility functions to convert a pointer to an optional value
template <class T>
inline c10::optional<T> pointer_to_optional(T* ptr) {
  return ptr ? c10::make_optional(*ptr) : c10::nullopt;
}

template <class T, class U, typename = std::enable_if_t<!std::is_same_v<T, U>>>
inline c10::optional<T> pointer_to_optional(U* ptr) {
  return ptr ? c10::make_optional<T>(T(*ptr)) : c10::nullopt;
}

template <>
inline c10::optional<at::Tensor> pointer_to_optional(AtenTensorHandle* ptr) {
  return ptr ? c10::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : c10::nullopt;
}

template <>
inline c10::optional<at::Tensor> pointer_to_optional(
    const AtenTensorHandle* ptr) {
  return ptr ? c10::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : c10::nullopt;
}

inline c10::optional<c10::Device> pointer_to_optional_device(
    int32_t* device_type,
    int32_t device_index) {
  return device_type ? c10::make_optional(c10::Device(
                           static_cast<c10::DeviceType>(*device_type),
                           static_cast<c10::DeviceIndex>(device_index)))
                     : c10::nullopt;
}

// utility functions to convert a pointer to a list
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<c10::optional<T>> : std::true_type {};

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
  for (int64_t i = 0; i < len; i++) {
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
  for (int64_t i = 0; i < len; i++) {
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
  for (int64_t i = 0; i < len; i++) {
    result.emplace_back(*tensor_handle_to_tensor_pointer(*ptr));
  }
  return result;
}

template <>
inline std::vector<c10::optional<at::Tensor>> pointer_to_list(
    const AtenTensorHandle** ptr,
    int64_t len) {
  std::vector<c10::optional<at::Tensor>> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; i++) {
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

// utility functions to convert a pointer to a list of optional values
template <class T, class U>
inline c10::optional<c10::ArrayRef<T>> pointer_to_optional_list(
    U** ptr,
    int64_t len) {
  return ptr
      ? c10::make_optional<c10::ArrayRef<T>>(pointer_to_list<T>(*ptr, len))
      : c10::nullopt;
}

} // namespace torch::aot_inductor
