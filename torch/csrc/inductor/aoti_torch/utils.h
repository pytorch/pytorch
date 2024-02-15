#pragma once

#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
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
template <class T>
c10::optional<T> pointer_to_optional(T* ptr) {
  return ptr ? c10::make_optional(*ptr) : c10::nullopt;
}

template <class T, class U, typename = std::enable_if_t<!std::is_same_v<T, U>>>
c10::optional<T> pointer_to_optional(U* ptr) {
  return ptr ? c10::make_optional<T>(T(*ptr)) : c10::nullopt;
}

template <>
c10::optional<at::Tensor> pointer_to_optional(AtenTensorHandle* ptr) {
  return ptr ? c10::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : c10::nullopt;
}

} // namespace torch::aot_inductor