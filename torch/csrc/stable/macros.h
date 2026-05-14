#pragma once
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/macros/Macros.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// Users of this macro are expected to include cuda_runtime.h
#define STD_CUDA_CHECK(EXPR)                      \
  do {                                            \
    const cudaError_t __err = EXPR;               \
    char* __error_msg = nullptr;                  \
    torch_c10_cuda_check_msg(                     \
        static_cast<int32_t>(__err),              \
        __FILE__,                                 \
        __func__,                                 \
        static_cast<uint32_t>(__LINE__),          \
        true,                                     \
        &__error_msg);                            \
    if (__error_msg != nullptr) {                 \
      std::string __msg(__error_msg);             \
      torch_c10_cuda_free_error_msg(__error_msg); \
      throw std::runtime_error(__msg);            \
    }                                             \
  } while (0)

// Users of this macro are expected to include cuda_runtime.h
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(cudaGetLastError())

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_BEGIN(torch, stable, detail)
[[maybe_unused]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << torch_exception_get_what_without_backtrace();
  ss << " (originally from " << call << " API call failed at " << file
     << ", line " << line << ")";

  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::cerr << "[" << std::put_time(&tm, "%H:%M:%S") << " " << file << ":"
            << line << "] Exception across libtorch C API boundary: "
            << torch_exception_get_what();
  throw std::runtime_error(ss.str());
}
HIDDEN_NAMESPACE_END(torch, stable, detail)

// This macro is similar to the header-only macro TORCH_ERROR_CODE_CHECK, but
// this macro is NOT header-only! It depends on the stable ABI but provides more
// info in the exception, including the error message as retrieved through the c
// shims from the original error message.
#define STABLE_TORCH_ERROR_CODE_CHECK(call)                            \
  if ((call) != TORCH_SUCCESS) {                                       \
    torch::stable::detail::throw_exception(#call, __FILE__, __LINE__); \
  }

#else
#define STABLE_TORCH_ERROR_CODE_CHECK(call) TORCH_ERROR_CODE_CHECK(call)
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0
