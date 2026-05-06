#pragma once
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/macros/Macros.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef _WIN32
#include <dlfcn.h>
#endif

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

HIDDEN_NAMESPACE_BEGIN(torch, stable, detail)
// Fallback for functions that produce const char* types, returns a nullptr.
[[maybe_unused]] static const char* torch_shim_bc_const_char_ptr(...) {
  return nullptr;
}
HIDDEN_NAMESPACE_END(torch, stable, detail)

// This macro allows for an at-runtime lookup of a symbol, this is relatively
// expensive and should only be done in exceptional circumstances, it only
// supports platforms that have dlsym available.
//
// We structure these macro's as immediately invoked lambda's to be able to
// 'return' a value from the macro call.
//
// Return type is inferred from the return value of FALLBACK_FUNCTION.
#if !defined(C10_MOBILE) && !defined(_WIN32)
#define TORCH_DYNAMIC_VERSION_CALL(                                   \
    VERSION_SHIM_PRESENT, FALLBACK_FUNCTION, SHIM_FUNCTION, ...)      \
  ([]() {                                                             \
    if (aoti_torch_abi_version() >= VERSION_SHIM_PRESENT) {           \
      const auto fn_ptr = dlsym(RTLD_DEFAULT, #SHIM_FUNCTION);        \
      if (fn_ptr) {                                                   \
        using ReturnType =                                            \
            std::invoke_result_t<decltype(&FALLBACK_FUNCTION), void>; \
        const auto typed_ptr =                                        \
            reinterpret_cast<ReturnType (*)(__VA_ARGS__)>(fn_ptr);    \
        return (typed_ptr)(__VA_ARGS__);                              \
      }                                                               \
    }                                                                 \
    return FALLBACK_FUNCTION(__VA_ARGS__);                            \
  })()
#else
// On a platform without dlsym, so immediately call the fallback function.
#define TORCH_DYNAMIC_VERSION_CALL(                              \
    VERSION_SHIM_PRESENT, FALLBACK_FUNCTION, SHIM_FUNCTION, ...) \
  ([]() { return FALLBACK_FUNCTION(__VA_ARGS__); })()
#endif

// Entry point for dynamic version calls that exist from 2.13.0 and onward.
// Putting the version expectation in the macro ensures we can bypass the symbol
// lookup if the target version exceeds the version in which this shim function
// was added.
#if TORCH_TARGET_VERSION >= TORCH_VERSION_2_13_0
// Target version is greater than version that added the method, we can call the
// shim.
#define TORCH_DYNAMIC_VERSION_CALL_2_13_0( \
    FALLBACK_FUNCTION, SHIM_FUNCTION, ...) \
  ([]() { return SHIM_FUNCTION(__VA_ARGS__); })()
#else
// Target version is less than the version that added the shim, try a dynamic
// lookup.
#define TORCH_DYNAMIC_VERSION_CALL_2_13_0( \
    FALLBACK_FUNCTION, SHIM_FUNCTION, ...) \
  TORCH_DYNAMIC_VERSION_CALL(              \
      TORCH_VERSION_2_13_0, FALLBACK_FUNCTION, SHIM_FUNCTION, __VA_ARGS__)
#endif

HIDDEN_NAMESPACE_BEGIN(torch, stable, detail)
[[maybe_unused]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  const auto& error_msg_without = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
      torch_shim_bc_const_char_ptr, torch_exception_get_what_without_backtrace);
  if (error_msg_without) {
    ss << ", with: " << error_msg_without;
  }

  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::cerr << "[" << std::put_time(&tm, "%H:%M:%S") << " " << file << ":"
            << line;

  const auto& error_msg = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
      torch_shim_bc_const_char_ptr, torch_exception_get_what);
  if (error_msg) {
    std::cerr << "] Exception in aoti_torch: " << error_msg;
  }
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
