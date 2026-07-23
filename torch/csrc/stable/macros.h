#pragma once
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/macros/Macros.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#include <torch/headeronly/util/win32-headers.h>
#else
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
// Look up a symbol already loaded in the current process. Used to reach
// stable-ABI shims that were added after the extension's TORCH_TARGET_VERSION
// when the running libtorch is new enough to provide them. Returns nullptr if
// the symbol is absent or the platform has no in-process dynamic lookup. This
// is the single place that holds the per-platform lookup; the macros below are
// platform agnostic.
[[maybe_unused]] static void* lookup_stable_symbol(const char* name) {
#if defined(C10_MOBILE)
  (void)name;
  return nullptr;
#elif defined(_WIN32)
  // Windows has no RTLD_DEFAULT equivalent, so resolve against the modules that
  // export the stable shims. Today these live in torch_cpu.dll and
  // torch_cuda.dll, so try each in turn. GetModuleHandleW does not take a
  // reference on the returned handle.
  const wchar_t* dlls[] = {L"torch_cpu.dll", L"torch_cuda.dll"};
  for (const wchar_t* dll : dlls) {
    HMODULE handle = GetModuleHandleW(dll);
    if (handle != nullptr) {
      void* sym = reinterpret_cast<void*>(GetProcAddress(handle, name));
      if (sym != nullptr) {
        return sym;
      }
    }
  }
  return nullptr;
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

// Fallback for the const char* exception getters, returns a nullptr. A fallback
// MUST share the exact signature (return type and arguments) of the shim it
// backs: the looked-up pointer is called through decltype(&FALLBACK_FUNCTION).
[[maybe_unused]] static const char* torch_shim_bc_const_char_ptr() {
  return nullptr;
}
HIDDEN_NAMESPACE_END(torch, stable, detail)

// IMPORTANT: this macro is intended for RARE, EXCEPTIONAL use only. It performs
// an at-runtime symbol lookup (dlsym / GetProcAddress), which is relatively
// expensive and only works on platforms that support dynamic symbol lookup.
// Reach for it only when an older-target extension genuinely needs a newer shim
// that would otherwise be unavailable (e.g. richer error messages); prefer
// normal version-guarded shim calls everywhere else.
//
// Calls SHIM_FUNCTION via the runtime lookup when the running libtorch is at
// least VERSION_SHIM_PRESENT, otherwise calls FALLBACK_FUNCTION.
//
// We structure this macro as an immediately-invoked lambda so it can 'return' a
// value from the macro call. The called pointer type is derived from
// FALLBACK_FUNCTION, which must therefore share SHIM_FUNCTION's exact signature
// (including arguments).
#define TORCH_DYNAMIC_VERSION_CALL(                                    \
    VERSION_SHIM_PRESENT, SHIM_FUNCTION, FALLBACK_FUNCTION, ...)       \
  ([&]() {                                                             \
    if (aoti_torch_abi_version() >= (VERSION_SHIM_PRESENT)) {          \
      void* fn_ptr =                                                   \
          torch::stable::detail::lookup_stable_symbol(#SHIM_FUNCTION); \
      if (fn_ptr != nullptr) {                                         \
        using FunctionType = decltype(&FALLBACK_FUNCTION);             \
        return reinterpret_cast<FunctionType>(fn_ptr)(__VA_ARGS__);    \
      }                                                                \
    }                                                                  \
    return FALLBACK_FUNCTION(__VA_ARGS__);                             \
  })()

// Entry point for dynamic version calls for shims added in 2.13.0. Putting the
// version expectation in the macro lets us bypass the symbol lookup entirely
// when the target version already includes the shim and call it directly;
// otherwise we fall through to the runtime lookup.
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0
// Target version already includes the shim, call it directly.
#define TORCH_DYNAMIC_VERSION_CALL_2_13_0( \
    SHIM_FUNCTION, FALLBACK_FUNCTION, ...) \
  ([&]() { return SHIM_FUNCTION(__VA_ARGS__); })()
#else
// Target version predates the shim, try a dynamic lookup.
#define TORCH_DYNAMIC_VERSION_CALL_2_13_0( \
    SHIM_FUNCTION, FALLBACK_FUNCTION, ...) \
  TORCH_DYNAMIC_VERSION_CALL(              \
      TORCH_VERSION_2_13_0, SHIM_FUNCTION, FALLBACK_FUNCTION, __VA_ARGS__)
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_BEGIN(torch, stable, detail)
[[maybe_unused]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  const auto& error_msg_without = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
      torch_exception_get_what_without_backtrace, torch_shim_bc_const_char_ptr);
  if (error_msg_without) {
    ss << error_msg_without;
    ss << " (originally from " << call << " API call failed at " << file
       << ", line " << line << ")";
  } else {
    ss << call << " API call failed at " << file << ", line " << line;
  }

  const auto& error_msg = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
      torch_exception_get_what, torch_shim_bc_const_char_ptr);
  if (error_msg) {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::cerr << "[" << std::put_time(&tm, "%H:%M:%S") << " " << file << ":"
              << line
              << "] Exception across libtorch C API boundary: " << error_msg;
  }

  throw std::runtime_error(ss.str());
}
HIDDEN_NAMESPACE_END(torch, stable, detail)

// This macro is similar to the header-only macro TORCH_ERROR_CODE_CHECK,
// but this macro is NOT header-only! It depends on the stable ABI but
// provides more info in the exception, including the error message as
// retrieved through the c shims from the original error message.
#define STABLE_TORCH_ERROR_CODE_CHECK(call)                            \
  if ((call) != TORCH_SUCCESS) {                                       \
    torch::stable::detail::throw_exception(#call, __FILE__, __LINE__); \
  }
