#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/macros/Macros.h>

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
  ss << call << " API call failed at " << file << ", line " << line;
  ss << ", with: " << torch_exception_get_what_without_backtrace();
  throw std::runtime_error(ss.str());
}
HIDDEN_NAMESPACE_END(torch, stable, detail)

// Worker function for the error check, its first argument is a unique variable
// name that is used to store the previous printing state, we do this to ensure
// that we don't accidentally shadow variables in the outer scope.
#define STABLE_TORCH_ERROR_CODE_CHECK_IMPL(variable_name, call)               \
  {                                                                           \
    const bool variable_name = torch_exception_set_exception_printing(false); \
    if ((call) != TORCH_SUCCESS) {                                            \
      torch_exception_set_exception_printing(variable_name);                  \
      torch::stable::detail::throw_exception(#call, __FILE__, __LINE__);      \
    }                                                                         \
    torch_exception_set_exception_printing(variable_name);                    \
  }

// This macro is similar to the header-only macro TORCH_ERROR_CODE_CHECK, but
// this macro is NOT header-only! It depends on the stable ABI but provides more
// info in the exception, including the error message as retrieved through the c
// shims from the original error message.
#define STABLE_TORCH_ERROR_CODE_CHECK(call) \
  STABLE_TORCH_ERROR_CODE_CHECK_IMPL(       \
      C10_ANONYMOUS_VARIABLE(previous_exception_printing), call)

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0
