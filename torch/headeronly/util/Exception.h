#pragma once

#include <torch/headeronly/macros/Export.h>
#include <torch/headeronly/macros/Macros.h>

#include <sstream>
#include <string>

namespace c10 {
// On nvcc, C10_UNLIKELY thwarts missing return statement analysis.  In cases
// where the unlikely expression may be a constant, use this macro to ensure
// return statement analysis keeps working (at the cost of not getting the
// likely/unlikely annotation on nvcc).
// https://github.com/pytorch/pytorch/issues/21418
//
// Currently, this is only used in the error reporting macros below.  If you
// want to use it more generally, move me to Macros.h
//
// TODO: Brian Vaughan observed that we might be able to get this to work on
// nvcc by writing some sort of C++ overload that distinguishes constexpr inputs
// from non-constexpr.  Since there isn't any evidence that losing C10_UNLIKELY
// in nvcc is causing us perf problems, this is not yet implemented, but this
// might be an interesting piece of C++ code for an intrepid bootcamper to
// write.
#if defined(__CUDACC__)
#define C10_UNLIKELY_OR_CONST(e) e
#else
#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)
#endif

} // namespace c10

// STD_TORCH_CHECK throws std::runtime_error instead of c10::Error which is
// useful when certain headers are used in a libtorch-independent way,
// e.g. when Vectorized<T> is used in AOTInductor generated code, or
// for custom ops to have an ABI stable dependency on libtorch.
#ifdef STRIP_ERROR_MESSAGES
#define STD_TORCH_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
#else // so STRIP_ERROR_MESSAGES is not defined
namespace torch::headeronly::detail {
template <typename... Args>
std::string stdTorchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  // This is similar to the one in c10/util/Exception.h, but does
  // not depend on the more complex c10::str() function. ostringstream
  // supports fewer data types than c10::str(), but should be sufficient
  // in the headeronly world.
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}

inline const char* stdTorchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline const char* stdTorchCheckMsgImpl(const char* /*msg*/, const char* args) {
  return args;
}
} // namespace torch::headeronly::detail

#define STD_TORCH_CHECK_MSG(cond, type, ...)               \
  (torch::headeronly::detail::stdTorchCheckMsgImpl(        \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#endif // STRIP_ERROR_MESSAGES

#define STD_TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {           \
    throw std::runtime_error(STD_TORCH_CHECK_MSG( \
        cond,                                     \
        "",                                       \
        __func__,                                 \
        ", ",                                     \
        __FILE__,                                 \
        ":",                                      \
        __LINE__,                                 \
        ", ",                                     \
        ##__VA_ARGS__));                          \
  }
