#pragma once

#ifdef TORCH_STANDALONE
#include <torch/standalone/macros/Macros.h>

#include <sstream>
#include <string>

// In the standalone version, TORCH_STANDALONE_CHECK throws std::runtime_error
// instead of c10::Error, because c10::Error transitively calls too
// much code to be implemented as header-only.

#ifdef STRIP_ERROR_MESSAGES
#define TORCH_STANDALONE_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " TORCH_STANDALONE_STRINGIZE(__FILE__))
#define TORCH_STANDALONE_CHECK(cond, ...)                \
  if (TORCH_STANDALONE_UNLIKELY_OR_CONST(!(cond))) {     \
    throw std::runtime_error(TORCH_STANDALONE_CHECK_MSG( \
        cond,                                            \
        "",                                              \
        __func__,                                        \
        ", ",                                            \
        __FILE__,                                        \
        ":",                                             \
        __LINE__,                                        \
        ", ",                                            \
        __VA_ARGS__));                                   \
  }

#else // STRIP_ERROR_MESSAGES
namespace torch::standalone::detail {
template <typename... Args>
std::string torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  // This is similar to the one in c10/util/Exception.h, but does
  // not depend on the more complex c10::str() function. ostringstream
  // supports less data types than c10::str(), but should be sufficient
  // in the standalone world.
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}
inline TORCH_STANDALONE_API const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline TORCH_STANDALONE_API const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}
} // namespace torch::standalone::detail

#define TORCH_STANDALONE_CHECK_MSG(cond, type, ...)        \
  (::torch::standalone::detail::torchCheckMsgImpl(         \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#define TORCH_STANDALONE_CHECK(cond, ...)                \
  if (TORCH_STANDALONE_UNLIKELY_OR_CONST(!(cond))) {     \
    throw std::runtime_error(TORCH_STANDALONE_CHECK_MSG( \
        cond,                                            \
        "",                                              \
        __func__,                                        \
        ", ",                                            \
        __FILE__,                                        \
        ":",                                             \
        __LINE__,                                        \
        ", ",                                            \
        ##__VA_ARGS__));                                 \
  }

#endif // STRIP_ERROR_MESSAGES

#else // TORCH_STANDALONE
#include <c10/util/Exception.h>

#define TORCH_STANDALONE_CHECK(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)

#endif // TORCH_STANDALONE
