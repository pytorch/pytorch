#pragma once
#include <c10/util/Exception.h>

namespace c10 {
class DelegatedBackendRuntimeException : public c10::Error {
 public:
  // Use debug_handle to throw exception
  DelegatedBackendRuntimeException(
      SourceLocation loc,
      std::string msg,
      int64_t debug_handle)
      : c10::Error(loc, msg) {
    debug_handles.push_back(debug_handle);
  }
  // If rethrowing, can push another debug_handle
  void pushDebugHandle(int64_t debug_handle) {
    debug_handles.push_back(debug_handle);
  }
  const std::vector<int64_t>& getDebugHandles() {
    return debug_handles;
  }

 private:
  // Stores stack of debug handles.
  std::vector<int64_t> debug_handles;
};

} // namespace c10
#define TORCH_DELEGATED_BACKEND_THROW(cond, msg, debug_handle) \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                        \
    throw ::c10::DelegatedBackendRuntimeException(             \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        msg,                                                   \
        debug_handle);                                         \
  }

#define TORCH_DELEGATED_BACKEND_RETHROW(e, debug_handle) \
  do {                                                   \
    e.pushDebugHandle(debug_handle);                     \
    throw;                                               \
  } while (false)
