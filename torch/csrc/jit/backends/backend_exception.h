#pragma once
#include <c10/util/Exception.h>

namespace c10 {
class TORCH_API BackendRuntimeException : public c10::Error {
 public:
  // Use debug_handle to throw exception
  BackendRuntimeException(
      SourceLocation loc,
      std::string msg,
      int64_t debug_handle)
      : c10::Error(loc, msg) {
    debug_handles.push_back(debug_handle);
  }
  // If rethrowing, can push another debug_handle
  // This is useful in couple of scenarios.
  // 1. A submodule is lowered and lite interperter has CallMethod
  //    to lowered module's method. In this case lowered module will throw with
  //    a handle, plus there will be another debug handle corresponding
  //    to the CallMethod node in lite interpreter. Both together give complete
  //    trace. This function allows lite interpreter to rethrow with debug
  //    handle it has for CallMethod.
  // 2. Another scenarios is when lite interperter can make function calls or
  //    the lowered backend also has function call ability. Thus we have
  //    multiple function frames. Now we need a stack of handles to symbolicate
  //    entire stack trace.
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
    throw ::c10::BackendRuntimeException(                      \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        msg,                                                   \
        debug_handle);                                         \
  }

#define TORCH_DELEGATED_BACKEND_RETHROW(e, debug_handle) \
  do {                                                   \
    e.pushDebugHandle(debug_handle);                     \
    throw;                                               \
  } while (false)

#define DEBUG_HANDLE_UNKNOWN -1
