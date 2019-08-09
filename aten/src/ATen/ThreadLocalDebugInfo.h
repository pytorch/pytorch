#pragma once

#include <c10/macros/Export.h>

#include <memory>
#include <string>

namespace at {

// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is supposed to be utilized
// by the user's code to pass extra information from higher layers (e.g. model id)
// down to the operator callbacks (e.g. used for logging)

class CAFFE2_API ThreadLocalDebugInfoBase {
 public:
  ThreadLocalDebugInfoBase() {}
  virtual ~ThreadLocalDebugInfoBase() {}
};

namespace internal {
CAFFE2_API extern thread_local std::shared_ptr<ThreadLocalDebugInfoBase> debug_info;
}

inline std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() noexcept {
  return internal::debug_info;
}

// Sets debug information, returns the previously set thread local debug information
inline std::shared_ptr<ThreadLocalDebugInfoBase> setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> info) noexcept {
  auto ret = std::move(internal::debug_info);
  internal::debug_info = std::move(info);
  return ret;
}

class CAFFE2_API DebugInfoGuard {
 public:
  inline DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfoBase> info) {
    prev_info_ = setThreadLocalDebugInfo(std::move(info));
  }

  inline ~DebugInfoGuard() {
    setThreadLocalDebugInfo(std::move(prev_info_));
  }

 private:
  std::shared_ptr<ThreadLocalDebugInfoBase> prev_info_;
};

} // namespace at
