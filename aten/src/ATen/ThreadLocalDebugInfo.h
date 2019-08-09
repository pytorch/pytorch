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

CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() noexcept;

// Sets debug information, returns the previously set thread local debug information
CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> info) noexcept;

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
