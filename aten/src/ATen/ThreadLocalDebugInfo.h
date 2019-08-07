#pragma once

#include <c10/macros/Export.h>

#include <memory>
#include <string>

namespace at {

class CAFFE2_API ThreadLocalDebugInfoBase {
 public:
  ThreadLocalDebugInfoBase() {}
  virtual ~ThreadLocalDebugInfoBase() {}
};

// Thread local debug information is propagated across the forward
// (including async fork tasks) and backward passes and is used to pass extra
// information from higher layers (e.g. model id) down to the operator callbacks

CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() noexcept;

// Sets debug information, returns the previously set thread local debug information
CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> debug_info) noexcept;

} // namespace at
