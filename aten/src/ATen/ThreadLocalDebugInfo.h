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

CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo();

// returns the previously set thread local debug info
CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> debug_info);

} // namespace at
