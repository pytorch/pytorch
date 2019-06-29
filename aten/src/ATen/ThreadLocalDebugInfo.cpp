#include "ATen/ThreadLocalDebugInfo.h"

namespace at {

namespace {
thread_local std::shared_ptr<ThreadLocalDebugInfoBase> debug_info;
}

std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() {
  return debug_info;
}

void setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> info) {
  debug_info = info;
}

} // namespace at
