#include "ATen/ThreadLocalDebugInfo.h"

namespace at {

namespace {
struct DebugInfo {
  std::shared_ptr<ThreadLocalDebugInfoBase> debug_info = nullptr;
};

DebugInfo& info() {
  static DebugInfo info;
  return info;
}
}

std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() {
  return info().debug_info;
}

void setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> debug_info) {
  info().debug_info = debug_info;
}

} // namespace at
