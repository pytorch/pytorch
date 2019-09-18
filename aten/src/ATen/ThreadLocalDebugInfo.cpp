#include "ATen/ThreadLocalDebugInfo.h"

namespace at {

namespace {
thread_local std::shared_ptr<ThreadLocalDebugInfoBase> debug_info;
}

std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo() noexcept {
  return debug_info;
}

std::shared_ptr<ThreadLocalDebugInfoBase> setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> info) noexcept {
  auto ret = std::move(debug_info);
  debug_info = std::move(info);
  return ret;
}

} // namespace at
