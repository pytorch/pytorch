#include <ATen/ThreadLocalDebugInfo.h>

namespace at {

namespace {
thread_local std::shared_ptr<ThreadLocalDebugInfo> debug_info = nullptr;
}

/* static */
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::get(
    DebugInfoKind kind) {
  auto cur = debug_info;
  while (cur) {
    if (cur->kind_ == kind) {
      return cur->debug_info_;
    }
    cur = cur->parent_info_;
  }
  return nullptr;
}

/* static */
std::shared_ptr<ThreadLocalDebugInfo> ThreadLocalDebugInfo::_current() {
  return debug_info;
}

/* static */
void ThreadLocalDebugInfo::_forceCurrentDebugInfo(
    const std::shared_ptr<ThreadLocalDebugInfo>& info) {
  debug_info = info;
}

DebugInfoGuard::DebugInfoGuard(
    DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info) {
  if (!info) {
    return;
  }
  TORCH_CHECK(!ThreadLocalDebugInfo::get(kind), "Debug info is already set");
  prev_info_ = debug_info;
  debug_info = std::make_shared<ThreadLocalDebugInfo>();
  debug_info->parent_info_ = prev_info_;
  debug_info->kind_ = kind;
  debug_info->debug_info_ = info;
  active_ = true;
}

DebugInfoGuard::~DebugInfoGuard() {
  if (active_) {
    debug_info = prev_info_;
  }
}

// Used only for setting a debug info after crossing the thread boundary;
// in this case we assume that thread pool's thread does not have an
// active debug info
DebugInfoGuard::DebugInfoGuard(
    std::shared_ptr<ThreadLocalDebugInfo> info) {
  if (!info) {
    return;
  }
  prev_info_ = debug_info;
  debug_info = info;
  active_ = true;
}

} // namespace at
