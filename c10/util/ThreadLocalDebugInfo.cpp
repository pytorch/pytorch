#include <c10/util/Exception.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <utility>

namespace c10 {

C10_DEFINE_TLS_static(std::shared_ptr<ThreadLocalDebugInfo>, tls_debug_info);
#define debug_info (tls_debug_info.get())

/* static */
DebugInfoBase* ThreadLocalDebugInfo::get(DebugInfoKind kind) {
  ThreadLocalDebugInfo* cur = debug_info.get();
  while (cur) {
    if (cur->kind_ == kind) {
      return cur->info_.get();
    }
    cur = cur->parent_info_.get();
  }
  return nullptr;
}

/* static */
std::shared_ptr<ThreadLocalDebugInfo> ThreadLocalDebugInfo::current() {
  return debug_info;
}

/* static */
void ThreadLocalDebugInfo::_forceCurrentDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfo> info) {
  debug_info = std::move(info);
}

/* static */
void ThreadLocalDebugInfo::_push(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  auto prev_info = debug_info;
  debug_info = std::make_shared<ThreadLocalDebugInfo>();
  debug_info->parent_info_ = prev_info;
  debug_info->kind_ = kind;
  debug_info->info_ = std::move(info);
}

/* static */
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_pop(DebugInfoKind kind) {
  TORCH_CHECK(
      debug_info && debug_info->kind_ == kind,
      "Expected debug info of type ",
      (size_t)kind);
  auto res = debug_info;
  debug_info = debug_info->parent_info_;
  return res->info_;
}

/* static */
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_peek(DebugInfoKind kind) {
  TORCH_CHECK(
      debug_info && debug_info->kind_ == kind,
      "Expected debug info of type ",
      (size_t)kind);
  return debug_info->info_;
}

DebugInfoGuard::DebugInfoGuard(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  if (!info) {
    return;
  }
  prev_info_ = debug_info;
  ThreadLocalDebugInfo::_push(kind, std::move(info));
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
DebugInfoGuard::DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info) {
  if (!info) {
    return;
  }
  prev_info_ = std::move(debug_info);
  debug_info = std::move(info);
  active_ = true;
}

} // namespace c10
