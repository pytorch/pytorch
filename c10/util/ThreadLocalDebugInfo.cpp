#include <c10/util/Exception.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/ThreadLocalDebugInfo.h>

namespace c10 {

C10_DEFINE_TLS_static(std::shared_ptr<ThreadLocalDebugInfo>, tls_debug_info);
#define debug_info (tls_debug_info.get())

// Thread local values are default initialized. However we must manually zero
// the lookup cache storage when using `c10::ThreadLocal`.
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

struct Cache {
  Cache() { storage.fill(nullptr); }
  ThreadLocalDebugInfo::lookup_cache_t storage_;
};
static ::c10::ThreadLocal<Cache> tls_lookup_cache
#define lookup_cache (tls_lookup_cache->storage_)

#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
static thread_local ThreadLocalDebugInfo::lookup_cache_t lookup_cache;
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

/* static */
DebugInfoBase* ThreadLocalDebugInfo::get(DebugInfoKind kind) {
  const auto index = static_cast<size_t>(kind);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(index < NUM_DEBUG_INFO_KINDS);
  return lookup_cache[index];
}

/* static */
std::shared_ptr<ThreadLocalDebugInfo> ThreadLocalDebugInfo::current() {
  return debug_info;
}

/* static */
void ThreadLocalDebugInfo::_forceCurrentDebugInfo(
    const std::shared_ptr<ThreadLocalDebugInfo>& info) {
  debug_info = info;
  lookup_cache = ThreadLocalDebugInfo::_make_lookup_cache();
}

/* static */
void ThreadLocalDebugInfo::_push(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  auto prev_info = debug_info;
  debug_info = std::make_shared<ThreadLocalDebugInfo>();
  debug_info->parent_info_ = prev_info;
  debug_info->kind_ = kind;
  debug_info->info_ = info;
  lookup_cache[static_cast<size_t>(kind)] = info.get();
}

/* static */
std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_pop(DebugInfoKind kind) {
  TORCH_CHECK(
      debug_info && debug_info->kind_ == kind,
      "Expected debug info of type ",
      (size_t)kind);
  auto res = debug_info;
  debug_info = debug_info->parent_info_;
  lookup_cache = ThreadLocalDebugInfo::_make_lookup_cache();
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

/* static */
ThreadLocalDebugInfo::lookup_cache_t ThreadLocalDebugInfo::
    _make_lookup_cache() {
  ThreadLocalDebugInfo* cur = debug_info.get();

  lookup_cache_t out;
  out.fill(nullptr);
  while (cur) {
    const auto index = static_cast<size_t>(cur->kind_);
    if (out[index] == nullptr) {
      out[index] = cur->info_.get();
    }
    cur = cur->parent_info_.get();
  }
  return out;
}

DebugInfoGuard::DebugInfoGuard(
    DebugInfoKind kind,
    std::shared_ptr<DebugInfoBase> info) {
  if (!info) {
    return;
  }
  prev_info_ = debug_info;
  ThreadLocalDebugInfo::_push(kind, info);
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
  prev_info_ = debug_info;
  debug_info = info;
  active_ = true;
}

} // namespace c10
