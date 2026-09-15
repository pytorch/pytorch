#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <mutex>
#include <vector>

#include <torch/csrc/distributed/c10d/RankLocal.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace {

// Each rank operates on a different `c10d::ProcessGroup` instance for the same
// logical process group. Use `RankLocal<GroupRegistry>::get()` to ensure each
// rank gets a unique registry.
class GroupRegistry {
 public:
  void register_group(
      const std::string& group_name,
      const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
    std::unique_lock write_lock(lock_);
    // By reference, not by value and moved: registry_ holds weak pointers and
    // weak_intrusive_ptr only converts from a const intrusive_ptr&, so a move
    // here never moved -- it just cost a refcount round trip per call.
    auto [_, inserted] = registry_.try_emplace(group_name, group);
    TORCH_CHECK(
        inserted,
        "A process group is already registered under the name",
        group_name);
  }

  c10::intrusive_ptr<c10d::ProcessGroup> resolve_group(
      const std::string& group_name) {
    std::shared_lock read_lock(lock_);
    auto it = registry_.find(group_name);
    TORCH_CHECK(
        it != registry_.end(),
        "Could not resolve the process group registered under the name ",
        group_name);

    auto group = it->second.lock();
    TORCH_CHECK(
        group != nullptr,
        "Process group registered under the name ",
        group_name,
        " has already been destroyed.");
    return group;
  }

  bool has_group(const std::string& group_name) {
    std::shared_lock read_lock(lock_);
    auto it = registry_.find(group_name);
    return it != registry_.end() && it->second.lock() != nullptr;
  }

  void unregister_group(const std::string& group_name) {
    std::unique_lock write_lock(lock_);
    registry_.erase(group_name);
  }

  std::vector<std::string> unregister_all_groups() {
    std::unique_lock write_lock(lock_);
    std::vector<std::string> names;
    names.reserve(registry_.size());
    for (const auto& entry : registry_) {
      names.push_back(entry.first);
    }
    registry_.clear();
    return names;
  }

 private:
  std::map<std::string, c10::weak_intrusive_ptr<c10d::ProcessGroup>> registry_;
  std::shared_mutex lock_;
};

} // namespace

namespace c10d {

static bool thread_isolation_mode = false;
static GroupRegistry process_registry;

void set_thread_isolation_mode(bool enable) {
  thread_isolation_mode = enable;
}

bool get_thread_isolation_mode() {
  return thread_isolation_mode;
}

void register_process_group(
    const std::string& group_name,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().register_group(group_name, group);
  } else {
    process_registry.register_group(group_name, group);
  }
}

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name) {
  if (thread_isolation_mode) {
    return RankLocal<::GroupRegistry>::get().resolve_group(group_name);
  } else {
    return process_registry.resolve_group(group_name);
  }
}

bool is_process_group_registered(const std::string& group_name) {
  if (thread_isolation_mode) {
    return RankLocal<::GroupRegistry>::get().has_group(group_name);
  } else {
    return process_registry.has_group(group_name);
  }
}

namespace {

// Function-local so the list is built before any registration can reach it,
// whatever order the translation units initialize in.
std::vector<std::function<void(const std::string&)>>& unregister_hooks() {
  static std::vector<std::function<void(const std::string&)>> hooks;
  return hooks;
}

std::mutex& unregister_hooks_mutex() {
  static std::mutex m;
  return m;
}

void run_unregister_hooks(const std::string& group_name) {
  // Copy, then call with the lock released: a hook runs arbitrary consumer
  // code, and holding a non-recursive mutex across it deadlocks the moment
  // one of them registers a hook or unregisters a group.
  std::vector<std::function<void(const std::string&)>> hooks;
  {
    std::lock_guard<std::mutex> lock(unregister_hooks_mutex());
    hooks = unregister_hooks();
  }
  for (const auto& hook : hooks) {
    try {
      hook(group_name);
    } catch (const std::exception& e) {
      C10D_ERROR(
          "group unregister hook failed for '{}': {}", group_name, e.what());
    }
  }
}

} // namespace

void register_group_unregister_hook(
    std::function<void(const std::string&)> hook) {
  std::lock_guard<std::mutex> lock(unregister_hooks_mutex());
  unregister_hooks().push_back(std::move(hook));
}

void unregister_process_group(const std::string& group_name) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().unregister_group(group_name);
  } else {
    process_registry.unregister_group(group_name);
  }
  run_unregister_hooks(group_name);
}

void unregister_all_process_groups() {
  std::vector<std::string> names;
  if (thread_isolation_mode) {
    names = RankLocal<::GroupRegistry>::get().unregister_all_groups();
  } else {
    names = process_registry.unregister_all_groups();
  }
  for (const auto& name : names) {
    run_unregister_hooks(name);
  }
}

} // namespace c10d
