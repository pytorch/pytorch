#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <torch/csrc/distributed/c10d/RankLocal.hpp>

namespace {

// Each rank operates on a different `c10d::ProcessGroup` instance for the same
// logical process group. Use `RankLocal<GroupRegistry>::get()` to ensure each
// rank gets a unique registry.
class GroupRegistry {
 public:
  void register_group(
      const std::string& group_name,
      c10::intrusive_ptr<c10d::ProcessGroup> group) {
    std::unique_lock write_lock(lock_);
    auto [_, inserted] = registry_.emplace(group_name, group);
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

  void unregister_group(const std::string& group_name) {
    std::unique_lock write_lock(lock_);
    registry_.erase(group_name);
  }

  void unregister_all_groups() {
    std::unique_lock write_lock(lock_);
    registry_.clear();
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
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
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

void unregister_process_group(const std::string& group_name) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().unregister_group(group_name);
  } else {
    process_registry.unregister_group(group_name);
  }
}

void unregister_all_process_groups() {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().unregister_all_groups();
  } else {
    process_registry.unregister_all_groups();
  }
}

} // namespace c10d
