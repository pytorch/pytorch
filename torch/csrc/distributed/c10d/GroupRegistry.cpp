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

 private:
  std::map<std::string, c10::weak_intrusive_ptr<c10d::ProcessGroup>> registry_;
  std::shared_mutex lock_;
};

} // namespace

namespace c10d {

void register_process_group(
    const std::string& group_name,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  RankLocal<::GroupRegistry>::get().register_group(group_name, group);
}

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name) {
  return RankLocal<::GroupRegistry>::get().resolve_group(group_name);
}

} // namespace c10d
