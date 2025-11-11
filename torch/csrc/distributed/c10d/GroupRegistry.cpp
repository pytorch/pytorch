#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <torch/csrc/distributed/c10d/RankLocal.hpp>

namespace {

constexpr std::array<char, 9>
    _rank_map_prefix{'r', 'a', 'n', 'k', '_', 'm', 'a', 'p', ':'};

// Each rank operates on a different `c10d::ProcessGroup` instance for the same
// logical process group. Use `RankLocal<GroupRegistry>::get()` to ensure each
// rank gets a unique registry.
class GroupRegistry {
 public:
  void set_global_rank(int rank) {
    std::unique_lock write_lock(lock_);
    global_rank_ = rank;
    // Clear the cache since cached values are for the old rank
    rank_map_cache_.clear();
  }

  void register_group(
      const std::string& group_name,
      c10::intrusive_ptr<c10d::ProcessGroup> group) {
    std::unique_lock write_lock(lock_);
    auto [_, inserted] = registry_.try_emplace(group_name, std::move(group));
    TORCH_CHECK(
        inserted,
        "A process group is already registered under the name",
        group_name);
  }

  c10::intrusive_ptr<c10d::ProcessGroup> resolve_group(
      const std::string& group_name) {
    if (_is_rank_map(group_name)) {
      // If this group is a rank_map then we need to extract our rank's value
      // out of it and then resolve that.
      return resolve_group(_deref_rank_map(group_name));
    }

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
  int global_rank_ = -1;

  // Memoization cache for rank map lookups (maps rank_map string to resolved
  // value)
  std::unordered_map<std::string, std::string> rank_map_cache_;

  /// Return true if the string represents a rank map ("rank_map:...")
  static bool _is_rank_map(const std::string& s) {
    if (s.size() <= _rank_map_prefix.size()) {
      return false;
    }
    return strncmp(
               s.data(), _rank_map_prefix.data(), _rank_map_prefix.size()) == 0;
  }

  /// Deserialize a rank map string into a real map.
  static std::unordered_map<int, std::string> _decode_rank_map(
      const std::string& s) {
    // "rank_map:0:a,1:b,2:cdef" -> {0: "a", 1: "b", 2: "cdef"}
    std::unordered_map<int, std::string> result;
    // Start just before the initial colon.
    const char* ptr = s.data() + _rank_map_prefix.size() - 1;
    const char* const end = s.data() + s.size();

    while (ptr < end) {
      ++ptr; // Skip ':' or ','

      // Parse the rank number
      int rank = 0;
      while (ptr < end && isdigit(*ptr)) {
        rank = rank * 10 + (*ptr++ - '0');
      }

      // Expect a colon
      TORCH_CHECK(
          ptr < end && *ptr == ':',
          "Invalid rank_map format: expected ':' after rank");
      ++ptr;

      // Find the end of the value (either ',' or end of string)
      const char* const value_start = ptr;
      while (ptr < end && *ptr != ',') {
        ++ptr;
      }

      // Extract the value
      result[rank] = std::string(value_start, ptr);
    }

    return result;
  }

  /// Extract the value for the current rank from a rank map string.
  std::string _deref_rank_map(const std::string& s) {
    std::unique_lock write_lock(lock_);

    // Check if we have the result cached for this rank_map
    auto cache_it = rank_map_cache_.find(s);
    if (cache_it != rank_map_cache_.end()) {
      return cache_it->second;
    }

    // Not cached, decode and look up the value for our rank
    auto map = _decode_rank_map(s);
    auto it = map.find(global_rank_);
    TORCH_CHECK(it != map.end(), "Could not resolve rank in rank_map");

    // Cache the result for this rank_map and return it
    rank_map_cache_[s] = it->second;
    return it->second;
  }
};

} // namespace

namespace c10d {

static bool thread_isolation_mode = false;
static GroupRegistry process_registry;

void set_global_rank(int rank) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().set_global_rank(rank);
  } else {
    process_registry.set_global_rank(rank);
  }
}

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
