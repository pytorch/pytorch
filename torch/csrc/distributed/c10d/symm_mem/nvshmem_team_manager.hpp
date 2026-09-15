#pragma once

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Starting from NVSHMEM 3.3.9, nvshmem_host.h exists so that we can cleanly
// include only the nvshmem host library headers:
// #include <nvshmem_host.h>
// It translates into the following two lines:
#if !defined(USE_ROCM)
#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
#else
#include <rocshmem/rocshmem.hpp>
#endif
// For maximum compatibility, we use the "host/" style for now.

namespace c10d::nvshmem_extension {

// This corresponds to max nblocks
constexpr int MAX_N_TEAMS = 128;

// A pool of teams for each group. These are duplicate teams.
using TeamPool = std::vector<nvshmem_team_t>;

// Manage all the team business. Singleton.
class TeamManager {
 public:
  // Constructor
  explicit TeamManager(const c10::Device device) : device_(device) {}

  // Get single, global manager.
  static TeamManager& get(const c10::Device device);

  // Release a group's teams without constructing the singleton if NVSHMEM
  // collectives have not been used in this process.
  static void release_group_if_initialized(const std::string& group_name);

  // Get a team for a group.
  nvshmem_team_t get_team(
      const std::string& group_name,
      const std::vector<int>& global_ranks) {
    auto [team_pool, pool_updated] =
        group_to_team_pool(group_name, global_ranks, 1);
    // Return the first available team
    return team_pool[0];
  }

  // Get n teams for a group.
  // The first element of the returned pair is the team pool on host side.
  // The second element of the returned pair is the team pool on device side.
  // This API must be called with a device guard.
  std::pair<const TeamPool&, nvshmem_team_t*> get_n_teams(
      const std::string& group_name,
      const std::vector<int>& global_ranks,
      const int need_n) {
    // A device guard is required for malloc and memcpy below
    c10::cuda::CUDAGuard guard(device_);
    // Get the team pool with the requested number of teams
    auto [team_pool, pool_updated] =
        group_to_team_pool(group_name, global_ranks, need_n);
    // Check if the pool already exists in device memory
    nvshmem_team_t* team_pool_dev = nullptr;
    constexpr auto pool_bytes = sizeof(nvshmem_team_t) * MAX_N_TEAMS;
    auto it = team_pool_devptrs_.find(group_name);
    if (it == team_pool_devptrs_.end()) {
      // If not, allocate a new pool in device memory
      team_pool_dev = reinterpret_cast<nvshmem_team_t*>(
          c10::cuda::CUDACachingAllocator::raw_alloc(pool_bytes));
      team_pool_devptrs_[group_name] = team_pool_dev;
    } else {
      team_pool_dev = it->second;
    }
    // Update the pool in device memory if host side pool is updated
    if (pool_updated) {
      TORCH_INTERNAL_ASSERT(team_pool.size() == MAX_N_TEAMS);
      auto stream = at::cuda::getCurrentCUDAStream();
      C10_CUDA_CHECK(cudaMemcpyAsync(
          team_pool_dev,
          team_pool.data(),
          pool_bytes,
          cudaMemcpyHostToDevice,
          stream));
    }
    return std::make_pair(std::cref(team_pool), team_pool_dev);
  }

  // Retire a group's team pool for exclusive reuse by a future process group
  // with the same membership. Team destruction is collective, while process
  // group destruction does not provide the ordering guarantees needed to call
  // it safely here.
  void release_group(const std::string& group_name) {
    auto team_it = group_name_to_team_pool_.find(group_name);
    if (team_it == group_name_to_team_pool_.end()) {
      return;
    }

    c10::cuda::CUDAGuard guard(device_);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    auto dev_it = team_pool_devptrs_.find(group_name);
    if (dev_it != team_pool_devptrs_.end()) {
      c10::cuda::CUDACachingAllocator::raw_delete(dev_it->second);
      team_pool_devptrs_.erase(dev_it);
    }

    auto ranks_it = group_name_to_global_ranks_.find(group_name);
    auto pool_id_it = group_name_to_pool_id_.find(group_name);
    TORCH_INTERNAL_ASSERT(ranks_it != group_name_to_global_ranks_.end());
    TORCH_INTERNAL_ASSERT(pool_id_it != group_name_to_pool_id_.end());
    auto& membership = membership_state(ranks_it->second);
    auto [_, inserted] = membership.pending_pools.emplace(
        pool_id_it->second, std::move(team_it->second));
    TORCH_INTERNAL_ASSERT(inserted);
    group_name_to_pool_id_.erase(pool_id_it);
    group_name_to_global_ranks_.erase(ranks_it);
    group_name_to_team_pool_.erase(team_it);
  }

  ~TeamManager() noexcept {
    // Free the team pools in device memory
    // Note that we do it in a best effort manner because the team pool is
    // managed by a static TeamManager and the destruction order of static
    // objects is undetermined. If the destructor is called after the CUDA
    // context is destroyed, cudaFree would fail.
    try {
      // cudaFree generally implies a device synchronization, meaning it will
      // block until all preceding CUDA operations on the device have completed
      // before freeing the memory. Thus we don't need to worry about freeing
      // the memory before CUDA kernels complete.
      for (auto& [_, team_pool_dev] : team_pool_devptrs_) {
        c10::cuda::CUDACachingAllocator::raw_delete(team_pool_dev);
      }
    } catch (...) {
      // Ignore the error
      std::cerr << "Failed to free the team pool in device memory, skipping\n";
    }
  }

 private:
  static constexpr uint64_t kNoPendingPool =
      std::numeric_limits<uint64_t>::max();

  struct MembershipState {
    std::vector<int> global_ranks;
    uint64_t next_pool_id{0};
    std::map<uint64_t, TeamPool> pending_pools;
  };

  MembershipState& membership_state(const std::vector<int>& global_ranks) {
    auto it = std::find_if(
        membership_states_.begin(),
        membership_states_.end(),
        [&](const auto& state) { return state.global_ranks == global_ranks; });
    if (it == membership_states_.end()) {
      membership_states_.push_back(MembershipState{global_ranks});
      return membership_states_.back();
    }
    return *it;
  }

  std::optional<std::pair<uint64_t, TeamPool>> take_reusable_team_pool(
      const std::string& group_name,
      MembershipState& membership) {
    auto group = c10d::resolve_process_group(group_name);
    c10d::symmetric_memory::StoreExchange exchange(
        "NVSHMEMTeamManagerReuse");
    uint64_t search_from = 0;
    while (true) {
      auto it = membership.pending_pools.lower_bound(search_from);
      auto candidate =
          it == membership.pending_pools.end() ? kNoPendingPool : it->first;
      auto candidates = exchange.all_gather(
          group->getStore(), group->getRank(), group->getSize(), candidate);

      uint64_t next_search_from = candidate;
      bool all_match = candidate != kNoPendingPool;
      for (const auto peer_candidate : candidates) {
        if (peer_candidate == kNoPendingPool) {
          return std::nullopt;
        }
        all_match &= peer_candidate == candidate;
        next_search_from = std::max(next_search_from, peer_candidate);
      }
      if (!all_match) {
        search_from = next_search_from;
        continue;
      }

      TORCH_INTERNAL_ASSERT(it != membership.pending_pools.end());
      auto result =
          std::make_pair(candidate, std::move(it->second));
      membership.pending_pools.erase(it);
      return result;
    }
  }

  // Get the team pool for a group. If the pool doesn't exist, create it. If the
  // pool exists but is not large enough, create more teams.
  // The first element of the returned pair is the team pool on host side.
  // The second element of the returned pair is a boolean indicating if the pool
  // is updated.
  std::pair<const TeamPool&, bool> group_to_team_pool(
      const std::string& group_name,
      const std::vector<int>& global_ranks,
      const int need_n) {
    TORCH_CHECK(need_n < MAX_N_TEAMS, "Too many teams requested");
    // Guarding the NVSHMEM API calls below just to be safe
    c10::cuda::CUDAGuard guard(device_);

    // Insert a new team pool if not exists. At this already-collective
    // boundary, first agree on a pool that every rank has retired.
    auto [it, inserted] = group_name_to_team_pool_.emplace(
        group_name, TeamPool(MAX_N_TEAMS, NVSHMEM_TEAM_INVALID));
    if (inserted) {
      auto& membership = membership_state(global_ranks);
      auto reusable = take_reusable_team_pool(group_name, membership);
      uint64_t pool_id;
      if (reusable.has_value()) {
        pool_id = reusable->first;
        it->second = std::move(reusable->second);
      } else {
        pool_id = membership.next_pool_id++;
      }
      group_name_to_global_ranks_.emplace(group_name, global_ranks);
      group_name_to_pool_id_.emplace(group_name, pool_id);
    } else {
      TORCH_INTERNAL_ASSERT(
          group_name_to_global_ranks_.at(group_name) == global_ranks);
    }
    auto& team_pool = it->second;
    bool pool_updated = inserted;

    // Create new teams if what's requested is more than what we have
    int stride = 0; // stride in globe, uninitialized
    for (int i = 0; i < need_n; ++i) {
      if (team_pool[i] != NVSHMEM_TEAM_INVALID) {
        continue;
      }
      // Some checks before we create new teams
      if (stride == 0) { // Check only once
        TORCH_CHECK(global_ranks.size() > 1);
        stride = global_ranks[1] - global_ranks[0];
        for (size_t r = 1; r < global_ranks.size(); ++r) {
          TORCH_CHECK(global_ranks[r] - global_ranks[r - 1] == stride);
        }
      }
      nvshmem_team_t team = NVSHMEM_TEAM_INVALID;
      nvshmem_team_split_strided(
          NVSHMEM_TEAM_WORLD,
          global_ranks[0],
          stride,
          global_ranks.size(),
          nullptr,
          0,
          &team);
      TORCH_CHECK(team != NVSHMEM_TEAM_INVALID, "Failed to create a new team");
      team_pool[i] = team;
      pool_updated = true;
    }
    return std::make_pair(std::cref(team_pool), pool_updated);
  }

 private:
  // Device where the team manager is created
  struct State;
  static State& state();

  const c10::Device device_;
  // A map from group name to team pool for that group.
  std::unordered_map<std::string, TeamPool> group_name_to_team_pool_;
  // Membership and pool identity of each live group.
  std::unordered_map<std::string, std::vector<int>>
      group_name_to_global_ranks_;
  std::unordered_map<std::string, uint64_t> group_name_to_pool_id_;
  // Per-membership retired pools and their rank-comparable identities.
  std::vector<MembershipState> membership_states_;
  // A map from group name to team pool array in device memory.
  std::unordered_map<std::string, nvshmem_team_t*> team_pool_devptrs_;
};

struct TeamManager::State {
  std::mutex mutex;
  std::unique_ptr<TeamManager> manager;
};

inline TeamManager::State& TeamManager::state() {
  static State state;
  return state;
}

inline TeamManager& TeamManager::get(const c10::Device device) {
  auto& state = TeamManager::state();
  std::lock_guard lock(state.mutex);
  if (state.manager == nullptr) {
    state.manager = std::make_unique<TeamManager>(device);
  }
  TORCH_CHECK(
      state.manager->device_ == device,
      "Detected use of TeamManager on multiple devices. This is not supported.");
  return *state.manager;
}

inline void TeamManager::release_group_if_initialized(
    const std::string& group_name) {
  auto& state = TeamManager::state();
  std::lock_guard lock(state.mutex);
  if (state.manager != nullptr) {
    state.manager->release_group(group_name);
  }
}

} // namespace c10d::nvshmem_extension
