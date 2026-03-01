#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <string>
#include <unordered_map>
#include <vector>

// Starting from NVSHMEM 3.3.9, nvshmem_host.h exists so that we can cleanly
// include only the nvshmem host library headers:
// #include <nvshmem_host.h>
// It translates into the following two lines:
#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
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
  static TeamManager& get(const c10::Device device) {
    static TeamManager manager(device);
    TORCH_CHECK(
        manager.device_ == device,
        "Detected use of TeamManager on multiple devices. This is not supported.");
    return manager;
  }

  // Get a team for a group.
  nvshmem_team_t get_team(
      const std::string& group_name,
      const std::vector<int>& global_ranks) {
    auto [team_pool, pool_updated] =
        group_to_team_pool(group_name, global_ranks, 1);
    // Return the fist available team
    return team_pool[0];
  }

  // Get n teams for a group.
  // The first element of the returned pair is the team pool on host side.
  // The second element of the returned pair is the team pool on device side.
  // This API must be call with a device guard.
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

    // Insert a new team pool if not exists
    auto [it, inserted] = group_name_to_team_pool_.emplace(
        group_name, TeamPool(MAX_N_TEAMS, NVSHMEM_TEAM_INVALID));
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
  const c10::Device device_;
  // A map from group name to team pool for that group.
  std::unordered_map<std::string, TeamPool> group_name_to_team_pool_;
  // A map from group name to team pool array in device memory.
  std::unordered_map<std::string, nvshmem_team_t*> team_pool_devptrs_;
};

} // namespace c10d::nvshmem_extension
