#pragma once

#include <c10/cuda/CUDAException.h>
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
constexpr int MAX_N_TEAMS = 32;

// A pool of teams for each group. These are duplicate teams.
using nvshmemTeamPool_t = std::vector<nvshmem_team_t>;

// Manage all the team business. Singleton.
class TeamManager {
 public:
  // Get single, global manager.
  static TeamManager& get() {
    static TeamManager manager;
    return manager;
  }

  // Get a team for a group.
  nvshmem_team_t get_team(
      const std::string& group_name,
      const std::vector<int>& global_ranks) {
    bool pool_updated = false; // unused
    auto& team_pool =
        group_to_team_pool(group_name, global_ranks, 1, pool_updated);
    // Return the fist available team
    return team_pool[0];
  }

  // Get n teams for a group.
  // The first element of the returned pair is the team pool on host side.
  // The second element of the returned pair is the team pool on device side.
  std::pair<const nvshmemTeamPool_t&, nvshmem_team_t*> get_n_teams(
      const std::string& group_name,
      const std::vector<int>& global_ranks,
      const int need_n) {
    // Get the team pool with the requested number of teams
    bool pool_updated = false;
    auto& team_pool =
        group_to_team_pool(group_name, global_ranks, need_n, pool_updated);
    // Check if the pool already exists in device memory
    nvshmem_team_t* team_pool_dev = nullptr;
    constexpr auto pool_bytes = sizeof(nvshmem_team_t) * MAX_N_TEAMS;
    auto it = team_pool_devptrs_.find(group_name);
    if (it == team_pool_devptrs_.end()) {
      // If not, allocate a new pool in device memory
      C10_CUDA_CHECK(cudaMalloc((void**)&team_pool_dev, pool_bytes));
      team_pool_devptrs_[group_name] = team_pool_dev;
    } else {
      team_pool_dev = it->second;
    }
    // Update the pool in device memory if host side pool is updated
    if (pool_updated) {
      TORCH_INTERNAL_ASSERT(team_pool.size() == MAX_N_TEAMS);
      C10_CUDA_CHECK(cudaMemcpy(
          team_pool_dev, team_pool.data(), pool_bytes, cudaMemcpyHostToDevice));
    }
    return std::make_pair(team_pool, team_pool_dev);
  }

 private:
  // Get the team pool for a group. If the pool doesn't exist, create it. If the
  // pool exists but is not large enough, create more teams.
  const nvshmemTeamPool_t& group_to_team_pool(
      const std::string& group_name,
      const std::vector<int>& global_ranks,
      const int need_n,
      bool& pool_updated) {
    TORCH_CHECK(need_n < MAX_N_TEAMS, "Too many teams requested");

    // Insert a new team pool if not exists
    auto pair = group_name_to_team_pool_.emplace(
        group_name, nvshmemTeamPool_t(MAX_N_TEAMS, NVSHMEM_TEAM_INVALID));
    // pair.first is an iterator to the inserted element or the existing element
    auto& team_pool = pair.first->second;
    // pair.second is true if the element is inserted, false if the element
    // already exists
    pool_updated = pair.second;

    // Some checks before we create new teams
    TORCH_CHECK(global_ranks.size() > 1);
    int stride = global_ranks[1] - global_ranks[0];
    for (size_t r = 1; r < global_ranks.size(); ++r) {
      TORCH_CHECK(global_ranks[r] - global_ranks[r - 1] == stride);
    }

    // Create new teams if what's requested is more than what we have
    for (int i = 0; i < need_n; ++i) {
      if (team_pool[i] != NVSHMEM_TEAM_INVALID) {
        continue;
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
    return team_pool;
  }

 private:
  // A map from group name to team pool for that group.
  std::unordered_map<std::string, nvshmemTeamPool_t> group_name_to_team_pool_;
  // A map from group name to team pool array in device memory.
  std::unordered_map<std::string, nvshmem_team_t*> team_pool_devptrs_;
};

} // namespace c10d::nvshmem_extension