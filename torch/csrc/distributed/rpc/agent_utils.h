#pragma once

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

// All RPC peers should call into this function at the same time. Each peer
// provides its own id and name, and this function uses the given Store to
// gather global name-to-id mapping on all peers.
TORCH_API std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName,
    const int worldSize);

// Ranks in dynamic RPC groups will initially call into this to establish the
// name-to-id mapping for the current peers in the group. The current rank will
// put its own worker info in the store and discover all the ranks that came
// before it. NOTE: This needs to be called with the Dynamic RPC group
// membership management token held.
TORCH_API std::unordered_map<std::string, worker_id_t> collectCurrentNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName);

// Remove name frmo Store, used in dynamic RPC groups.
// NOTE: This needs to be called with the Dynamic RPC group
// membership management token held.
TORCH_API void removeCurrentName(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName);

// This performs a synchronization of all call counts by using store.
// All RPC peers wait for others to join to exit at the same time.
TORCH_API int syncCallCount(
    ::c10d::PrefixStore store,
    const int worldSize,
    int activeCalls = 0);

} // namespace rpc
} // namespace distributed
} // namespace torch
