#pragma once

#include <c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

// All RPC peers should call into this function at the same time. Each peer
// provides its own id and name, and this function uses the given Store to
// gather global name-to-id mapping on all peers.
std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName,
    const int worldSize);

// This performs a synchronization of all call counts by using store.
// All RPC peers wait for others to join to exit at the same time.
int syncCallCount(
    ::c10d::PrefixStore store,
    const int worldSize,
    int activeCalls = 0);

} // namespace rpc
} // namespace distributed
} // namespace torch
