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

} // namespace rpc
} // namespace distributed
} // namespace torch
