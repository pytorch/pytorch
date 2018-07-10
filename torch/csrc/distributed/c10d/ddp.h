#pragma once

#include <torch/csrc/utils/tensor_flatten.h>

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

namespace c10d {
inline void dist_broadcast_coalesced(
    std::vector<at::Tensor>& tensors,
    int64_t buffer_size,
    ProcessGroup& process_group) {
  for (auto& group : torch::utils::take_tensors(tensors, buffer_size)) {
    std::vector<at::Tensor> flat_tensor = {
        torch::utils::flatten_dense_tensors(group.tensors)};
    BroadcastOptions broadcast_options;
    broadcast_options.rootRank = 0;
    broadcast_options.rootTensor = 0;
    process_group.broadcast(flat_tensor, broadcast_options)->wait();
    auto synced =
        torch::utils::unflatten_dense_tensors(flat_tensor[0], group.tensors);
    AT_ASSERT(synced.size() == group.tensors.size());
    for (size_t i = 0; i < synced.size(); ++i) {
      group.tensors[i].copy_(synced[i]);
    }
  }
}
} // namespace c10d
