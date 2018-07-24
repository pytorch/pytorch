#pragma once

#include <torch/csrc/utils/tensor_flatten.h>

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

namespace c10d {
inline void distBroadcastCoalesced(
    std::vector<at::Tensor>& tensors,
    int64_t bufferSize,
    ProcessGroup& processGroup) {
  for (auto& group : torch::utils::take_tensors(tensors, bufferSize)) {
    std::vector<at::Tensor> flatTensor = {
        torch::utils::flatten_dense_tensors(group.tensors)};
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = 0;
    broadcastOptions.rootTensor = 0;
    processGroup.broadcast(flatTensor, broadcastOptions)->wait();
    auto synced =
        torch::utils::unflatten_dense_tensors(flatTensor[0], group.tensors);
    AT_ASSERT(synced.size() == group.tensors.size());
    for (size_t i = 0; i < synced.size(); ++i) {
      group.tensors[i].copy_(synced[i]);
    }
  }
}
} // namespace c10d
