#pragma once

#include <torch/csrc/utils/tensor_flatten.h>

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace c10d {
inline void distBroadcastCoalesced(
    std::vector<at::Tensor>& tensors,
    int64_t bufferSize,
    ProcessGroup& processGroup) {
  auto tensorGroups = torch::utils::take_tensors(tensors, bufferSize);
  // We store single-element vectors in `flatTensors` because
  // `ProcessGroup::broadcast` takes a reference to a vector, which must be
  // alive until the `wait()` call on the returned `Work` completes.
  std::vector<std::vector<at::Tensor>> flatTensors;
  std::vector<std::shared_ptr<ProcessGroup::Work>> work;
  flatTensors.reserve(tensorGroups.size());
  work.reserve(tensorGroups.size());
  for (const auto& group : tensorGroups) {
    flatTensors.push_back({torch::utils::flatten_dense_tensors(group.tensors)});
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = 0;
    broadcastOptions.rootTensor = 0;
    work.push_back(
        processGroup.broadcast(flatTensors.back(), broadcastOptions));
  }
  for (size_t group = 0; group < tensorGroups.size(); ++group) {
    auto& tensors = tensorGroups[group].tensors;
    work[group]->wait();
    const auto synced =
        torch::utils::unflatten_dense_tensors(flatTensors[group][0], tensors);
    AT_ASSERT(synced.size() == tensors.size());
    for (size_t i = 0; i < synced.size(); ++i) {
      tensors[i].copy_(synced[i]);
    }
  }
}
} // namespace c10d
