#include <torch/csrc/distributed/c10d/ddp.h>

#include <torch/csrc/cuda/comm.h>
#include <torch/csrc/utils/tensor_flatten.h>

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <vector>




#include <iostream>




namespace c10d {
namespace {
/// For every replica except the root, copy the data from `broadcastTensors`
/// to `replicaData`.
void copyBroadcastTensorsToReplicas(
    const std::vector<std::vector<at::Tensor>>& broadcastTensors,
    std::vector<std::vector<at::Tensor>>& replicaData) {
  AT_ASSERT(replicaData.size() == broadcastTensors.size());
  // replica = 1 means we skip the root (replica 0).
  for (size_t replica = 1; replica < replicaData.size(); ++replica) {
    AT_ASSERT(replicaData[replica].size() == broadcastTensors[replica].size());
    for (size_t tensor = 0; tensor < replicaData[replica].size(); ++tensor) {
      replicaData[replica][tensor].set_(broadcastTensors[replica][tensor]);
    }
  }
}
} // namespace

void distBroadcastCoalesced(
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
    // Flatten each group of tensors (whose size equals `bufferSize`) into a
    // single tensor.
    flatTensors.push_back({torch::utils::flatten_dense_tensors(group.tensors)});
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = 0;
    broadcastOptions.rootTensor = 0;
    // Enqueue a work item and collect the `Work` (essentially a "future") so we
    // can `wait()` for its completion after we have collected all `Work` items.
    work.push_back(
        processGroup.broadcast(flatTensors.back(), broadcastOptions));
  }
  // Now loop through each group, wait for the broadcast to complete, and
  // un-flatten the broadcast tensor back into device-local individual tensors.
  for (size_t group = 0; group < tensorGroups.size(); ++group) {
    auto& tensors = tensorGroups[group].tensors;
    work[group]->wait();
    const auto synced =
        torch::utils::unflatten_dense_tensors(flatTensors[group][0], tensors);
    AT_ASSERT(synced.size() == tensors.size());
    for (size_t i = 0; i < synced.size(); ++i) {
      // Copy into the per-process tensors.
      tensors[i].copy_(synced[i], /*non_blocking=*/true);
    }
  }
}

void syncParams(
    ProcessGroup& processGroup,
    std::vector<std::vector<at::Tensor>>& parameterData,
    std::vector<std::vector<at::Tensor>>& bufferData,
    const std::vector<int64_t>& devices,
    int64_t broadcastBucketSize,
    bool broadcastBuffers) {
  std::cout << 1 << std::endl;
  AT_ASSERT(!parameterData.empty());
  AT_ASSERT(!bufferData.empty());
  AT_ASSERT(!devices.empty());
  std::cout << 2 << std::endl;

  // Do an intra-node sync if we have more than one device.
  if (devices.size() > 1) {
    std::cout << 3 << std::endl;
    // Broadcast the parameters, get back a vector<vector<Tensor>>, one
    // vector<Tensor> per device. Each such vector then needs to be copied into
    // the `parameterData` of every step.
    auto result = torch::cuda::broadcast_coalesced(
        parameterData[0], devices, broadcastBucketSize);
    std::cout << 4 << std::endl;
    copyBroadcastTensorsToReplicas(result, parameterData);
    std::cout << 5 << std::endl;
  }
  std::cout << 6 << std::endl;

  if (broadcastBuffers && !bufferData[0].empty()) {
    std::cout << 7 << std::endl;
    // Do an inter-node sync first.
    distBroadcastCoalesced(bufferData[0], broadcastBucketSize, processGroup);
    std::cout << 8 << std::endl;
    // Then an intra-node sync if we have more than one device.
    if (devices.size() > 1) {
      std::cout << 9 << std::endl;
      auto result = torch::cuda::broadcast_coalesced(
          bufferData[0], devices, broadcastBucketSize);
      std::cout << 10 << std::endl;
      copyBroadcastTensorsToReplicas(result, bufferData);
      std::cout << 11 << std::endl;
    }
  }
  std::cout << 12 << std::endl;
}

} // namespace c10d
