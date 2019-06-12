#include <torch/csrc/distributed/c10d/ddp.h>

#include <torch/csrc/cuda/comm.h>
#include <torch/csrc/utils/tensor_flatten.h>

#include <torch/csrc/cuda/nccl.h>

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGuard.h>

#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

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

std::vector<std::vector<at::Tensor>> bucketTensors(
    std::vector<at::Tensor>& tensors,
    int64_t bucketSize,
    bool fineGrained) {
  std::vector<std::vector<at::Tensor>> bucketedTensors;
  auto tensorGroups =
      torch::utils::take_tensors(tensors, bucketSize, fineGrained);

  bucketedTensors.reserve(tensorGroups.size());
  for (auto& tensorGroup : tensorGroups) {
    bucketedTensors.push_back(std::move(tensorGroup.tensors));
  }
  return bucketedTensors;
}

void distBroadcastCoalesced(
    ProcessGroup& processGroup,
    std::vector<at::Tensor>& tensors,
    int64_t bufferSize,
    bool fineGrained) {
  std::vector<std::vector<at::Tensor>> bucketedTensors =
      bucketTensors(tensors, bufferSize, fineGrained);
  // We store single-element vectors in `flatTensors` because
  // `ProcessGroup::broadcast` takes a reference to a vector, which must be
  // alive until the `wait()` call on the returned `Work` completes.
  std::vector<std::vector<at::Tensor>> flatTensors;
  std::vector<std::shared_ptr<ProcessGroup::Work>> work;
  flatTensors.reserve(bucketedTensors.size());
  work.reserve(bucketedTensors.size());
  for (const auto& tensorBucket : bucketedTensors) {
    // Flatten each bucket of tensors (whose size equals `bufferSize`) into a
    // single tensor.
    flatTensors.push_back({torch::utils::flatten_dense_tensors(tensorBucket)});
    BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = 0;
    broadcastOptions.rootTensor = 0;
    // Enqueue a work item and collect the `Work` (essentially a "future") so we
    // can `wait()` for its completion after we have collected all `Work` items.
    work.push_back(
        processGroup.broadcast(flatTensors.back(), broadcastOptions));
  }
  // Now loop through each bucket, wait for the broadcast to complete, and
  // un-flatten the broadcast tensor back into device-local individual tensors.
  for (size_t bucket = 0; bucket < bucketedTensors.size(); ++bucket) {
    auto& tensors = bucketedTensors[bucket];
    work[bucket]->wait();
    const auto synced =
        torch::utils::unflatten_dense_tensors(flatTensors[bucket][0], tensors);
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
  AT_ASSERT(!parameterData.empty());
  AT_ASSERT(!bufferData.empty());
  AT_ASSERT(!devices.empty());

  // Do an intra-node sync if we have more than one device.
  if (devices.size() > 1) {
    // Broadcast the parameters, get back a vector<vector<Tensor>>, one
    // vector<Tensor> per device. Each such vector then needs to be copied into
    // the `parameterData` of every step.
    auto result = torch::cuda::broadcast_coalesced(
        parameterData[0], devices, broadcastBucketSize);
    copyBroadcastTensorsToReplicas(result, parameterData);
  }

  if (broadcastBuffers && !bufferData[0].empty()) {
    // Do an inter-node sync first.
    distBroadcastCoalesced(processGroup, bufferData[0], broadcastBucketSize);
    // Then an intra-node sync if we have more than one device.
    if (devices.size() > 1) {
      auto result = torch::cuda::broadcast_coalesced(
          bufferData[0], devices, broadcastBucketSize);
      copyBroadcastTensorsToReplicas(result, bufferData);
    }
  }
}

std::tuple<std::shared_ptr<ProcessGroup::Work>, at::Tensor> queueReduction(
    ProcessGroup& processGroup,
    std::vector<std::vector<at::Tensor>>& gradsBatch,
    const std::vector<int64_t>& devices) {
  AT_ASSERT(!gradsBatch.empty());
  AT_ASSERT(!devices.empty());

  // Events to record the current state on the default stream of each GPUs
  std::vector<at::cuda::CUDAEvent> events;
  events.resize(devices.size());

  // Creating a separate CUDA stream to allow memory copy
  // and intra-node reduce to be operated on this worker stream to
  // improve performance
  std::vector<at::cuda::CUDAStream> workerStreams;
  for (size_t devIdx = 0; devIdx < devices.size(); ++devIdx) {
    at::cuda::CUDAGuard guard(devices[devIdx]);
    events[devIdx].record();
    workerStreams.push_back(
        at::cuda::getStreamFromPool(false, devices[devIdx]));
    // Let the worker stream to wait for the default stream
    events[devIdx].block(workerStreams.back());
  }

  // Stream guards, now the current stream is the worker stream
  at::cuda::CUDAMultiStreamGuard cudaGuard(workerStreams);

  std::vector<at::Tensor> gradsBatchCoalesced;
  for (size_t devIdx = 0; devIdx < devices.size(); ++devIdx) {
    at::cuda::CUDAGuard guard(devices[devIdx]);
    gradsBatchCoalesced.push_back(
        torch::utils::flatten_dense_tensors(gradsBatch[devIdx]));
  }

  if (devices.size() > 1) {
    torch::cuda::nccl::reduce(gradsBatchCoalesced, 0);
  }

  gradsBatchCoalesced[0] /= processGroup.getSize();

  std::vector<at::Tensor> allreduceInput = {gradsBatchCoalesced[0]};
  auto reductionWork = processGroup.allreduce(allreduceInput);

  return std::make_tuple(reductionWork, gradsBatchCoalesced[0]);
}

void syncReduction(
    std::shared_ptr<ProcessGroup::Work>& reductionWork,
    std::vector<at::Tensor>& gradsBatch,
    at::Tensor& gradsBatchCoalesced) {
  // Creating a separate CUDA stream to allow memory copy
  // and intra-node reduce to be operated on this worker stream to
  // improve performance
  at::cuda::CUDAStream workerStream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard cudaGuard(workerStream);

  // Let the worker stream wait on the reduction stream
  reductionWork->wait();
  // Now do the copy in worker stream
  std::vector<at::Tensor> gradsReduced =
      torch::utils::unflatten_dense_tensors(gradsBatchCoalesced, gradsBatch);

  AT_ASSERT(gradsReduced.size() == gradsBatch.size());

  for (size_t i = 0; i < gradsReduced.size(); ++i) {
    gradsBatch[i].copy_(gradsReduced[i]);
  }

  // Record the state in the worker stream
  at::cuda::CUDAEvent event;
  event.record(workerStream);

  // Now let the BW stream wait for the worker stream
  // (NB: original_stream is the current stream PRIOR to the guard.  Might
  // live on a completely different device than our current device here!)
  event.block(cudaGuard.original_stream());
}

} // namespace c10d
