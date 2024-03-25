#pragma once

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace c10d {

class TORCH_API ProcessGroupWrapper : public Backend {
 public:
  explicit ProcessGroupWrapper(
      c10::intrusive_ptr<Backend> backend,
      c10::intrusive_ptr<Backend> glooBackend);

  const std::string getBackendName() const override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& data,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  // This function is deprecated and will be moved out of ProcessGroup to comms:
  // * do not add dependencies on this function,
  // * do not implement it in your ProcessGroup, implement _allgather_base
  //   instead.
  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  void monitoredBarrier(const BarrierOptions& opts, bool waitAllRanks = false)
      override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store. Only implemented
  // for GLOO and NCCL backends currently.
  // dont implement this
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override; // just call underlying

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts) override;

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Backend> getWrappedPg() const;

 private:
  // Underlying process group that actual application collectives will be
  // dispatched to
  c10::intrusive_ptr<Backend> backend_;
  // Gloo process group responsible for internal coordination such as monitored
  // barrier, sequence number checking, collective fingerprint collecting.
  c10::intrusive_ptr<Backend> glooBackend_;
  // Conducts several checks to ensure that the underlying collective is well
  // formed with the goal of notifying the user about incorrect collective use
  // in the application.
  void runCollectiveChecks(
      OpType op_type,
      const std::vector<at::Tensor>& tensors);
};
} // namespace c10d

#endif // USE_C10D_GLOO
