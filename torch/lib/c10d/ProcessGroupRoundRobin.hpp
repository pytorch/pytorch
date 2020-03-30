#pragma once

#include <vector>

#include <c10d/ProcessGroup.hpp>

namespace c10d {

// ProcessGroupRoundRobin implements simple load balancing.
//
// It is constructed with multiple processes groups. Each call is dispatched to
// one of the specified process groups in a round robin fashion. Each process
// group instance must have the same rank and size.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group. This is the only way that we
// can guarantee to match up the same calls among all processes.
//
class ProcessGroupRoundRobin final : public ProcessGroup {
 public:
  explicit ProcessGroupRoundRobin(
      int rank,
      int size,
      std::vector<std::shared_ptr<ProcessGroup>> processGroups);

  ~ProcessGroupRoundRobin() override;

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

 private:
  std::vector<std::shared_ptr<ProcessGroup>> processGroups_;
  std::vector<std::shared_ptr<ProcessGroup>>::const_iterator iterator_;

  // Returns the next ProcessGroup to use.
  const std::shared_ptr<ProcessGroup>& next();
};

} // namespace c10d
