#pragma once

#ifdef USE_C10D_UCC

#include <c10d/Store.hpp>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

constexpr const char* UCC_BACKEND_NAME = "_internal_ucc";

// ProcessGroupUCC implements UCC bindings for c10d.
//
// The UCC binding is not published to the user directly, but it provided
// a process group called `_internal_ucc`. The `_internal_ucc` is only for
// testing purposes, and for users who really knows what they are doing.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// ************* TODO: edit below *************
//
// All NCCL functions provided by this class are asynchronous functions. More
// specifically, each NCCL call is scheduled on a separate CUDA stream that is
// different from the current CUDA stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the CUDA stream their
// code works on needs to wait for the NCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkNCCL::wait() or WorkNCCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkNCCL::finishedGPUExecution() is a helper function only
// provided by ProcessGroupNCCL to check if the NCCL operation of WorkNCCL has
// finished execution on the GPU (not just scheduled).
//
// Example on using the NCCL process group
//
//   ProcessGroupNCCL pg(store, rank, size);
//   std::shared_ptr<WorkNCCL> work = pg.allreduce(tensors);
//
//   // At this point, NCCL kernel has already by queued successfully
//   // Now, let current stream wait for the NCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_API ProcessGroupUCC final : public ProcessGroup {
public:
  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size);

  ~ProcessGroupUCC() override;

  const std::string getBackendName() const override {
      return std::string(UCC_BACKEND_NAME);
  }

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;
};

} // namespace c10d

#endif
