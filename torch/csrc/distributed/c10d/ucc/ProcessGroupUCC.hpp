#pragma once

#ifdef USE_C10D_UCC

#include <c10d/Store.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/UCXUtils.hpp>

namespace c10d {

constexpr const char* UCC_BACKEND_NAME = "_internal_ucc";

// ProcessGroupUCC implements UCC & UCX bindings for c10d. UCC is used for
// collective operations, and UCX is used for P2P operations.
//
// The UCC & UCX binding is not published to the user directly, but it provided
// a process group called `_internal_ucc`. The `_internal_ucc` is only for
// testing purposes, and for power users who really knows what they are doing.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// Links:
// ucx: https://github.com/openucx/ucx
// ucc: https://github.com/openucx/ucc
// Original torch_ucc: https://github.com/facebookresearch/torch_ucc
//
// *****************************************************************************
// This ProcessGroup is still under development, and there are some know issues:
// - Only send and recv are supported.
// - It is fake async: UCP worker are progressed only when checking status.
class TORCH_API ProcessGroupUCC final : public ProcessGroup {
public:
  class WorkUCP : public ProcessGroup::Work {
    bool *finished;
  public:
    WorkUCP(ucs_status_ptr_t ptr) : finished(reinterpret_cast<bool *>(ptr)) {}
    ~WorkUCP() { ucp_request_free(finished); }
    bool isCompleted() override {
      // TODO: progress worker in a side thread for true async
      ucp_worker_progress(UCPContext::get()->worker);
      return *finished;
    };
    bool isSuccess() const override {
      // TODO: progress worker in a side thread for true async
      ucp_worker_progress(UCPContext::get()->worker);
      return *finished;
    };
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {
      while(!isCompleted());
      return true;
    };
  };

  class ImmediatelyCompletedWork : public ProcessGroup::Work {
  public:
    bool isCompleted() override { return true; };
    bool isSuccess() const override { return true; };
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {
      return true;
    };
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size);

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

private:
  c10::intrusive_ptr<Store> store;
  void lazyInitUCP();
  std::vector<std::shared_ptr<UCPEndpoint>> ucp_endpoints = {};
};

} // namespace c10d

#endif
