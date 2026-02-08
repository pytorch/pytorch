#pragma once

#include <chrono>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <include/Macros.h>

namespace c10d {

//
// ProcessGroupTest implements dummy bindings for c10d.
//

constexpr const char* OCCL_BACKEND_NAME = "occl";

class OPENREG_EXPORT ProcessGroupOCCL : public Backend {
 public:
  class DummyWork : public Work {
   public:
    DummyWork();

    virtual ~DummyWork();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout) override;
    void synchronize() override;
    void abort() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupOCCL;

   private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
  };

  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }
  };

  explicit ProcessGroupOCCL(int rank = -1, int size = -1);
  virtual ~ProcessGroupOCCL();
  const std::string getBackendName() const override {
    return std::string(OCCL_BACKEND_NAME);
  }

  bool supportsSplitting() const override {
    return false;
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  c10::intrusive_ptr<Work> broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts =
      AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_lists,
    std::vector<at::Tensor>& input_list,
    const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputCounts,
    std::vector<int64_t>& inputCounts,
    const AllToAllOptions& opts = AllToAllOptions()) override;

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

 protected:
  const c10::intrusive_ptr<Options> options_;
};
OPENREG_EXPORT c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout);

} // namespace c10d
