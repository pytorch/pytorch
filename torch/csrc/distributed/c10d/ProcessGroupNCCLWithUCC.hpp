#pragma once

#include <ATen/DynamicLibrary.h>
#include <c10d/Store.hpp>
#include <c10d/ProcessGroup.hpp>

#if defined(USE_C10D_NCCL) || defined(USE_C10D_UCC)

constexpr const char *TORCH_NCCL_ENABLED = "TORCH_NCCL_ENABLED";
constexpr const char *TORCH_UCC_ENABLED = "TORCH_UCC_ENABLED";

namespace c10d {

constexpr const char *BACKEND_NAME = "nccl";

// ProcessGroupNCCLWithUCC is the unified NCCL & UCX & UCC bindings for c10d.
//
// This process group itself does not do any operation itself. But instead, it
// act as a dispatcher:
// When the user creates a backend "nccl" from Python, the user actually creates
// an object of ProcessGroupNCCLWithUCC. ProcessGroupNCCLWithUCC is a container
// process group that has both a ProcessGroupNCCL and a ProcessGroupUCC, where
// the ProcessGroupUCC provides the binding for both UCX and UCC. Operations
// in ProcessGroupNCCLWithUCC will dispatch to either ProcessGroupNCCL or
// ProcessGroupUCC based on the operation. Most GPU operations are dispatched
// to ProcessGroupNCCL. Non-GPU operations are dispatched to ProcessGroupUCC.
//
// Some GPU operations are dispatched to ProcessGroupUCC because NCCL does not
// support such operations. Examples are:
//   - send/recv/recvAnysource with non-zero tag
//
// Users can control whether NCCL or UCX & UCC bindings are enabled by environmental
// variables TORCH_NCCL_ENABLED and TORCH_UCC_ENABLED. When NCCL is disabled
// by the user, ProcessGroupUCC will be used for GPU operations.
//
// Both ProcessGroupNCCL and ProcessGroupUCC supports profilers. Operations done
// by ProcessGroupNCCL will be profiled as something like `nccl:recv` and oparations
// of ProcessGroupUCC will appear as `ucc:recv`. This allows users to tell which
// backend is actually used by profiling.
class TORCH_API ProcessGroupNCCLWithUCC : public ProcessGroup {
public:

  // Options is only used by NCCL.
  struct Options : ProcessGroup::Options {
    // NOTE: timeout in ProcessGroupNCCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(
        bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule NCCL operations on high priority CUDA streams
    bool is_high_priority_stream;
  };

  ProcessGroupNCCLWithUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  virtual ~ProcessGroupNCCLWithUCC();

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override {
      return std::string(BACKEND_NAME);
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
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

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

  void setSequenceNumberForGroup() override;

  uint64_t getSequenceNumberForGroup() override;

  static void groupStart();

  static void groupEnd();

  static bool is_ucc_available();

private:
  c10::intrusive_ptr<Options> options_;
  c10::intrusive_ptr<ProcessGroup> pg_nccl;
  c10::intrusive_ptr<ProcessGroup> pg_ucc;
};

} // namespace c10d

#endif // USE_C10D_NCCL || USE_C10D_UCC
