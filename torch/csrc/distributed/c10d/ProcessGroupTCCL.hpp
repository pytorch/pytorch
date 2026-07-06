#pragma once

#ifdef USE_C10D_TCCL

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

constexpr const char* TCCL_BACKEND_NAME = "tccl";

// Forward declaration — definition in TCCLUtils.hpp (transport layer).
class TCCLConnection;
class TCCLSharedBuffer;
class TCCLEngine;

class TORCH_API ProcessGroupTCCL : public Backend {
 public:
  // Thin Work subclass. Completion state (mutex_, cv_, completed_,
  // exception_) is owned by the Work base class. Use finishWork(eptr) to
  // signal completion.
  class TORCH_API TCCLWork : public Work {
   public:
    TCCLWork(OpType opType, uint64_t seq, std::vector<at::Tensor> outputs = {});

    uint64_t seq() const noexcept {
      return seq_;
    }

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    // Public completion entry point (the worker lambda calls this).
    void finishWork(std::exception_ptr exception = nullptr) {
      if (exception) {
        future_->setError(exception);
      } else {
        future_->markCompleted(c10::IValue(outputs_));
      }
      finish(std::move(exception));
    }

   private:
    const uint64_t seq_;
    std::vector<at::Tensor> outputs_;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
  };

  enum class Topology { Mesh, Ring };

  struct TORCH_API Options : Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }

    // Name of the RDMA device as exposed by librdma (e.g. "rdma_en2").
    std::string device_name;

    // (multi-wire is not yet supported).
    int num_wires{1};
    Topology topology{Topology::Mesh};
  };

  explicit ProcessGroupTCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupTCCL() override;

  const std::string getBackendName() const override {
    return std::string(TCCL_BACKEND_NAME);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
  }

  // Collective overrides
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
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

  // Coalesced all-gather-into-tensor. This is the virtual that DTensor /
  // Tensor-Parallel reach via _functional_collectives (all_gather_into_tensor
  // -> Functional.cpp -> group->allgather_into_tensor_coalesced), NOT
  // _allgather_base.
  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
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

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  // Coalesced reduce-scatter-tensor. The virtual DTensor / sequence-parallel
  // reach via _functional_collectives (reduce_scatter_tensor -> Functional.cpp
  // -> group->reduce_scatter_tensor_coalesced).
  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

 protected:
  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
  Topology topology_{Topology::Mesh};  // resolved in ctor (options_ + TCCL_TOPOLOGY)
  uint64_t seq_{0};
  uint64_t barrierSeq_{0};

  // One TCCLConnection per (peer_rank, wire) — one UC queue pair each. Index
  // = peer_rank * num_wires + wire. Slots at peer_rank == rank_ stay null.
  // Total size = size_ * options_->num_wires.
  std::vector<std::unique_ptr<TCCLConnection>> connections_;

  // One send buffer + one recv buffer per peer. Both vectors are sized to
  // `size_`; self-slot stays empty (TCCLSharedBuffer with data_=nullptr).
  std::vector<TCCLSharedBuffer> sendBuffers_;
  std::vector<TCCLSharedBuffer> recvBuffers_;

  // RDMA collective engine. Constructed after buffers are. Holds references
  // into connections_/sendBuffers_/recvBuffers_;
  std::unique_ptr<TCCLEngine> engine_;

  // Worker thread model. Spawned at the end of the constructor
  // (after the RTS barrier) and joined in the destructor.
  std::thread workerThread_;
  std::deque<std::function<void()>> workQueue_;
  std::mutex workMutex_;
  std::condition_variable workCV_;
  std::atomic<bool> stop_{false};

  // Shared async scaffold for collectives. Records an MPS event on the
  // calling (main) thread so the worker waits for the GPU to flush writes to
  // the input tensors, enqueues `fn` onto the worker, and returns a TCCLWork
  // whose Future completes with `outputs` when `fn` returns. `fn` captures the
  // tensors it needs and does the mpsSharedCpuView + engine_-> call; `outputs`
  // are the tensors the Future should carry (for DDP's getFuture() chaining).
  c10::intrusive_ptr<Work> enqueueCollective(
      OpType opType,
      std::vector<at::Tensor> outputs,
      std::function<void()> fn);

  void runLoop();
};

} // namespace c10d

#endif // USE_C10D_TCCL
