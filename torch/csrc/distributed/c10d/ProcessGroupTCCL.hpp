#pragma once

#ifdef USE_C10D_TCCL

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

// Forward declaration - definition in TCCLUtils.hpp (transport layer).
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

    uint64_t getSequencenumber() const override {
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

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
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

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> all_to_all_single(
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

  // batch_isend_irecv: the sends and recvs issued between start and end run
  // as one exchange, so a pair or a ring shift does not wait on itself.
  bool supportsCoalescing() const override {
    return true;
  }
  void startCoalescing() override;
  c10::intrusive_ptr<Work> endCoalescing() override;

 protected:
  // A batch_isend_irecv batch's Work: it completes with the last of the
  // batch's ops, or fails with the first that fails (then `work` is null).
  struct P2pBatch {
    std::size_t left;
    c10::intrusive_ptr<TCCLWork> work;
  };
  struct P2POp {
    bool isSend;
    int peer;
    at::Tensor tensor;
    c10::intrusive_ptr<TCCLWork> work;
    std::shared_ptr<P2pBatch> batch;
  };
  // One queued item: a collective (fn) or a point-to-point op (p2p). The
  // worker runs a p2p op together with the p2p ops queued right behind it, so
  // an irecv issued after an isend is posted while the send waits. A
  // collective is a fence: it starts only once every p2p op queued before it
  // has completed, so the peers must issue it in the same order relative to
  // their sends and recvs.
  struct Task {
    std::function<void()> fn;
    std::optional<P2POp> p2p;
    std::optional<uint32_t> mpsEventId;
  };
  bool coalescing_{false};
  std::vector<P2POp> coalescedOps_;

  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
  // Resolved in ctor from options_ + TCCL_TOPOLOGY
  Topology topology_{Topology::Mesh};
  uint64_t seq_{0};
  uint64_t barrierSeq_{0};

  // One TCCLConnection per (peer_rank, wire) - one UC queue pair each. Index
  // = peer_rank * num_wires + wire. Slots at peer_rank == rank_ stay null.
  // Total size = size_ * options_->num_wires.
  std::vector<std::unique_ptr<TCCLConnection>> connections_;

  // One send buffer + one recv buffer per peer. Both vectors are sized to
  // `size_`; self-slot stays empty (TCCLSharedBuffer with data_=nullptr).
  std::vector<TCCLSharedBuffer> sendBuffers_;
  std::vector<TCCLSharedBuffer> recvBuffers_;

  // Depth-indexed staging pools for the pipelined bidirectional ring, flat
  // [peer * kPipelineDepth + slot]. Sized size_ * kPipelineDepth; non-neighbor
  // and self slots stay empty under ring topology.
  std::vector<TCCLSharedBuffer> pipeSendBuffers_;
  std::vector<TCCLSharedBuffer> pipeRecvBuffers_;

  // RDMA collective engine. Constructed after buffers are. Holds references
  // into connections_/sendBuffers_/recvBuffers_;
  std::unique_ptr<TCCLEngine> engine_;

  // Worker thread model. Spawned at the end of the constructor
  // (after the RTS barrier) and joined in the destructor.
  std::thread workerThread_;
  std::deque<Task> workQueue_;
  std::mutex workMutex_;
  std::condition_variable workCV_;
  std::atomic<bool> stop_{false};

  // Set by the worker when an operation fails. The failure can leave work
  // requests posted, so the queue pairs are moved to the error state and the
  // group refuses new work. brokenReason_ is written once, before broken_.
  std::atomic<bool> broken_{false};
  std::string brokenReason_;

  // Shared async scaffold. Records an MPS event so the worker waits for the GPU
  // to flush writes to the inputs, enqueues `fn`, and returns a TCCLWork whose
  // Future completes with `outputs` when `fn` returns. `fn` does the
  // mpsSharedCpuView + engine_ call; `outputs` are what the Future carries (DDP
  // getFuture() chaining).
  c10::intrusive_ptr<Work> enqueueCollective(
      OpType opType,
      std::vector<at::Tensor> outputs,
      std::function<void()> fn);
  // Queues a send or recv; see Task.
  c10::intrusive_ptr<Work> enqueueP2p(P2POp op);
  // The single and coalesced forms share these; opType is what the Work
  // reports (flight recorder, retrieveOpType).
  c10::intrusive_ptr<Work> allGatherIntoTensors(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      OpType opType);
  c10::intrusive_ptr<Work> reduceScatterTensors(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts,
      OpType opType);
  // Runs `first` and the p2p ops queued behind it until all have completed.
  void runP2p(Task first);
  static void finishP2p(P2POp& op, const std::exception_ptr& ep);
  // After a failure: records the reason, flushes every queue pair and marks the
  // group broken, so it refuses new work.
  void markBroken(const std::exception_ptr& ep);
  std::string refusedMessage() const;

  // send/recv argument checks, shared by both directions.
  void checkP2p(const std::vector<at::Tensor>& tensors, int peer, bool isSend)
      const;
  // Drops an open batch, failing its recorded ops with `ep`. The coalescing
  // manager has no finally, so after a rejected op an open batch would record
  // every later send/recv and never run it.
  void closeCoalescing(const std::exception_ptr& ep);

  void runLoop();
};

} // namespace c10d

#endif // USE_C10D_TCCL
