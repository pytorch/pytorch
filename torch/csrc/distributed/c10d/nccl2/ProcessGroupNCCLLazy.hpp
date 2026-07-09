// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <mutex>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

// "nccl2-lazy" backend: wraps ProcessGroupNCCL so each point-to-point peer
// pair gets its own lazily-created 2-rank ncclComm (and therefore its own
// stream). Port of torchcomms' LazyBackend<TorchCommNCCL>.
//
// Motivation: in older versions of c10d's ProcessGroupNCCL, each P2P peer
// pair owned its own 2-rank ncclComm, bootstrapped on first send/recv between
// the two ranks without any global coordination; send/recv to different peers
// could overlap with each other and with collectives. This wrapper reproduces
// that behaviour: collectives stay on the primary comm, while P2P traffic to
// peer X transparently uses a 2-rank sub-comm built on demand.
//
// The pair comm is just another ProcessGroupNCCL bootstrapped over a
// PrefixStore carved out of the caller's store (c10d PGs boot from a Store,
// so no bespoke ncclUniqueId exchange is needed, unlike torchcomms'
// createPairComm). Inside a pair comm the lower-numbered global rank is local
// rank 0. Batched P2P (startCoalescing/endCoalescing) stays on the primary
// because a batch may touch multiple peers within one ncclGroupStart/End;
// send/recv issued while a batch is active are routed to the primary too.
//
// Thread safety: user-facing calls follow the Backend contract (caller
// serializes them); pair_mu_ additionally protects the pair map so abort()
// (which may fire from a background thread) can walk it safely. The lock is
// released around the slow pair-comm bootstrap.
class TORCH_API ProcessGroupNCCLLazy : public ::c10d::Backend {
 public:
  static constexpr std::string_view kBackendName = "nccl2-lazy";

  ProcessGroupNCCLLazy(
      c10::intrusive_ptr<::c10d::Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupNCCL::Options> options =
          ProcessGroupNCCL::Options::create());
  ~ProcessGroupNCCLLazy() override;

  const std::string getBackendName() const override {
    return std::string(kBackendName);
  }
  c10::intrusive_ptr<::c10d::Backend::Options> getBackendOptions() override {
    return primary_->getBackendOptions();
  }

  // ---- P2P: dispatched to per-peer 2-rank pair comms ----
  c10::intrusive_ptr<::c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;
  c10::intrusive_ptr<::c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  // Batched P2P stays on the primary (multiple peers per group).
  bool supportsCoalescing() const override {
    return true;
  }
  void startCoalescing() override;
  c10::intrusive_ptr<::c10d::Work> endCoalescing() override;

  // ---- Collectives: forwarded to the primary comm ----
  c10::intrusive_ptr<::c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const ::c10d::BroadcastOptions& opts) override {
    return primary_->broadcast(tensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceOptions& opts) override {
    return primary_->allreduce(tensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceCoalescedOptions& opts) override {
    return primary_->allreduce_coalesced(tensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::ReduceOptions& opts) override {
    return primary_->reduce(tensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts) override {
    return primary_->allgather(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts) override {
    return primary_->allgather_coalesced(outputTensorLists, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::AllgatherOptions& opts) override {
    return primary_->allgather_into_tensor_coalesced(outputs, inputs, opts);
  }
  c10::intrusive_ptr<::c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ::c10d::AllgatherOptions& opts) override {
    return primary_->_allgather_base(outputBuffer, inputBuffer, opts);
  }
  c10::intrusive_ptr<::c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::GatherOptions& opts) override {
    return primary_->gather(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ::c10d::ScatterOptions& opts) override {
    return primary_->scatter(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ::c10d::ReduceScatterOptions& opts) override {
    return primary_->reduce_scatter(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::ReduceScatterOptions& opts) override {
    return primary_->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  }
  c10::intrusive_ptr<::c10d::Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ::c10d::ReduceScatterOptions& opts) override {
    return primary_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
  }
  c10::intrusive_ptr<::c10d::Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const ::c10d::AllToAllOptions& opts) override {
    return primary_->alltoall_base(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }
  c10::intrusive_ptr<::c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllToAllOptions& opts) override {
    return primary_->alltoall(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<::c10d::Work> barrier(
      const ::c10d::BarrierOptions& opts) override {
    return primary_->barrier(opts);
  }

  // ---- Windows / memory: forwarded to the primary comm ----
  bool supportsWindow() const override {
    return primary_->supportsWindow();
  }
  c10::intrusive_ptr<::c10d::Window> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override {
    return primary_->new_window(tensor);
  }
  std::shared_ptr<c10::Allocator> getMemAllocator() override {
    return primary_->getMemAllocator();
  }

  // ---- Lifecycle / fault tolerance: fan out to every comm we own ----
  void eagerConnectSingleDevice(at::Device device) override {
    primary_->eagerConnectSingleDevice(device);
  }
  void setTimeout(std::chrono::milliseconds timeout) override;
  void shutdown() override;
  void abort() override;
  ::c10d::ErrorType getError() override {
    return primary_->getError();
  }
  void suspend() override {
    primary_->suspend();
  }
  void resume() override {
    primary_->resume();
  }
  std::unordered_map<std::string, uint64_t> getMemoryStats() override {
    return primary_->getMemoryStats();
  }
  void registerAbortHook(int64_t hook_id, ::c10d::AbortHook hook) override;
  void unregisterAbortHook(int64_t hook_id) override;

  // Reconfigure: the primary reconfigures in place; stale pair comms (built
  // for the previous membership) are aborted and rebuilt lazily on demand.
  bool supportsReconfigure() const override {
    return true;
  }
  ::c10d::ReconfigureHandle get_reconfigure_handle() const override;
  c10::intrusive_ptr<::c10d::Work> reconfigure(
      const ::c10d::ReconfigureOptions& opts) override;

  // ---- Test / introspection helpers ----
  c10::intrusive_ptr<ProcessGroupNCCL> getPrimary() const {
    return primary_;
  }
  size_t numActiveChannels() const {
    std::lock_guard<std::mutex> lk(pair_mu_);
    return pair_comms_.size();
  }

 private:
  // Returns the pair comm for the given peer, creating it on first use.
  c10::intrusive_ptr<ProcessGroupNCCL> channelFor(int peer);
  // The peer's local rank in the 2-rank pair comm: the lower-numbered global
  // rank is local rank 0, so the peer's index is the opposite of ours.
  int peerInPair(int peer) const {
    return (getRank() < peer) ? 1 : 0;
  }
  void dropPairComms();

  c10::intrusive_ptr<::c10d::Store> store_;
  c10::intrusive_ptr<ProcessGroupNCCL::Options> options_;
  c10::intrusive_ptr<ProcessGroupNCCL> primary_;

  mutable std::mutex pair_mu_;
  std::unordered_map<int, c10::intrusive_ptr<ProcessGroupNCCL>> pair_comms_;

  bool coalescing_active_{false};
  std::unordered_map<int64_t, ::c10d::AbortHook> abort_hooks_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
