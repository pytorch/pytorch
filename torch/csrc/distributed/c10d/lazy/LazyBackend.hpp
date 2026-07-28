// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

// LazyBackend wraps an underlying c10d::Backend implementation of type T and
// lazily creates per-peer 2-rank sibling communicators dedicated to
// point-to-point traffic. Port of torchcomms' LazyBackend<T> rebased onto
// c10d::Backend.
//
// Motivation: in older versions of c10d's ProcessGroupNCCL, each P2P peer
// pair owned its own 2-rank comm (and therefore its own stream), bootstrapped
// on first send/recv between the two ranks without any global coordination;
// send/recv to different peers could overlap with each other and with
// collectives. This wrapper reproduces that behaviour for any backend:
// collectives stay on the primary comm, while P2P traffic to peer X
// transparently uses a 2-rank sub-comm built on demand.
//
// Contract on T: subclasses supply a PairFactory that returns a fully
// constructed (but not necessarily bootstrapped) 2-rank sibling backend for
// the pair (this->getRank(), peer). Inside a pair comm the lower-numbered
// global rank is local rank 0. The wrapper supplies a `pair_name` unique
// across allocations for the same {lo,hi} pair, so a store-based bootstrap
// can safely use it as a key prefix without collisions.
//
// Batched P2P (startCoalescing/endCoalescing) stays on the primary because a
// batch may touch multiple peers within one group; send/recv issued while a
// batch is active are routed to the primary too.
//
// Thread safety: user-facing calls follow the Backend contract (the caller
// serializes them); pair_mu_ additionally protects the pair map so abort()
// (which may fire from a background thread) can walk it safely. The lock is
// released around the slow pair-comm construction.
template <typename T>
class LazyBackend : public Backend {
  static_assert(
      std::is_base_of_v<Backend, T>,
      "LazyBackend<T> requires T to derive from c10d::Backend");

 public:
  // Builds the 2-rank sibling backend for a peer. The pair rank is 0 for the
  // lower-numbered global rank and 1 for the higher; pair_name is unique per
  // allocation for the same pair.
  using PairFactory = std::function<
      c10::intrusive_ptr<T>(int pair_rank, const std::string& pair_name)>;

  LazyBackend(
      int rank,
      int size,
      c10::intrusive_ptr<T> primary,
      PairFactory pair_factory)
      : Backend(rank, size),
        primary_(std::move(primary)),
        pair_factory_(std::move(pair_factory)) {
    TORCH_CHECK(primary_, "LazyBackend: null primary backend");
    TORCH_CHECK(pair_factory_, "LazyBackend: null pair factory");
  }

  const std::string getBackendName() const override {
    return primary_->getBackendName() + "-lazy";
  }
  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return primary_->getBackendOptions();
  }

  // ---- P2P: dispatched to per-peer 2-rank pair comms ----
  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    if (coalescing_active_) {
      return primary_->send(tensors, dstRank, tag);
    }
    return channelFor(dstRank)->send(tensors, peerInPair(dstRank), tag);
  }
  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    if (coalescing_active_) {
      return primary_->recv(tensors, srcRank, tag);
    }
    return channelFor(srcRank)->recv(tensors, peerInPair(srcRank), tag);
  }

  // Batched P2P stays on the primary (multiple peers per group).
  bool supportsCoalescing() const override {
    return primary_->supportsCoalescing();
  }
  void startCoalescing() override {
    primary_->startCoalescing();
    coalescing_active_ = true;
  }
  c10::intrusive_ptr<Work> endCoalescing() override {
    coalescing_active_ = false;
    return primary_->endCoalescing();
  }

  // ---- Collectives: forwarded to the primary comm ----
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts) override {
    return primary_->broadcast(tensors, opts);
  }
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts) override {
    return primary_->allreduce(tensors, opts);
  }
  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts) override {
    return primary_->allreduce_coalesced(tensors, opts);
  }
  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts) override {
    return primary_->reduce(tensors, opts);
  }
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts) override {
    return primary_->allgather(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts) override {
    return primary_->allgather_coalesced(outputTensorLists, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts) override {
    return primary_->allgather_into_tensor_coalesced(outputs, inputs, opts);
  }
  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts) override {
    return primary_->_allgather_base(outputBuffer, inputBuffer, opts);
  }
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts) override {
    return primary_->gather(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts) override {
    return primary_->scatter(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts) override {
    return primary_->reduce_scatter(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts) override {
    return primary_->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  }
  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts) override {
    return primary_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
  }
  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts) override {
    return primary_->alltoall_base(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }
  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts) override {
    return primary_->alltoall(outputTensors, inputTensors, opts);
  }
  c10::intrusive_ptr<Work> barrier(const BarrierOptions& opts) override {
    return primary_->barrier(opts);
  }

  // ---- Windows / memory: forwarded to the primary comm ----
  bool supportsWindow() const override {
    return primary_->supportsWindow();
  }
  c10::intrusive_ptr<Window> new_window(
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
  void setTimeout(std::chrono::milliseconds timeout) override {
    primary_->setTimeout(timeout);
    std::lock_guard<std::mutex> lk(pair_mu_);
    for (auto& [_, channel] : pair_comms_) {
      channel->setTimeout(timeout);
    }
  }
  void shutdown() override {
    // Drain pair comms before the primary, since they may share global state
    // (e.g. a caching-allocator hook) with it.
    std::unordered_map<int, c10::intrusive_ptr<T>> drained;
    {
      std::lock_guard<std::mutex> lk(pair_mu_);
      drained.swap(pair_comms_);
    }
    for (auto& [_, channel] : drained) {
      channel->shutdown();
    }
    primary_->shutdown();
  }
  // abort() may fire from a background thread. Hold pair_mu_ only long enough
  // to walk the map; each child's abort is itself non-blocking.
  void abort() override {
    primary_->abort();
    std::lock_guard<std::mutex> lk(pair_mu_);
    for (auto& [_, channel] : pair_comms_) {
      channel->abort();
    }
  }
  ErrorType getError() override {
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
  void registerAbortHook(int64_t hook_id, AbortHook hook) override {
    abort_hooks_.emplace(hook_id, hook);
    primary_->registerAbortHook(hook_id, hook);
    std::lock_guard<std::mutex> lk(pair_mu_);
    for (auto& [_, channel] : pair_comms_) {
      channel->registerAbortHook(hook_id, hook);
    }
  }
  void unregisterAbortHook(int64_t hook_id) override {
    abort_hooks_.erase(hook_id);
    primary_->unregisterAbortHook(hook_id);
    std::lock_guard<std::mutex> lk(pair_mu_);
    for (auto& [_, channel] : pair_comms_) {
      channel->unregisterAbortHook(hook_id);
    }
  }

  // Reconfigure: the primary reconfigures in place; stale pair comms (built
  // for the previous membership) are aborted and rebuilt lazily on demand.
  bool supportsReconfigure() const override {
    return primary_->supportsReconfigure();
  }
  ReconfigureHandle get_reconfigure_handle() const override {
    return primary_->get_reconfigure_handle();
  }
  c10::intrusive_ptr<Work> reconfigure(
      const ReconfigureOptions& opts) override {
    // Pair comms encode the previous membership's global ranks; they cannot
    // be carried across a reconfigure. Abort them and let P2P traffic rebuild
    // fresh channels lazily against the new membership.
    dropPairComms();
    auto work = primary_->reconfigure(opts);
    rank_ = primary_->getRank();
    size_ = primary_->getSize();
    return work;
  }

  // ---- Test / introspection helpers ----
  c10::intrusive_ptr<T> getPrimary() const {
    return primary_;
  }
  size_t numActiveChannels() const {
    std::lock_guard<std::mutex> lk(pair_mu_);
    return pair_comms_.size();
  }

 protected:
  // Returns the pair comm for the given peer, creating it on first use.
  c10::intrusive_ptr<T> channelFor(int peer) {
    TORCH_CHECK(
        peer != getRank() && peer >= 0 && peer < getSize(),
        "LazyBackend: invalid peer rank ",
        peer,
        " (self=",
        getRank(),
        ", size=",
        getSize(),
        ")");
    {
      std::lock_guard<std::mutex> lk(pair_mu_);
      auto it = pair_comms_.find(peer);
      if (it != pair_comms_.end()) {
        return it->second;
      }
    }

    // Slow path: build the pair comm without holding the lock so abort() and
    // other map-walking paths stay responsive. Under the Backend
    // single-threaded user contract no other thread races us on this peer, so
    // re-inserting after construction is safe.
    const int lo = std::min(getRank(), peer);
    const int hi = std::max(getRank(), peer);
    const std::string pair_name = c10::str(
        getGroupUid().empty() ? getBackendName() : getGroupUid(),
        "/p2p-",
        lo,
        "-",
        hi,
        "-",
        nextPairAttempt(lo, hi));

    const int pair_rank = (getRank() < peer) ? 0 : 1;
    auto sub = pair_factory_(pair_rank, pair_name);
    TORCH_CHECK(sub, "LazyBackend: pair factory returned null for peer ", peer);
    if (getBoundDeviceId().has_value()) {
      sub->setBoundDeviceId(getBoundDeviceId());
    }

    // Fan registered hooks out to the new channel so user-registered abort
    // hooks observe events from every comm we own.
    for (const auto& [hook_id, hook] : abort_hooks_) {
      sub->registerAbortHook(hook_id, hook);
    }

    std::lock_guard<std::mutex> lk(pair_mu_);
    auto [it, inserted] = pair_comms_.emplace(peer, std::move(sub));
    return it->second;
  }

  // The peer's local rank in the 2-rank pair comm: the lower-numbered global
  // rank is local rank 0, so the peer's index is the opposite of ours.
  int peerInPair(int peer) const {
    return (getRank() < peer) ? 1 : 0;
  }

  void dropPairComms() {
    std::unordered_map<int, c10::intrusive_ptr<T>> dropped;
    {
      std::lock_guard<std::mutex> lk(pair_mu_);
      dropped.swap(pair_comms_);
    }
    for (auto& [_, channel] : dropped) {
      channel->abort();
    }
  }

 private:
  // Per-pair monotonically increasing counter so successive pair-comm
  // allocations for the same {lo,hi} produce distinct names (and therefore
  // distinct store-bootstrap key namespaces). Both ranks of a pair increment
  // in lockstep because each create on one side has a matching create on the
  // other. Process-wide static so it is shared across LazyBackend instances.
  static int nextPairAttempt(int lo, int hi) {
    static std::mutex mu;
    static std::unordered_map<int64_t, int> counters;
    const int64_t key =
        (static_cast<int64_t>(lo) << 32) | static_cast<uint32_t>(hi);
    std::lock_guard<std::mutex> guard(mu);
    return counters[key]++;
  }

  c10::intrusive_ptr<T> primary_;
  PairFactory pair_factory_;

  mutable std::mutex pair_mu_;
  std::unordered_map<int, c10::intrusive_ptr<T>> pair_comms_;

  bool coalescing_active_{false};
  std::unordered_map<int64_t, AbortHook> abort_hooks_;
};

} // namespace c10d
