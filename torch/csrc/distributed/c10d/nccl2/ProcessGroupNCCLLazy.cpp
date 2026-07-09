// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLLazy.hpp>

#include <algorithm>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>

namespace c10d::nccl2 {

namespace {

// Per-pair monotonically increasing counter so successive pair-comm
// allocations for the same {lo,hi} produce distinct names (and therefore
// distinct store-bootstrap key namespaces). Both ranks of a pair increment in
// lockstep because each create on one side has a matching create on the
// other. Process-wide static so it is shared across ProcessGroupNCCLLazy
// instances (port of torchcomms' LazyBackend::nextPairAttempt).
int nextPairAttempt(int lo, int hi) {
  static std::mutex mu;
  static std::unordered_map<int64_t, int> counters;
  const int64_t key =
      (static_cast<int64_t>(lo) << 32) | static_cast<uint32_t>(hi);
  std::lock_guard<std::mutex> guard(mu);
  return counters[key]++;
}

} // namespace

ProcessGroupNCCLLazy::ProcessGroupNCCLLazy(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<ProcessGroupNCCL::Options> options)
    : Backend(rank, size),
      store_(std::move(store)),
      options_(
          options ? std::move(options) : ProcessGroupNCCL::Options::create()) {
  primary_ =
      c10::make_intrusive<ProcessGroupNCCL>(store_, rank, size, options_);
}

ProcessGroupNCCLLazy::~ProcessGroupNCCLLazy() = default;

c10::intrusive_ptr<ProcessGroupNCCL> ProcessGroupNCCLLazy::channelFor(
    int peer) {
  TORCH_CHECK(
      peer != getRank() && peer >= 0 && peer < getSize(),
      "ProcessGroupNCCLLazy: invalid peer rank ",
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
      getGroupUid().empty() ? std::string(kBackendName) : getGroupUid(),
      "/p2p-",
      lo,
      "-",
      hi,
      "-",
      nextPairAttempt(lo, hi));

  auto pair_store = c10::make_intrusive<::c10d::PrefixStore>(pair_name, store_);
  auto pair_options = ProcessGroupNCCL::Options::create();
  pair_options->timeout = options_->timeout;
  pair_options->is_high_priority_stream = options_->is_high_priority_stream;
  pair_options->abort_process_on_timeout_or_error =
      options_->abort_process_on_timeout_or_error;
  pair_options->hints = options_->hints;
  pair_options->group_name = pair_name;

  const int pair_rank = (getRank() < peer) ? 0 : 1;
  auto sub = c10::make_intrusive<ProcessGroupNCCL>(
      pair_store, pair_rank, /*size=*/2, pair_options);
  // The NCCL bootstrap itself stays lazy: it runs on the first send/recv,
  // which knows the tensor's device.
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

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCLLazy::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  if (coalescing_active_) {
    return primary_->send(tensors, dstRank, tag);
  }
  return channelFor(dstRank)->send(tensors, peerInPair(dstRank), tag);
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCLLazy::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  if (coalescing_active_) {
    return primary_->recv(tensors, srcRank, tag);
  }
  return channelFor(srcRank)->recv(tensors, peerInPair(srcRank), tag);
}

void ProcessGroupNCCLLazy::startCoalescing() {
  // Batched P2P may touch multiple peers within a single ncclGroupStart/End,
  // so the whole window (including its send/recv) runs on the primary.
  primary_->startCoalescing();
  coalescing_active_ = true;
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCLLazy::endCoalescing() {
  coalescing_active_ = false;
  return primary_->endCoalescing();
}

void ProcessGroupNCCLLazy::setTimeout(std::chrono::milliseconds timeout) {
  primary_->setTimeout(timeout);
  std::lock_guard<std::mutex> lk(pair_mu_);
  for (auto& [_, channel] : pair_comms_) {
    channel->setTimeout(timeout);
  }
}

void ProcessGroupNCCLLazy::shutdown() {
  // Drain pair comms before the primary, since they share global state (the
  // caching-allocator hook) with it.
  std::unordered_map<int, c10::intrusive_ptr<ProcessGroupNCCL>> drained;
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
void ProcessGroupNCCLLazy::abort() {
  primary_->abort();
  std::lock_guard<std::mutex> lk(pair_mu_);
  for (auto& [_, channel] : pair_comms_) {
    channel->abort();
  }
}

void ProcessGroupNCCLLazy::registerAbortHook(
    int64_t hook_id,
    ::c10d::AbortHook hook) {
  abort_hooks_.emplace(hook_id, hook);
  primary_->registerAbortHook(hook_id, hook);
  std::lock_guard<std::mutex> lk(pair_mu_);
  for (auto& [_, channel] : pair_comms_) {
    channel->registerAbortHook(hook_id, hook);
  }
}

void ProcessGroupNCCLLazy::unregisterAbortHook(int64_t hook_id) {
  abort_hooks_.erase(hook_id);
  primary_->unregisterAbortHook(hook_id);
  std::lock_guard<std::mutex> lk(pair_mu_);
  for (auto& [_, channel] : pair_comms_) {
    channel->unregisterAbortHook(hook_id);
  }
}

void ProcessGroupNCCLLazy::dropPairComms() {
  std::unordered_map<int, c10::intrusive_ptr<ProcessGroupNCCL>> dropped;
  {
    std::lock_guard<std::mutex> lk(pair_mu_);
    dropped.swap(pair_comms_);
  }
  for (auto& [_, channel] : dropped) {
    channel->abort();
  }
}

::c10d::ReconfigureHandle ProcessGroupNCCLLazy::get_reconfigure_handle() const {
  return primary_->get_reconfigure_handle();
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCLLazy::reconfigure(
    const ::c10d::ReconfigureOptions& opts) {
  // Pair comms encode the previous membership's global ranks; they cannot be
  // carried across a reconfigure. Abort them and let P2P traffic rebuild
  // fresh channels lazily against the new membership.
  dropPairComms();
  auto work = primary_->reconfigure(opts);
  rank_ = primary_->getRank();
  size_ = primary_->getSize();
  return work;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
