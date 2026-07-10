// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/FlightRecorderHook.hpp>

#include <atomic>

namespace c10d {

namespace {

// Hook ids must not collide with user-registered hooks; carve out a range
// far above small hand-picked ids.
std::atomic<int64_t> next_hook_id{0x46524543 /* 'FREC' */};

std::string_view hookOpName(HookOpName name) {
  switch (name) {
    case HookOpName::SEND:
      return "send";
    case HookOpName::RECV:
      return "recv";
    case HookOpName::BROADCAST:
      return "broadcast";
    case HookOpName::ALLREDUCE:
      return "allreduce";
    case HookOpName::REDUCE:
      return "reduce";
    case HookOpName::ALLGATHER:
      return "allgather";
    case HookOpName::REDUCE_SCATTER:
      return "reduce_scatter";
    case HookOpName::ALLTOALL:
      return "alltoall";
    case HookOpName::BARRIER:
      return "barrier";
    case HookOpName::SCATTER:
      return "scatter";
    case HookOpName::GATHER:
      return "gather";
    case HookOpName::SPLIT:
      return "split";
    case HookOpName::NEW_WINDOW:
      return "new_window";
    case HookOpName::UNKNOWN:
      break;
  }
  return "unknown";
}

bool isP2POp(HookOpName name) {
  return name == HookOpName::SEND || name == HookOpName::RECV;
}

// FlightRecorder keys pg_id by a per-recorder monotonic id in the built-in
// backends (ProcessGroupGloo's local_id_, ProcessGroupNCCL's local_id_). Use
// a separate counter for hook-attached groups.
std::atomic<size_t> next_pg_id{0};

} // namespace

std::shared_ptr<FlightRecorderHook> FlightRecorderHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg) {
  auto hook = std::shared_ptr<FlightRecorderHook>(
      new FlightRecorderHook(std::move(pg)));
  // Registration is deferred out of the constructor (shared_from_this is
  // invalid in a ctor). The lambdas hold a weak_ptr so the hook -> pg -> hook
  // cycle is broken: when the caller drops the returned handle, the hook
  // destructor unregisters from the process group.
  std::weak_ptr<FlightRecorderHook> weak = hook;
  hook->pg_->registerPreHook(hook->hook_id_, [weak](const PreHookArgs& args) {
    if (auto self = weak.lock()) {
      self->onPre(args);
    }
  });
  hook->pg_->registerPostHook(hook->hook_id_, [weak](const PostHookArgs& args) {
    if (auto self = weak.lock()) {
      self->onPost(args);
    }
  });
  return hook;
}

FlightRecorderHook::FlightRecorderHook(c10::intrusive_ptr<ProcessGroup> pg)
    : pg_(std::move(pg)),
      hook_id_(next_hook_id++),
      pg_id_(next_pg_id++),
      pg_status_(std::make_shared<ProcessGroupStatus>()) {
  TORCH_CHECK(pg_, "FlightRecorderHook: null process group");
  // Backend options are optional on custom backends (getBackendOptions
  // throws by default); fall back to identity ranks and the default timeout.
  std::vector<uint64_t> ranks;
  try {
    auto options = pg_->getDefaultBackend()->getBackendOptions();
    ranks = options->global_ranks_in_group;
    timeout_ = options->timeout;
  } catch (const std::exception&) {
  }
  if (ranks.empty()) {
    ranks.reserve(pg_->getSize());
    for (int r = 0; r < pg_->getSize(); ++r) {
      ranks.push_back(static_cast<uint64_t>(r));
    }
  }
  FlightRecorder<c10::Event>::get()->record_pg_ranks(
      std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
      std::move(ranks));
}

FlightRecorderHook::~FlightRecorderHook() {
  remove();
}

void FlightRecorderHook::remove() {
  if (pg_) {
    pg_->unregisterPreHook(hook_id_);
    pg_->unregisterPostHook(hook_id_);
    pg_.reset();
  }
}

void FlightRecorderHook::onPre(const PreHookArgs& args) {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool is_p2p = isP2POp(args.name);
  size_t collective_seq = is_p2p ? collective_seq_ : ++collective_seq_;
  size_t p2p_seq = is_p2p ? ++p2p_seq_ : p2p_seq_;

  pg_status_->lastEnqueuedSeq = static_cast<int64_t>(args.op_id);
  pg_status_->lastEnqueuedWorkName = std::string(hookOpName(args.name));

  // Null start/end events: no GPU duration (same as ProcessGroupGloo's
  // built-in FR recording), but full op/tensor/sequencing metadata.
  auto trace_id = FlightRecorder<c10::Event>::get()->recordWithResetEnabled(
      pg_id_,
      std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
      collective_seq,
      p2p_seq,
      static_cast<size_t>(args.op_id),
      c10::str("c10d:", hookOpName(args.name)),
      args.input_tensors,
      args.output_tensors,
      /*start=*/nullptr,
      /*end=*/nullptr,
      timeout_,
      pg_status_,
      is_p2p);
  inflight_.emplace(args.op_id, trace_id);
}

void FlightRecorderHook::onPost(const PostHookArgs& args) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = inflight_.find(args.op_id);
  if (it == inflight_.end()) {
    return;
  }
  pg_status_->lastCompletedSeq = static_cast<int64_t>(args.op_id);
  pg_status_->lastCompletedWorkName = std::string(hookOpName(args.name));
  FlightRecorder<c10::Event>::get()->retire_id(
      it->second.id, it->second.reset_epoch, /*compute_duration=*/false);
  inflight_.erase(it);
}

} // namespace c10d
