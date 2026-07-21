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
      return "all_reduce";
    case HookOpName::REDUCE:
      return "reduce";
    case HookOpName::ALLGATHER:
      return "all_gather";
    case HookOpName::REDUCE_SCATTER:
      return "reduce_scatter";
    case HookOpName::ALLTOALL:
      return "all_to_all";
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

bool tracksWorkCompletion(std::string_view backend) {
  // Other backends are not guaranteed to signal completion through Work.
  return backend == "gloo" || backend == "nccl" || backend == "nccl2" ||
      backend == "nccl-lazy";
}

// FlightRecorder keys process groups by a per-recorder monotonic id.
std::atomic<size_t> next_pg_id{0};

} // namespace

std::shared_ptr<FlightRecorderHook> FlightRecorderHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg) {
  TORCH_CHECK(pg, "FlightRecorderHook: null process group");
  if (pg->flight_recorder_hook_ && pg->flight_recorder_hook_->pg_) {
    return pg->flight_recorder_hook_;
  }
  auto* pg_ptr = pg.get();
  auto hook = std::shared_ptr<FlightRecorderHook>(
      new FlightRecorderHook(pg_ptr, std::move(pg)));
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

std::shared_ptr<FlightRecorderHook> FlightRecorderHook::attachOwned(
    ProcessGroup* pg) {
  TORCH_CHECK(pg, "FlightRecorderHook: null process group");
  auto hook = std::shared_ptr<FlightRecorderHook>(new FlightRecorderHook(pg));
  // Registration is deferred out of the constructor (shared_from_this is
  // invalid in a ctor). The lambdas hold a weak_ptr so the group owns the hook
  // without creating a cycle.
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

bool FlightRecorderHook::isEnabled() {
  return FlightRecorder<c10::Event>::get()->enabled_;
}

FlightRecorderHook::FlightRecorderHook(
    ProcessGroup* pg,
    c10::intrusive_ptr<ProcessGroup> pg_keepalive)
    : pg_(pg),
      pg_keepalive_(std::move(pg_keepalive)),
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
    ranks.clear();
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
    pg_ = nullptr;
    pg_keepalive_.reset();
  }
}

std::string FlightRecorderHook::backendName(const PreHookArgs& args) const {
  const at::Tensor* tensor = nullptr;
  if (!args.input_tensors.empty()) {
    tensor = &args.input_tensors.front();
  } else if (!args.output_tensors.empty()) {
    tensor = &args.output_tensors.front();
  }
  if (tensor && tensor->defined()) {
    return pg_->getBackend(tensor->device().type())->getBackendName();
  }
  return pg_->getDefaultBackend()->getBackendName();
}

void FlightRecorderHook::onPre(const PreHookArgs& args) {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool is_p2p = isP2POp(args.name);
  size_t collective_seq = is_p2p ? collective_seq_ : ++collective_seq_;
  size_t p2p_seq = is_p2p ? ++p2p_seq_ : p2p_seq_;

  pg_status_->lastEnqueuedSeq =
      static_cast<int64_t>(is_p2p ? p2p_seq : collective_seq);
  pg_status_->lastEnqueuedWorkName = std::string(hookOpName(args.name));

  auto inputs = args.input_tensors;
  auto outputs = args.output_tensors;
  if (inputs.empty() && args.name == HookOpName::RECV) {
    inputs = outputs;
  }
  if (outputs.empty() &&
      (args.name == HookOpName::BROADCAST ||
       args.name == HookOpName::ALLREDUCE || args.name == HookOpName::REDUCE ||
       args.name == HookOpName::BARRIER || args.name == HookOpName::SEND)) {
    outputs = inputs;
  }
  auto backend = backendName(args);

  auto trace_id = FlightRecorder<c10::Event>::get()->recordWithResetEnabled(
      pg_id_,
      std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
      collective_seq,
      p2p_seq,
      static_cast<size_t>(args.op_id),
      c10::str(backend, ":", hookOpName(args.name)),
      inputs,
      outputs,
      /*start=*/nullptr,
      /*end=*/nullptr,
      timeout_,
      pg_status_,
      is_p2p);
  inflight_.emplace(
      args.op_id,
      InflightTrace{
          trace_id,
          static_cast<int64_t>(is_p2p ? p2p_seq : collective_seq),
          std::string(hookOpName(args.name)),
          tracksWorkCompletion(backend)});
}

void FlightRecorderHook::onPost(const PostHookArgs& args) {
  std::optional<InflightTrace> trace;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = inflight_.find(args.op_id);
    if (it == inflight_.end()) {
      return;
    }
    trace = std::move(it->second);
    inflight_.erase(it);
  }

  auto mark_completed =
      [status = pg_status_, sequence = trace->sequence, name = trace->name]() {
        status->lastCompletedSeq = sequence;
        status->lastCompletedWorkName = name;
      };
  if (args.work && trace->track_completion) {
    args.work->setFlightRecorderTrace(
        trace->id.id, trace->id.reset_epoch, std::move(mark_completed));
  } else {
    mark_completed();
    FlightRecorder<c10::Event>::get()->retire_id(
        trace->id.id, trace->id.reset_epoch, /*compute_duration=*/false);
  }
}

} // namespace c10d
