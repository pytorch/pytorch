// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/FlightRecorderHook.hpp>

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <atomic>

namespace c10d {

namespace {

// Hook ids must not collide with user-registered hooks; carve out a range
// far above small hand-picked ids.
std::atomic<int64_t> next_hook_id{0x46524543 /* 'FREC' */};

// These must be spelled the way the trace analyzer spells them, i.e. match
// COLLECTIVES / P2P in torch/distributed/flight_recorder/components/types.py
// and the profiling titles the native backends use ("nccl:all_reduce",
// "gloo:all_gather", ...). Anything else makes Op() reject the entry.
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

// The analyzer parses profiling_name as "<comm_lib>:<op>", plus a
// "<rank>-><peer>" / "<rank><-<peer>" suffix for p2p. Both ranks are
// group-local: Ops.cpp passes the dst/src it received from
// distributed_c10d.py, which canonicalizes to a group rank before calling
// ProcessGroup::send/recv, and ProcessGroupNCCL formats its own p2p names the
// same way from Backend::rank_.
//
// The comm_lib field stays the literal "c10d" rather than the backend name.
// ProcessGroup::getBackendName() returns "custom" for every backend outside
// the built-in BackendType enum (nccl2 included), and the backend's own name
// is an arbitrary user-chosen string that the analyzer's allowlist cannot
// track. It also keeps hook entries distinguishable from the ones a backend
// records natively into the same recorder (ProcessGroupGloo::enqueue).
std::string profilingName(const PreHookArgs& args, int rank) {
  auto op = hookOpName(args.name);
  switch (args.name) {
    case HookOpName::SEND:
      return c10::str("c10d:", op, " ", rank, "->", args.root);
    case HookOpName::RECV:
      return c10::str("c10d:", op, " ", rank, "<-", args.root);
    default:
      return c10::str("c10d:", op);
  }
}

// PreHookArgs carries no device, so take it from the op's tensors.
std::optional<c10::Device> opDevice(const PreHookArgs& args) {
  if (!args.input_tensors.empty() && args.input_tensors.front().defined()) {
    return args.input_tensors.front().device();
  }
  if (!args.output_tensors.empty() && args.output_tensors.front().defined()) {
    return args.output_tensors.front().device();
  }
  return std::nullopt;
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
  // Global rank of this process, for naming the dump file. Every group this
  // process belongs to maps its own rank back to the same global rank, so
  // attaching to several groups keeps setting the same value.
  auto local_rank = static_cast<size_t>(pg_->getRank());
  if (local_rank < ranks.size()) {
    FlightRecorder<c10::Event>::get()->setRank(
        static_cast<int>(ranks[local_rank]));
  }
  FlightRecorder<c10::Event>::get()->record_pg_ranks(
      std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
      std::move(ranks));
}

FlightRecorderHook::~FlightRecorderHook() {
  remove();
}

void FlightRecorderHook::remove() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pg_) {
    return;
  }
  pg_->unregisterPreHook(hook_id_);
  pg_->unregisterPostHook(hook_id_);
  pg_.reset();
  // Ops still in flight have entries borrowing our events; retire them so the
  // recorder drops those pointers before the events are freed below.
  for (auto& [op_id, op] : inflight_) {
    retire(op, /*record_end=*/false);
  }
  inflight_.clear();
}

void FlightRecorderHook::retire(InflightOp& op, bool record_end) {
  if (record_end && op.end && op.stream) {
    try {
      op.end->record(*op.stream);
    } catch (const std::exception& e) {
      LOG(ERROR) << "FlightRecorderHook: failed to record end event: "
                 << e.what();
    }
  }
  // The post-hook fires when the op is issued, so the end event has usually
  // not signalled yet and the duration comes from the host clock instead. Ops
  // abandoned by remove() never completed, so they get no duration at all.
  FlightRecorder<c10::Event>::get()->retire_id(
      op.trace_id.id,
      op.trace_id.reset_epoch,
      /*compute_duration=*/record_end,
      /*wall_clock_fallback=*/record_end);
}

void FlightRecorderHook::onPre(const PreHookArgs& args) {
  InflightOp op;
  auto device = opDevice(args);
  auto* recorder = FlightRecorder<c10::Event>::get();
  if (recorder->enabled_ && device && !device->is_cpu()) {
    // Device events let the recorder discover whether the op started and
    // completed. Created before taking mutex_ so a slow device API cannot
    // stall hooks on other threads.
    try {
      auto* guard_impl = c10::impl::getDeviceGuardImpl(device->type());
      op.start = std::make_unique<c10::Event>(
          device->type(), c10::EventFlag::BACKEND_DEFAULT);
      op.end = std::make_unique<c10::Event>(
          device->type(), c10::EventFlag::BACKEND_DEFAULT);
      op.stream = guard_impl->getStream(*device);
      op.start->record(*op.stream);
    } catch (const std::exception& e) {
      // Backend has no device guard impl or no event support; fall back to
      // null events.
      LOG(WARNING) << "FlightRecorderHook: no device events for "
                   << device->str() << ": " << e.what();
      op.start.reset();
      op.end.reset();
      op.stream.reset();
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!pg_) {
    // remove() ran while this hook call was blocked on mutex_.
    return;
  }
  const bool is_p2p = isP2POp(args.name);
  size_t collective_seq = is_p2p ? collective_seq_ : ++collective_seq_;
  size_t p2p_seq = is_p2p ? ++p2p_seq_ : p2p_seq_;

  pg_status_->lastEnqueuedSeq = static_cast<int64_t>(args.op_id);
  pg_status_->lastEnqueuedWorkName = std::string(hookOpName(args.name));

  // The entry borrows the events; they stay owned by inflight_ until the
  // matching retire() clears the entry's pointers.
  op.trace_id = recorder->recordWithResetEnabled(
      pg_id_,
      std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc()),
      collective_seq,
      p2p_seq,
      static_cast<size_t>(args.op_id),
      profilingName(args, pg_->getRank()),
      args.input_tensors,
      args.output_tensors,
      op.start.get(),
      op.end.get(),
      timeout_,
      pg_status_,
      is_p2p);
  auto [it, inserted] = inflight_.try_emplace(args.op_id, std::move(op));
  if (!inserted) {
    // op_id is monotonic per process group, so a collision means the previous
    // op never saw its post-hook. Retire it before dropping its events.
    retire(it->second, /*record_end=*/false);
    it->second = std::move(op);
  }
}

void FlightRecorderHook::onPost(const PostHookArgs& args) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = inflight_.find(args.op_id);
  if (it == inflight_.end()) {
    return;
  }
  pg_status_->lastCompletedSeq = static_cast<int64_t>(args.op_id);
  pg_status_->lastCompletedWorkName = std::string(hookOpName(args.name));
  retire(it->second, /*record_end=*/true);
  // Erase (and free the events) only after retire_id() returned, i.e. after
  // the entry stopped pointing at them.
  inflight_.erase(it);
}

} // namespace c10d
