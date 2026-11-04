// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/FlightRecorderHook.hpp>

#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/Type.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <set>
#include <utility>
#include <vector>

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
    case HookOpName::ALLREDUCE_COALESCED:
      return "allreduce_coalesced";
    case HookOpName::REDUCE:
      return "reduce";
    case HookOpName::ALLGATHER:
      return "all_gather";
    case HookOpName::ALLGATHER_BASE:
      return "_all_gather_base";
    case HookOpName::ALLGATHER_COALESCED:
      return "allgather_coalesced";
    case HookOpName::ALLGATHER_INTO_TENSOR_COALESCED:
      return "all_gather_into_tensor_coalesced";
    case HookOpName::REDUCE_SCATTER:
      return "reduce_scatter";
    case HookOpName::REDUCE_SCATTER_BASE:
      return "_reduce_scatter_base";
    case HookOpName::REDUCE_SCATTER_TENSOR_COALESCED:
      return "reduce_scatter_tensor_coalesced";
    case HookOpName::ALLTOALL:
      return "all_to_all";
    case HookOpName::ALLTOALL_BASE:
      return "all_to_all_single";
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

// The dispatcher schema for these has a single tensor list because they
// operate in place, so Ops.cpp fires the pre-hook with no output tensors and
// the entry would carry no output sizes or dtypes at all. The analyzer matches
// a rank's input sizes against its peers' output sizes, so that reads as a
// size mismatch on every rank (and, for reduce, as an unguarded index into an
// empty output list). The native backends record the same buffer on both sides
// -- ProcessGroupNCCL hands allreduce, reduce and broadcast the same tensor as
// input and output -- so mirror the inputs here. Not in Ops.cpp: the pre-hook
// args are shared with every other consumer.
bool recordsInPlace(HookOpName name) {
  return name == HookOpName::ALLREDUCE ||
      name == HookOpName::ALLREDUCE_COALESCED || name == HookOpName::REDUCE ||
      name == HookOpName::BROADCAST;
}

// The analyzer parses profiling_name as "<comm_lib>:<op>", plus a
// "<rank>-><peer>" / "<rank><-<peer>" suffix for p2p. Both ranks are
// group-local: Ops.cpp passes the dst/src it received from
// distributed_c10d.py, which canonicalizes to a group rank before calling
// ProcessGroup::send/recv, and ProcessGroupNCCL formats its own p2p names the
// same way from Backend::rank_.
//
// The comm_lib field is the backend's own name, taken from
// Backend::getBackendName() -- not ProcessGroup::getBackendName(), which maps
// through the BackendType enum and answers "custom" for everything outside it,
// nccl2 included.
std::string profilingName(
    const PreHookArgs& args,
    const std::string& backend,
    int rank) {
  auto op = hookOpName(args.name);
  switch (args.name) {
    case HookOpName::SEND:
      return c10::str(backend, ":", op, " ", rank, "->", args.root);
    case HookOpName::RECV:
      // recvAnysource has no peer to name -- the source is only known once a
      // message arrives -- and Ops.cpp fires the hook with root=-1 for it.
      // "?" is the analyzer's spelling for "unknown"; writing -1 instead made
      // it index the group's rank list from the end and pin the recv on the
      // highest-ranked member, inventing a peer that never sent anything.
      return args.root < 0
          ? c10::str(backend, ":", op, " ", rank, "<-?")
          : c10::str(backend, ":", op, " ", rank, "<-", args.root);
    default:
      return c10::str(backend, ":", op);
  }
}

// Backend names are arbitrary strings, but profiling_name is parsed by
// splitting on ":" into exactly two fields, so a colon inside the name has to
// go.
std::string sanitizeBackendName(std::string name) {
  std::replace(name.begin(), name.end(), ':', '_');
  return name;
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
// backends (ProcessGroupGloo's local_id_, ProcessGroupNCCL's local_id_). One
// process-wide counter for every hooked group: hooked groups never share a
// recorder with a natively recording backend (their ops are skipped), and two
// hooked groups get distinct ids from here whether or not they share one.
std::atomic<size_t> next_pg_id{0};

// Both accessors throw rather than return empty on a backend that does not
// implement them (Backend's defaults, which custom backends keep), and neither
// is required to record, so ask by trying.
c10::intrusive_ptr<Backend> tryGetBackend(
    ProcessGroup& pg,
    std::optional<c10::DeviceType> device) {
  try {
    return device ? pg.getBackend(*device) : pg.getDefaultBackend();
  } catch (const std::exception&) {
    return nullptr;
  }
}

c10::intrusive_ptr<Backend::Options> tryGetOptions(
    const c10::intrusive_ptr<Backend>& backend) {
  try {
    return backend->getBackendOptions();
  } catch (const std::exception&) {
    return nullptr;
  }
}

// Whether a CUDA graph capture (or the equivalent on another accelerator) is
// active on the stream an op on this device would be issued on. Asked through
// the device guard impl so the hook stays device agnostic.
//
// Both ways of answering false here are positive facts, not guesses: a device
// type with no guard impl registered has no capture mechanism at all, and a
// device whose current stream cannot even be obtained -- CUDA linked in but no
// visible GPU, say -- has no stream for a capture to be running on. "I do not
// know which device to ask" is a different answer and is not expressed here;
// see captureActive.
bool streamIsCapturing(c10::Device device) {
  if (device.is_cpu() || !c10::impl::hasDeviceGuardImpl(device.type())) {
    return false;
  }
  try {
    auto* guard_impl = c10::impl::getDeviceGuardImpl(device.type());
    return guard_impl->getStream(device).is_capturing();
  } catch (const std::exception&) {
    return false;
  }
}

} // namespace

std::shared_ptr<FlightRecorderHook> FlightRecorderHook::attach(
    c10::intrusive_ptr<ProcessGroup> pg,
    std::vector<uint64_t> global_ranks) {
  auto hook = std::shared_ptr<FlightRecorderHook>(
      new FlightRecorderHook(std::move(pg), std::move(global_ranks)));
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
  // The dump-on-failure trigger. Abort hooks are optional -- gloo has none and
  // Backend's default implementation throws -- and recording is useful either
  // way, so ask before registering rather than swallowing the exception, which
  // would also hide a registration that failed for a real reason.
  if (hook->pg_->supportsAbortHooks()) {
    hook->pg_->registerAbortHook(hook->hook_id_, [weak]() {
      if (auto self = weak.lock()) {
        self->onAbort();
      }
    });
    hook->abort_hook_registered_ = true;
  }
  return hook;
}

FlightRecorderHook::FlightRecorderHook(
    c10::intrusive_ptr<ProcessGroup> pg,
    std::vector<uint64_t> global_ranks)
    : pg_(std::move(pg)),
      hook_id_(next_hook_id++),
      pg_id_(next_pg_id++),
      pg_status_(std::make_shared<ProcessGroupStatus>()) {
  TORCH_CHECK(pg_, "FlightRecorderHook: null process group");

  auto makeTarget = [](const c10::intrusive_ptr<Backend>& backend) {
    BackendTarget target;
    auto name = sanitizeBackendName(backend->getBackendName());
    if (!name.empty()) {
      target.name = std::move(name);
    }
    // Backends that write to a FlightRecorder themselves keep their entries:
    // ProcessGroupGloo records in enqueue(), so recording its ops here too
    // would put a second, independently sequenced entry in the trace for every
    // gloo collective. Resolved per device rather than per group so a mixed
    // "cpu:gloo,cuda:nccl2" group still gets its CUDA half recorded.
    if (!recordsFlightRecorderNatively(target.name)) {
      target.recorder = getFlightRecorder(target.name);
    }
    if (auto options = tryGetOptions(backend)) {
      target.timeout = options->timeout;
    }
    return target;
  };

  if (auto backend = tryGetBackend(*pg_, std::nullopt)) {
    default_target_ = makeTarget(backend);
    if (global_ranks.empty()) {
      if (auto options = tryGetOptions(backend)) {
        global_ranks = options->global_ranks_in_group;
      }
    }
  }
  for (const auto& device : pg_->getDeviceTypes()) {
    if (auto backend = tryGetBackend(*pg_, device.type())) {
      targets_[device.type()] = makeTarget(backend);
    }
  }

  auto local_rank = static_cast<size_t>(pg_->getRank());
  auto pg_name = std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc());
  std::set<FlightRecorder<c10::Event>*> recorders{default_target_.recorder};
  for (const auto& [device, target] : targets_) {
    recorders.insert(target.recorder);
  }
  recorders.erase(nullptr);
  // Publishing a mapping we do not have is worse than publishing none. For a
  // subgroup, 0..size-1 makes its members claim global ranks that belong to
  // other processes: record_pg_ranks then reports a membership that never
  // existed, and setRank makes several ranks write to the same
  // <prefix><rank> dump file, so all but one post-mortem is lost. Backends
  // that fill in Options::global_ranks_in_group (gloo, nccl, nccl2,
  // nccl-lazy, xccl) never get here; mpi, ucc, fake and out-of-tree plugins
  // do unless the caller supplied the mapping.
  const bool ranks_known = local_rank < global_ranks.size();
  if (!ranks_known && !recorders.empty()) {
    LOG(WARNING)
        << "FlightRecorderHook: no global rank mapping for process group "
        << pg_->getGroupName()
        << "; its collectives are recorded but the group's membership and "
           "this rank's dump file name are left unset.";
  }
  for (auto* recorder : recorders) {
    if (ranks_known) {
      // Every group this process belongs to maps its own rank back to the
      // same global rank, so attaching to several groups keeps writing the
      // same value and last-writer-wins is not a hazard. It only was while a
      // group-local rank could be fabricated above.
      auto global_rank = static_cast<int>(global_ranks[local_rank]);
      auto previous = recorder->getRank();
      if (previous >= 0 && previous != global_rank) {
        LOG(WARNING) << "FlightRecorderHook: global rank changed from "
                     << previous << " to " << global_rank
                     << "; dump files from this process will disagree.";
      }
      recorder->setRank(global_rank);
      recorder->record_pg_ranks(pg_name, global_ranks);
    }
    max_inflight_ = std::max(max_inflight_, recorder->max_entries_);
  }
}

const FlightRecorderHook::BackendTarget& FlightRecorderHook::targetFor(
    std::optional<c10::Device> device) const {
  if (device) {
    auto it = targets_.find(device->type());
    if (it != targets_.end()) {
      return it->second;
    }
  }
  // barrier() reaches the hook with no tensors, so the dispatcher's device is
  // not visible here.
  return default_target_;
}

std::optional<bool> FlightRecorderHook::captureActive(
    std::optional<c10::Device> device) const {
  if (device) {
    return streamIsCapturing(*device);
  }
  // barrier() is the one op that reaches the hook with no tensors: Ops.cpp
  // binds a dummy tensor to pick the dispatch key and does not forward it, so
  // the device it was dispatched on is invisible here. It did go to one of the
  // group's own backends though, and that is enough to answer this question --
  // if none of the devices the group serves is on a capturing stream, then
  // neither is this op, whichever of them it went to. For the usual
  // single-device group that is one query, on CUDA for a CUDA group.
  //
  // Not getBoundDeviceId(): it is only set when init_process_group was given a
  // device_id, so it answers nothing for the groups that hit this, and its
  // index names the bound device rather than the one the calling thread is
  // capturing on. The unset index used here resolves to the current device,
  // which is where the capture is.
  if (targets_.empty()) {
    return std::nullopt;
  }
  for (const auto& [type, target] : targets_) {
    if (streamIsCapturing(c10::Device(type))) {
      return true;
    }
  }
  return false;
}

FlightRecorderHook::~FlightRecorderHook() {
  remove();
}

void FlightRecorderHook::remove() {
  // Nothing that calls into the backend runs under mutex_, unregistration
  // included. A backend hook may be executing on another thread right now,
  // blocked on mutex_, while the backend holds the lock that guards its own
  // hook table -- so unregistering under mutex_ is a lock-order inversion and
  // deadlocks. Detaching is therefore two steps: publish the removal by
  // clearing pg_ under the lock, which is what the hooks check, then unregister
  // with nothing held.
  c10::intrusive_ptr<ProcessGroup> pg;
  // Destroyed after the lock is released, with the Works it holds.
  std::map<int64_t, InflightOp> abandoned;
  std::vector<c10::intrusive_ptr<Work>> released;
  bool had_abort_hook = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pg_) {
      return;
    }
    pg = std::move(pg_);
    // Ops still in flight are simply abandoned -- they are never retired, so a
    // dump keeps reporting them as issued and never seen to finish, which is
    // all we know.
    abandoned.swap(inflight_);
    released.swap(released_works_);
    had_abort_hook = std::exchange(abort_hook_registered_, false);
  }
  pg->unregisterPreHook(hook_id_);
  pg->unregisterPostHook(hook_id_);
  if (had_abort_hook) {
    pg->unregisterAbortHook(hook_id_);
  }
}

std::optional<FlightRecorderHook::InflightOp> FlightRecorderHook::
    takeInflightLocked(int64_t op_id) {
  auto it = inflight_.find(op_id);
  if (it == inflight_.end()) {
    // Evicted, or the hook was removed.
    return std::nullopt;
  }
  auto op = std::move(it->second);
  inflight_.erase(it);
  return op;
}

void FlightRecorderHook::onResult(int64_t op_id, bool success) {
  // Runs on whichever thread published the result: a backend thread, a caller
  // of the Work's wait() or isCompleted(), or onPost() itself if the op had
  // already finished. Only the claim happens under mutex_; the recorder call
  // does not, for the same reason onPre's does not -- see the lock-order note.
  std::optional<InflightOp> op;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    op = takeInflightLocked(op_id);
    if (op && success && op_id > pg_status_->lastCompletedSeq) {
      // Results can be published out of issue order (several streams, or a
      // caller's wait() ahead of the watchdog), so keep the latest.
      pg_status_->lastCompletedSeq = op_id;
      pg_status_->lastCompletedWorkName = std::string(hookOpName(op->name));
    }
  }
  if (!op) {
    return;
  }
  // A failed or timed-out op stays un-retired, so a dump keeps saying it was
  // never seen to finish.
  if (success) {
    op->recorder->retire_completed(
        op->trace_id.id,
        op->trace_id.reset_epoch,
        op->work ? workDuration(*op->work) : std::nullopt);
  }
  if (op->work) {
    // The caller may have dropped its handle, making this the last reference,
    // and this may be the backend's watchdog: destroying a Work's outputs there
    // can need the GIL or autograd state (stock ProcessGroupNCCL keeps outputs_
    // off its watchdog for the same reason). onPost releases it instead.
    std::lock_guard<std::mutex> lock(mutex_);
    released_works_.push_back(std::move(op->work));
  }
}

std::optional<float> FlightRecorderHook::workDuration(const Work& work) {
  if (!work_can_time_.load(std::memory_order_relaxed)) {
    return std::nullopt;
  }
  try {
    return work.getDuration();
  } catch (const std::exception&) {
    // Refused because the backend does not time collectives, which is a
    // property of the backend and not of the op -- so ask no further op.
    work_can_time_.store(false, std::memory_order_relaxed);
    return std::nullopt;
  }
}

void FlightRecorderHook::onPre(const PreHookArgs& args) {
  auto device = opDevice(args);
  const auto& target = targetFor(device);
  auto* recorder = target.recorder;
  if (recorder == nullptr) {
    // The backend serving this op already records it natively.
    return;
  }
  if (!recorder->enabled_) {
    return;
  }
  // Not recorded under an active graph capture, nor when the hook cannot
  // establish that there is none. The collective does not run here, it runs at
  // replay, and its Work cannot be polled: querying a CUDA event recorded on a
  // capturing stream does not merely fail, it invalidates the capture, which
  // then surfaces from cudaStreamEndCapture nowhere near this code. An entry we
  // could never observe would also read as a collective that never finished,
  // i.e. as a hang. Stock ProcessGroupNCCL skips recording under capture too
  // (initWork's record flag follows whether the work can be enqueued for the
  // watchdog).
  //
  // So "unknown" defaults to "capturing": recording costs the user's graph,
  // while not recording costs one entry in the trace. That default only bites a
  // group that reports no device types at all, which cannot serve a collective
  // through the c10d ops in the first place.
  if (captureActive(device).value_or(true)) {
    return;
  }
  InflightOp op;
  op.recorder = recorder;
  op.name = args.name;

  const bool is_p2p = isP2POp(args.name);
  size_t collective_seq = 0;
  size_t p2p_seq = 0;
  std::tuple<std::string, std::string> pg_name;
  int rank = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pg_) {
      // remove() ran while this hook call was blocked on mutex_.
      return;
    }
    collective_seq = is_p2p ? collective_seq_ : ++collective_seq_;
    p2p_seq = is_p2p ? ++p2p_seq_ : p2p_seq_;
    pg_status_->lastEnqueuedSeq = static_cast<int64_t>(args.op_id);
    pg_status_->lastEnqueuedWorkName = std::string(hookOpName(args.name));
    pg_name = std::make_tuple(pg_->getGroupName(), pg_->getGroupDesc());
    rank = pg_->getRank();
  }

  // Recording happens outside mutex_ because recordWithResetEnabled() gathers
  // a traceback, which takes the GIL, and remove() blocks on mutex_ with the
  // GIL held -- see the lock-order note on mutex_. The price is that two
  // threads issuing collectives on one group can reach the recorder in a
  // different order than their sequence numbers; the analyzer keys off
  // collective_seq_id, not record order.
  //
  // No events are published. The hook owns none: completion comes from the op's
  // Work in the post-hook, and an event the hook recorded on the caller's
  // stream would say nothing about a collective the backend runs on a stream of
  // its own.
  const bool mirror_inputs =
      args.output_tensors.empty() && recordsInPlace(args.name);
  op.trace_id = recorder->recordWithResetEnabled(
      pg_id_,
      pg_name,
      collective_seq,
      p2p_seq,
      static_cast<size_t>(args.op_id),
      profilingName(args, target.name, rank),
      args.input_tensors,
      mirror_inputs ? args.input_tensors : args.output_tensors,
      /*start=*/nullptr,
      /*end=*/nullptr,
      target.timeout,
      pg_status_,
      is_p2p);

  // Destroyed after the lock is released, with the Works they hold.
  std::vector<InflightOp> evicted;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pg_) {
    // remove() ran while this entry was being recorded. Nothing to wait on any
    // more; the entry stays un-retired.
    return;
  }
  inflight_[args.op_id] = op;
  // An op whose work never completes -- a real hang -- must not grow this
  // without bound, and there is nothing to gain from waiting on an op whose
  // entry the ring buffer has already overwritten (retire_completed would no-op
  // on it), so the buffer's own capacity is the bound.
  while (inflight_.size() > max_inflight_) {
    auto oldest = inflight_.begin();
    evicted.push_back(std::move(oldest->second));
    inflight_.erase(oldest);
  }
}

void FlightRecorderHook::onAbort() {
  static const bool dump_on_timeout =
      getCvarBool(TORCH_FR_DUMP_ON_TIMEOUT, true);
  if (!dump_on_timeout) {
    return;
  }
  // Only the default backend's instance reaches disk: ProcessGroup routes
  // registerAbortHook to the default backend, so that is the one whose failure
  // we are reacting to. Stock is the same shape -- the only DebugInfoWriter
  // caller is ProcessGroupNCCL::dumpDebuggingInfo -- and it keeps a trace from
  // one backend out of another backend's dump file.
  if (default_target_.recorder == nullptr) {
    return;
  }
  // At most one dump per process. A failure is observed by every process group
  // sharing the fabric, by the backend's watchdog and by the next synchronous
  // collective, and every dump targets the same file, so the later ones would
  // only overwrite the snapshot closest to the failure.
  //
  // The loser of the race waits on the mutex instead of returning right away:
  // the thread that detects the failure is usually a watchdog, while the thread
  // that runs ::abort() is the next synchronous collective on the main thread.
  // Letting that one run ahead would terminate the process with the trace half
  // written, which is exactly the post-mortem this exists to produce. The wait
  // is bounded -- the dump only queries events, it never synchronizes on the
  // device. Both statics live in libtorch_cpu, the single library that holds
  // the recorder, so there is one instance of them per process.
  static std::mutex dump_mutex;
  static bool dumped = false;
  std::lock_guard<std::mutex> lock(dump_mutex);
  if (dumped) {
    return;
  }
  // One shot whatever the outcome: retrying a dump that already failed once
  // would just repeat itself on every watchdog tick.
  dumped = true;
  // No mutex_ here. The abort hook fires from inside the backend, on a thread
  // that may hold backend locks, whereas onPre/onPost take mutex_ and then the
  // recorder's; keeping this path off mutex_ leaves that order intact.
  LOG(ERROR) << "FlightRecorderHook: dumping trace on collective failure";

  // Stack traces are most of a post-mortem's value -- without them every
  // culprit torchfrtrace reports has no code location -- but symbolizing one
  // may need the GIL, and this runs on a watchdog thread or on a rank about to
  // ::abort(), where blocking on the GIL would lose the very dump we are here
  // to write. So attempt it with traces under a bounded wait and retry without
  // them if that does not land in time. Stock ProcessGroupNCCL's heartbeat
  // monitor does exactly this, and reads the same two env vars.
  bool include_stack_traces = getCvarBool(TORCH_INCLUDE_STACK_TRACE, true);
  const auto only_active = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
  const std::chrono::milliseconds wait{
      getCvarInt(TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC, 15 * 1000)};
  // ~future joins, so an attempt that ran out of time may not be destroyed
  // here: that would restore the unbounded wait this exists to avoid. Park it
  // in a leaked list instead. At most two per process, since onAbort is
  // one-shot.
  static auto* abandoned = new std::vector<std::future<void>>();
  while (true) {
    std::future<void> dump;
    try {
      dump = std::async(
          std::launch::async,
          [include_stack_traces,
           only_active,
           backend = default_target_.name]() {
            try {
              if (!try_dump_fr_trace_file(
                      /*includeCollectives=*/true,
                      include_stack_traces,
                      only_active,
                      backend)) {
                // Recorder off, or no rank was ever set, which means nothing
                // was recorded either.
                LOG(ERROR) << "FlightRecorderHook: no trace to dump.";
              }
            } catch (const std::exception& e) {
              LOG(ERROR) << "FlightRecorderHook: trace dump failed: "
                         << e.what();
            } catch (...) {
              LOG(ERROR) << "FlightRecorderHook: trace dump failed.";
            }
          });
    } catch (const std::exception& e) {
      LOG(ERROR) << "FlightRecorderHook: cannot start trace dump: " << e.what();
      return;
    }
    if (dump.wait_for(wait) == std::future_status::ready) {
      return;
    }
    abandoned->push_back(std::move(dump));
    if (!include_stack_traces) {
      LOG(ERROR) << "FlightRecorderHook: trace dump did not finish within "
                 << wait.count() << " ms, giving up.";
      return;
    }
    LOG(ERROR) << "FlightRecorderHook: trace dump did not finish within "
               << wait.count()
               << " ms, retrying without stack traces. Set "
                  "TORCH_INCLUDE_STACK_TRACE=0 to skip the first attempt.";
    include_stack_traces = false;
  }
}

void FlightRecorderHook::onPost(const PostHookArgs& args) {
  bool has_result_future = true;
  // Destroyed after the lock is released; see released_works_.
  std::vector<c10::intrusive_ptr<Work>> released;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    released.swap(released_works_);
    auto it = inflight_.find(args.op_id);
    if (it == inflight_.end()) {
      // Nothing was recorded for this op -- the backend records natively, the
      // op was issued under a graph capture, or the hook was removed in
      // between.
      return;
    }
    if (!args.work) {
      // Nothing was issued that we could point at (the hook contract allows a
      // null work), so the entry stays un-retired: the same honest report as
      // an op whose backend threw before the post-hook.
      inflight_.erase(it);
      return;
    }
    // The op is only *issued* at this point, so the entry stays un-retired
    // until the Work's result says it finished. The Work is kept only to read
    // its duration then.
    if (work_can_time_.load(std::memory_order_relaxed)) {
      it->second.work = args.work;
    }
    const Work& work = *args.work;
    has_result_future = !no_result_future_types_.count(typeid(work));
  }
  // With the lock dropped: both calls go into the backend, and the callback
  // runs inline if the op has already finished -- see the lock-order note.
  c10::intrusive_ptr<c10::ivalue::Future> result;
  if (has_result_future) {
    try {
      result = args.work->getFutureResult();
    } catch (const c10::NotImplementedError& e) {
      // This Work type reports no result (e.g. ucc, mpi, nccl2's coalesced p2p
      // sentinel). Not a warning: the default nccl2 path hits it.
      const Work& work = *args.work;
      VLOG(1) << "FlightRecorderHook: " << c10::demangle(typeid(work).name())
              << " has no result future, so its ops retire at issue: "
              << e.what();
      std::lock_guard<std::mutex> lock(mutex_);
      no_result_future_types_.insert(typeid(work));
    } catch (const std::exception& e) {
      // Unexpected: retire this op at issue rather than fail the collective.
      LOG(WARNING) << "FlightRecorderHook: getFutureResult failed for op "
                   << args.op_id << ": " << e.what();
    }
  }
  if (!result) {
    // Retired but not completed: nothing will ever tell this hook the op
    // finished, and an entry nothing retires reads as a hang, so retire it here
    // and leave it saying only what is known -- it was issued. No duration, and
    // "scheduled" rather than "completed".
    std::optional<InflightOp> op;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      op = takeInflightLocked(args.op_id);
    }
    if (op) {
      op->recorder->retire_id(
          op->trace_id.id,
          op->trace_id.reset_epoch,
          /*compute_duration=*/false);
    }
    return;
  }
  result->addCallback([weak = weak_from_this(),
                       op_id = args.op_id](c10::ivalue::Future& future) {
    // Not the last reference: a hook is only destroyed with no ops in flight.
    auto self = weak.lock();
    if (!self) {
      return;
    }
    // An exception would escape into whoever completed the future, usually
    // the backend's watchdog.
    try {
      // A result that is not a WorkResult still says the op finished.
      self->onResult(
          op_id,
          !future.hasError() &&
              (!future.value().isInt() ||
               future.value().toInt() ==
                   static_cast<int64_t>(WorkResult::SUCCESS)));
    } catch (const std::exception& e) {
      LOG(ERROR) << "FlightRecorderHook: cannot retire op " << op_id << ": "
                 << e.what();
    }
  });
}

} // namespace c10d
