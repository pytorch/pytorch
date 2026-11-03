// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/DesyncDebugHook.hpp>

#include <atomic>
#include <utility>

#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/hooks/HookUtils.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>

namespace c10d {

namespace {

std::atomic<int64_t> next_hook_id{0x4453594e /* 'DSYN' */};

// Prefix of the trace keys. The group's store is already scoped to the group,
// and ProcessGroupNCCL writes its own keys under the backend's "<device>/"
// prefix, so these collide with nothing.
constexpr const char* kTracePrefix = "c10d_desync";

// Ops a completion may still be awaited for. A failed op never completes, so
// this bounds the map; a completion older than this many ops is long stale.
constexpr size_t kMaxTracked = 1 << 14;

constexpr std::chrono::milliseconds kAbortReportTimeout{60 * 1000};

} // namespace

std::shared_ptr<DesyncDebugHook> DesyncDebugHook::attach(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    std::chrono::milliseconds flush_interval) {
  auto hook =
      std::shared_ptr<DesyncDebugHook>(new DesyncDebugHook(pg, flush_interval));
  // The lambdas own the hook, so it lives as long as the group (and its default
  // backend) does. Completion and abort hooks first: the post-hook reads
  // completion_hook_registered_, which must not change once ops can flow.
  if (pg->supportsCompletionHooks()) {
    pg->registerCompletionHook(
        hook->hook_id_, [hook](const CompletionHookArgs& args) {
          hook->onCompletion(args.completionKey);
        });
    hook->completion_hook_registered_ = true;
    std::lock_guard<std::mutex> lock(hook->state_->mutex);
    hook->state_->tracks_completion = true;
  }
  if (pg->supportsAbortHooks()) {
    pg->registerAbortHook(hook->hook_id_, [hook]() { hook->onAbort(); });
    hook->abort_hook_registered_ = true;
  }
  pg->registerPreHook(
      hook->hook_id_, [hook](const PreHookArgs& args) { hook->onPre(args); });
  pg->registerPostHook(
      hook->hook_id_, [hook](const PostHookArgs& args) { hook->onPost(args); });
  return hook;
}

DesyncDebugHook::DesyncDebugHook(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    std::chrono::milliseconds flush_interval)
    : pg_(pg), hook_id_(next_hook_id++) {
  TORCH_CHECK(pg, "DesyncDebugHook: null process group");
  auto store = pg->getStore();
  TORCH_CHECK(store, "DesyncDebugHook: process group has no store");
  for (const auto& device : pg->getDeviceTypes()) {
    device_types_.push_back(device.type());
  }
  state_ = std::make_shared<State>();
  state_->store = std::move(store);
  state_->rank = pg->getRank();
  state_->size = pg->getSize();
  state_->start_key = getTraceStartKey(kTracePrefix, state_->rank);
  state_->end_key = getTraceEndKey(kTracePrefix, state_->rank);
  state_->flush_interval = flush_interval;
  std::string group_name;
  try {
    group_name = pg->getGroupName();
  } catch (const std::exception&) {
    // A group with no backend registered has no name; only the log suffers.
  }
  state_->log_prefix =
      c10::str("[PG ", group_name, " Rank ", state_->rank, "] ");
  writer_ = std::thread([state = state_]() { writerLoop(state); });
}

DesyncDebugHook::~DesyncDebugHook() {
  stop();
}

void DesyncDebugHook::stop() {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->stopping = true;
  }
  state_->cv.notify_all();
  state_->served_cv.notify_all();
  // Not joined: the writer may be blocked in a store call against a store that
  // is gone (rank 0 exits first at the end of a job), and waiting for it here
  // would hang destroy_process_group for the store timeout. It owns its state
  // and exits once the call returns.
  if (writer_.joinable()) {
    writer_.detach();
  }
}

void DesyncDebugHook::remove() {
  // Unregistering drops the group's references, which are normally the only
  // ones keeping this hook alive.
  auto keepalive = weak_from_this().lock();
  // No hook lock is held while calling into the group: a backend may hold its
  // hook table lock while running our completion hook, which takes ours.
  if (auto pg = pg_.lock()) {
    pg->unregisterPreHook(hook_id_);
    pg->unregisterPostHook(hook_id_);
    if (completion_hook_registered_) {
      pg->unregisterCompletionHook(hook_id_);
    }
    if (abort_hook_registered_) {
      pg->unregisterAbortHook(hook_id_);
    }
  }
  pg_.reset();
  stop();
}

bool DesyncDebugHook::captureActive(std::optional<c10::Device> device) const {
  if (device) {
    return detail::streamIsCapturing(*device);
  }
  // barrier() reaches the hooks with no tensors; it went to one of the group's
  // devices, so it is captured only if one of them is capturing.
  for (auto type : device_types_) {
    if (detail::streamIsCapturing(c10::Device(type))) {
      return true;
    }
  }
  return false;
}

void DesyncDebugHook::onPre(const PreHookArgs& args) {
  // Only collectives are numbered, as in stock's seqCollective_: ranks issue
  // different numbers of p2p ops (pipeline stages), and p2p inside a coalescing
  // window returns a placeholder Work that is already complete.
  if (args.name == HookOpName::SEND || args.name == HookOpName::RECV) {
    return;
  }
  // Not counted under capture: the op runs at replay, and polling its Work in
  // the post-hook would invalidate the capture.
  if (captureActive(detail::opDevice(args))) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->stopping || state_->failed) {
    return;
  }
  Trace trace{++next_seq_, args.name};
  state_->start = trace;
  issued_[args.op_id] = trace;
  // A backend that throws between the pre- and the post-hook leaves its entry
  // behind; op ids are monotonic, so drop the oldest.
  while (issued_.size() > kMaxTracked) {
    issued_.erase(issued_.begin());
  }
  // Only the transition wakes the writer, so a burst of ops costs one wakeup.
  if (!std::exchange(state_->dirty, true)) {
    state_->cv.notify_one();
  }
}

void DesyncDebugHook::onPost(const PostHookArgs& args) {
  uint64_t completion_key = 0;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto it = issued_.find(args.op_id);
    if (it == issued_.end()) {
      // Issued under capture, or before attach / after stop.
      return;
    }
    auto trace = it->second;
    issued_.erase(it);
    if (!args.work || !completion_hook_registered_) {
      return;
    }
    completion_key = args.work->getCompletionKey();
    pending_[completion_key] = trace;
    completion_order_.push_back(completion_key);
    while (completion_order_.size() > kMaxTracked) {
      pending_.erase(completion_order_.front());
      completion_order_.pop_front();
    }
  }
  // A completion pushed before the key was registered above found nothing, so
  // ask the Work. Outside the lock: this calls into the backend.
  if (args.work->isCompleted() && args.work->isSuccess()) {
    onCompletion(completion_key);
  }
}

void DesyncDebugHook::onCompletion(uint64_t completion_key) {
  // May run on the backend's watchdog with backend locks held: map lookup only.
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto it = pending_.find(completion_key);
  if (it == pending_.end()) {
    return;
  }
  auto trace = it->second;
  pending_.erase(it);
  if (trace.seq <= state_->end.seq || state_->stopping || state_->failed) {
    return;
  }
  state_->end = trace;
  if (!std::exchange(state_->dirty, true)) {
    state_->cv.notify_one();
  }
}

bool DesyncDebugHook::waitReport(
    std::unique_lock<std::mutex>& lock,
    uint64_t id,
    std::chrono::milliseconds timeout) {
  state_->served_cv.wait_for(lock, timeout, [&]() {
    return state_->report_served >= id || state_->stopping;
  });
  return state_->report_served >= id;
}

std::string DesyncDebugHook::report(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  TORCH_CHECK(!state_->stopping, "DesyncDebugHook: hook was removed");
  auto id = ++state_->report_requested;
  state_->cv.notify_one();
  TORCH_CHECK(
      waitReport(lock, id, timeout),
      "DesyncDebugHook: report not ready within ",
      timeout.count(),
      " ms");
  TORCH_CHECK(state_->report_error.empty(), state_->report_error);
  return state_->report;
}

void DesyncDebugHook::onAbort() {
  // Abort hooks may run several times per failure, on different threads. The
  // first call requests the report and logs it; later ones block until it is
  // done rather than return, so a thread about to abort the process does not
  // cut the report short.
  std::string report;
  std::string error;
  {
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->stopping) {
      return;
    }
    if (abort_report_ != 0) {
      // Waits for the first call to finish, not for a fresh timeout: the
      // backend may call this every watchdog iteration.
      state_->served_cv.wait(
          lock, [&]() { return abort_done_ || state_->stopping; });
      return;
    }
    abort_report_ = ++state_->report_requested;
    state_->cv.notify_one();
    const bool served = waitReport(lock, abort_report_, kAbortReportTimeout);
    abort_done_ = true;
    state_->served_cv.notify_all();
    if (!served) {
      error = "report not ready in time";
    } else {
      report = state_->report;
      error = state_->report_error;
    }
  }

  ::c10d::C10dLoggingData log;
  log.integers["rank"] = state_->rank;
  log.integers["world_size"] = state_->size;
  // Distinguishes desync reports from flight recorder dumps, as stock does.
  log.strings["flight_recorder_version"] = "-1";
  if (error.empty()) {
    log.strings["status"] = "SUCCESS";
    LOG(ERROR) << state_->log_prefix << report;
  } else {
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = error;
    LOG(ERROR) << state_->log_prefix
               << "Failed to retrieve desync debug report: " << error;
  }
  if (auto logger = C10dLogger::getLogger()) {
    logger->log(log);
  }
}

void DesyncDebugHook::writerLoop(const std::shared_ptr<State>& state) {
  c10::setThreadName("pt_desync_dbg");
  // What the store holds for this rank, so an unchanged key is not rewritten.
  uint64_t written_start = 0;
  uint64_t written_end = 0;
  std::unique_lock<std::mutex> lock(state->mutex);
  while (true) {
    state->cv.wait(lock, [&]() {
      return state->stopping || state->dirty ||
          state->report_requested > state->report_served;
    });
    if (state->stopping) {
      return;
    }
    const auto start = state->start;
    const auto end = state->end;
    state->dirty = false;
    const auto report_id = state->report_requested;
    const bool report_pending = report_id > state->report_served;
    const bool tracks_completion = state->tracks_completion;
    bool ok = !state->failed;
    lock.unlock();

    // Store I/O with no lock held. start before end: a reader that sees the new
    // start with the old end reads "joined, not finished", which is true, while
    // the reverse order would briefly hide a finished op.
    if (ok && start.seq != written_start) {
      ok = traceUpdate(
          state->store,
          state->start_key,
          start.seq,
          std::string(detail::hookOpName(start.op)));
      written_start = start.seq;
    }
    if (ok && end.seq != written_end) {
      ok = traceUpdate(
          state->store,
          state->end_key,
          end.seq,
          std::string(detail::hookOpName(end.op)));
      written_end = end.seq;
    }
    std::string report;
    std::string error;
    if (report_pending) {
      if (!ok) {
        error = "store unavailable";
      } else if (
          start.seq == 0 || (tracks_completion && start.seq <= end.seq)) {
        // retrieveDesyncReport asserts this rank is inside a collective.
        error = "no collective pending on this rank";
      } else {
        try {
          report = retrieveDesyncReport(
              state->store, kTracePrefix, state->rank, state->size);
        } catch (const std::exception& e) {
          error = e.what();
        }
        // It returns nothing when this rank's own trace failed to parse.
        if (error.empty() && report.empty()) {
          error = "store returned no trace for this rank";
        }
      }
    }

    lock.lock();
    if (!ok && !state->failed) {
      state->failed = true;
      LOG(ERROR) << state->log_prefix
                 << "DesyncDebugHook: store write failed, disabling.";
    }
    if (report_pending) {
      state->report_served = report_id;
      state->report = std::move(report);
      state->report_error = std::move(error);
      state->served_cv.notify_all();
    }
    // Bounds store load: at most one flush per interval however fast ops are
    // issued. A report request still gets through right away.
    state->cv.wait_for(lock, state->flush_interval, [&]() {
      return state->stopping || state->report_requested > state->report_served;
    });
  }
}

} // namespace c10d
