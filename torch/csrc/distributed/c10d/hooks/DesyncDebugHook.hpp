// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// DesyncDebugHook: backend-agnostic port of ProcessGroupNCCL's
// TORCH_NCCL_DESYNC_DEBUG, built on the ProcessGroup hooks (Hooks.hpp).
//
// Every rank publishes the last op it issued ("start") and the last op it saw
// complete ("end") to the group's store. When a collective fails, the abort
// hook reads every rank's pair back and logs the stock desync report
// (retrieveDesyncReport): which ranks never joined, which joined but did not
// finish, which finished but did not join the next op.
//
// Stock does the store writes from its watchdog, two blocking sets per work
// while holding the lock the issuing thread needs to enqueue the next work,
// which stalls training on store round trips. Here no store I/O happens on any
// thread a collective or a backend depends on: the pre/post/completion hooks
// only update an in-memory (seq, op) pair under a hook-local mutex, and a
// dedicated writer thread publishes it. The writer coalesces -- however many
// ops were issued since the last flush, it writes at most the two keys, at
// most once per flush interval -- so store load is bounded per rank rather
// than per op.
//
// seq counts the collectives issued through the group since the hook was
// attached, from 1, so it is comparable across ranks as long as every rank
// attaches at the same point in the group's collective sequence
// (distributed_c10d attaches at group creation). p2p ops are not tracked. Ops
// issued under graph capture are not counted: they do not run until replay, and
// their Work cannot be polled.
//
// "end" needs completion hooks. On backends without them (gloo) only "start"
// is published, which still points at the rank that is behind, but cannot tell
// a rank that finished op N from one stuck in it.
//
// The report is logged by the abort hook on backends that have one (nccl2),
// once per hook; report() builds it on demand on any backend.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API DesyncDebugHook
    : public std::enable_shared_from_this<DesyncDebugHook> {
 public:
  // Attaches the hook to the process group, which owns it from then on, like
  // NanCheckHook. Publishes to the group's store.
  static std::shared_ptr<DesyncDebugHook> attach(
      const c10::intrusive_ptr<ProcessGroup>& pg,
      std::chrono::milliseconds flush_interval =
          std::chrono::milliseconds(100));

  ~DesyncDebugHook();

  DesyncDebugHook(const DesyncDebugHook&) = delete;
  DesyncDebugHook(DesyncDebugHook&&) = delete;
  DesyncDebugHook& operator=(const DesyncDebugHook&) = delete;
  DesyncDebugHook& operator=(DesyncDebugHook&&) = delete;

  // Detach from the process group. Idempotent, and a no-op once the group is
  // gone.
  void remove();

  // Publishes this rank's latest state, then builds the desync report from
  // every rank's. Runs on the writer thread; waits at most `timeout` for it.
  // Throws if the report cannot be built (store failure, timeout, or no op
  // pending on this rank).
  std::string report(
      std::chrono::milliseconds timeout = std::chrono::seconds(60));

 private:
  // An enum rather than a name so the hooks never allocate.
  struct Trace {
    uint64_t seq = 0;
    HookOpName op = HookOpName::UNKNOWN;
  };

  // Everything the writer thread touches. Shared with the thread so that
  // destroying the hook never has to join a thread that may be blocked in a
  // store call against a dead store; the thread exits on its own.
  struct State {
    c10::intrusive_ptr<Store> store;
    std::string start_key;
    std::string end_key;
    std::string log_prefix;
    int rank = 0;
    int size = 0;
    std::chrono::milliseconds flush_interval;

    // Guards everything below. Never held across store I/O or a backend call.
    std::mutex mutex;
    // Wakes the writer.
    std::condition_variable cv;
    // Wakes report() callers.
    std::condition_variable served_cv;
    bool stopping = false;
    bool dirty = false;
    Trace start;
    Trace end;
    // Whether end is published; set once at attach.
    bool tracks_completion = false;
    // Store failures disable publishing for good, like stock.
    bool failed = false;
    // report() requests are numbered; the writer serves the latest one and
    // marks every earlier one served with it.
    uint64_t report_requested = 0;
    uint64_t report_served = 0;
    std::string report;
    std::string report_error;
  };

  DesyncDebugHook(
      const c10::intrusive_ptr<ProcessGroup>& pg,
      std::chrono::milliseconds flush_interval);
  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args);
  void onCompletion(uint64_t completion_key);
  void onAbort();
  // Stops the writer thread without waiting for it.
  void stop();
  // Requests a report and waits for it under the held lock. False on timeout or
  // if the hook stopped first.
  bool waitReport(
      std::unique_lock<std::mutex>& lock,
      uint64_t id,
      std::chrono::milliseconds timeout);
  bool captureActive(std::optional<c10::Device> device) const;
  static void writerLoop(const std::shared_ptr<State>& state);

  // Weak: the group owns the hook.
  c10::weak_intrusive_ptr<ProcessGroup> pg_;
  int64_t hook_id_;
  std::vector<c10::DeviceType> device_types_;
  bool completion_hook_registered_{false};
  bool abort_hook_registered_{false};
  std::shared_ptr<State> state_;
  std::thread writer_;

  // Hook-side bookkeeping, guarded by state_->mutex.
  uint64_t next_seq_{0};
  // op_id -> seq between the pre- and the post-hook.
  std::map<int64_t, Trace> issued_;
  // completion key -> seq until the completion hook fires, bounded by
  // completion_order_.
  std::unordered_map<uint64_t, Trace> pending_;
  std::deque<uint64_t> completion_order_;
  // The abort hook's report, requested once: a hook reports one failure.
  uint64_t abort_report_{0};
  bool abort_done_{false};
};

} // namespace c10d
