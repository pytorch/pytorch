// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// WorkStatusHook: backend-agnostic port of stock ProcessGroupNCCL's periodic
// work status report, built on the ProcessGroup pre/post collective hooks
// (Hooks.hpp).
//
// Stock ProcessGroupNCCL's watchdog logs, every 30s, a C10dLogger record with
// the group's last enqueued / started / completed work (sequence number, op
// name and numels) and an INFO line "NCCL Work update periodically: ...".
// Other backends report nothing. This hook tracks the same status from the
// dispatcher hooks and emits the same record, with the same keys, from one
// process-wide thread for every attached group.
//
// Completion is learnt from the completion hook of the backend that ran the op
// where it has one (nccl2, nccl-lazy), otherwise, for CPU ops, from a callback
// on the Work's future (gloo). Only successful completion counts. Any other op
// (gloo's send/recv and reduce_scatter, nccl2's send/recv in a coalescing
// batch, device ops of backends without completion hooks, barrier in a group
// with several backends) is never reported completed itself, only passed over
// once a later op completes. Like stock, ops issued under CUDA graph capture
// are numbered but neither enqueued nor completed. The started fields keep
// their defaults: no generic hook observes a collective starting on the device,
// and stock only fills them in with TORCH_NCCL_ENABLE_TIMING.
//
// Ops are numbered regardless, but nothing else is tracked or logged unless a
// C10dLogger is registered. The INFO line is then also logged where glog has
// LOG_EVERY_MS, as stock only then does.
//
// attach() returns null, and nothing is attached, for a group served by stock
// ProcessGroupNCCL, which reports its own status, or when the period
// (TORCH_DIST_WORK_STATUS_PERIOD_MS, default 30000) is 0.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>

namespace c10d {

class TORCH_API WorkStatusHook {
 public:
  // Attaches a hook to the process group and returns it, or null if the group
  // reports its status natively or reporting is disabled. The hook stays
  // attached until remove() is called or the returned handle is destroyed.
  //
  // global_ranks maps group ranks to world ranks and fills the global_rank
  // key. Empty means the backend's Options::global_ranks_in_group is used, or
  // -1 if it has none.
  static std::shared_ptr<WorkStatusHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg,
      std::vector<uint64_t> global_ranks = {});

  // Synchronously emits the status of every attached group, as the periodic
  // thread does. For tests.
  static void logAll();

  ~WorkStatusHook();

  WorkStatusHook(const WorkStatusHook&) = delete;
  WorkStatusHook(WorkStatusHook&&) = delete;
  WorkStatusHook& operator=(const WorkStatusHook&) = delete;
  WorkStatusHook& operator=(WorkStatusHook&&) = delete;

  // Detach from the process group and stop reporting it. Idempotent.
  void remove();

  // The group's current record, with the keys stock ProcessGroupNCCL logs.
  C10dLoggingData status();

  // Stock ProcessGroupNCCL's log prefix for the group.
  std::string logPrefix() const;

 private:
  // Op tracking, shared with the hook callbacks. They hold it weakly so that
  // a backend thread never holds the last reference to the hook, whose
  // destruction releases the group.
  struct State;

  WorkStatusHook(
      c10::intrusive_ptr<ProcessGroup> pg,
      std::vector<uint64_t> global_ranks);

  // Null once removed. Guarded by the state's mutex.
  c10::intrusive_ptr<ProcessGroup> pg_;
  int64_t hook_id_;
  // Identity of the group, fixed at attach.
  size_t pg_id_;
  int rank_;
  int64_t global_rank_{-1};
  std::string pg_name_;
  std::string pg_desc_;
  bool push_completion_{false};
  std::shared_ptr<State> state_;
};

} // namespace c10d
