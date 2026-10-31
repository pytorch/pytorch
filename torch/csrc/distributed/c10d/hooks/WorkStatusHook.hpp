// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// WorkStatusHook: backend-agnostic port of nccl-legacy's periodic work status
// C10dLogger record, with its keys, built on the ProcessGroup collective hooks.
// One process-wide thread logs every attached group's record every
// TORCH_DIST_WORK_STATUS_PERIOD_MS (default 30000, read once per process)
// while a logger is registered. Without one, ops are only numbered.
//
// Completion is learnt from the completion hook of the backend that ran the op
// (nccl2, nccl-lazy), else for CPU ops from the Work's future (gloo). Only
// success counts. Other ops (gloo's send/recv and reduce_scatter, nccl2's
// batched p2p, device ops of backends without completion hooks, barrier unless
// on CPU or a pushing backend, ops whose backend call throws) are only passed
// over by later completions. Like legacy, ops under CUDA graph capture are
// numbered but not enqueued, and the started fields keep the untimed defaults.
// Groups served by nccl-legacy, which reports natively, are not attached.

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
  // Attaches a hook, or returns null if the group reports natively or
  // reporting is disabled. Detaches on remove() or destruction. global_ranks
  // maps group ranks to world ranks; empty uses the backend's
  // Options::global_ranks_in_group.
  static std::shared_ptr<WorkStatusHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg,
      std::vector<uint64_t> global_ranks = {});

  // Logs every attached group's record now, as the thread does. For tests.
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

 private:
  // Op tracking, held weakly by the hook callbacks.
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
