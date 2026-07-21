// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// FlightRecorderHook: backend-agnostic FlightRecorder integration built on the
// ProcessGroup pre/post collective hooks (Hooks.hpp). Port of torchcomms'
// hooks/fr FlightRecorderHook onto c10d.
//
// The pre-hook records an entry into the generic FlightRecorder<c10::Event>
// ring buffer (null start/end events, but full op/tensor/sequencing metadata).
// The post-hook associates the entry with the returned Work so it can be
// retired when the operation completes. Because the hooks fire from the
// dispatcher kernels in Ops.cpp, this works for any backend routed through
// c10d ops.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

class TORCH_API FlightRecorderHook
    : public std::enable_shared_from_this<FlightRecorderHook> {
 public:
  // Returns the process group's automatically installed hook, or attaches a
  // caller-owned hook when automatic recording is disabled.
  static std::shared_ptr<FlightRecorderHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg);

  static bool isEnabled();

  ~FlightRecorderHook();

  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  // Detach from the process group. Idempotent.
  void remove();

 private:
  friend class ProcessGroup;

  explicit FlightRecorderHook(
      ProcessGroup* pg,
      c10::intrusive_ptr<ProcessGroup> pg_keepalive = nullptr);
  static std::shared_ptr<FlightRecorderHook> attachOwned(ProcessGroup* pg);

  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args);
  std::string backendName(const PreHookArgs& args) const;

  ProcessGroup* pg_;
  c10::intrusive_ptr<ProcessGroup> pg_keepalive_;
  int64_t hook_id_;
  size_t pg_id_;
  std::shared_ptr<ProcessGroupStatus> pg_status_;
  std::chrono::milliseconds timeout_{kBackendDefaultTimeout};

  // Sequencing and the op_id -> trace-entry correlation map. The mutex guards
  // against concurrent collectives from multiple threads (the hooks fire on
  // the issuing thread).
  std::mutex mutex_;
  size_t collective_seq_{0};
  size_t p2p_seq_{0};
  struct InflightTrace {
    FlightRecorder<c10::Event>::TraceIdentifier id;
    int64_t sequence;
    std::string name;
    bool track_completion;
  };
  std::unordered_map<int64_t, InflightTrace> inflight_;
};

} // namespace c10d
