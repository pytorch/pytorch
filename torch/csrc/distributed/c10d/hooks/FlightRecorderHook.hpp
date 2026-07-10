// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// FlightRecorderHook: backend-agnostic FlightRecorder integration built on the
// ProcessGroup pre/post collective hooks (Hooks.hpp). Port of torchcomms'
// hooks/fr FlightRecorderHook onto c10d.
//
// The pre-hook records an entry into the generic FlightRecorder<c10::Event>
// ring buffer (null start/end events, like ProcessGroupGloo's built-in
// recording -- no GPU duration, but full op/tensor/sequencing metadata); the
// post-hook retires it. Because the hooks fire from the dispatcher kernels in
// Ops.cpp, this works for any backend routed through c10d ops -- including
// backends with no native FlightRecorder support (nccl2, custom backends) --
// and the traces are dumped with the existing _dump_fr_trace{,_json} APIs.

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
  // Attaches a hook to the process group and returns it. The hook stays
  // attached until remove() is called or the returned handle is destroyed.
  static std::shared_ptr<FlightRecorderHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg);

  ~FlightRecorderHook();

  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  // Detach from the process group. Idempotent.
  void remove();

 private:
  explicit FlightRecorderHook(c10::intrusive_ptr<ProcessGroup> pg);
  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args);

  c10::intrusive_ptr<ProcessGroup> pg_;
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
  std::unordered_map<int64_t, FlightRecorder<c10::Event>::TraceIdentifier>
      inflight_;
};

} // namespace c10d
