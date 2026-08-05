// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// FlightRecorderHook: backend-agnostic FlightRecorder integration built on the
// ProcessGroup pre/post collective hooks (Hooks.hpp). Port of torchcomms'
// hooks/fr FlightRecorderHook onto c10d.
//
// The pre-hook records an entry into the generic FlightRecorder<c10::Event>
// ring buffer along with a start/end device event pair owned by the hook; the
// post-hook records the end event and retires the entry. The events are what
// let a dump report whether an op actually started or completed on the
// device, which is the interesting case when an op is still in flight (a
// hang). A device-measured duration is only available when the end event is
// already complete at retire time; since the post-hook fires as soon as the op
// is issued, that is rare, so duration_ms usually falls back to host wall
// clock since the entry was created -- an upper bound, not kernel time.
// CPU-only ops have no device to record on and keep null start/end events,
// like ProcessGroupGloo's built-in recording, so they always take that
// fallback. Because the hooks fire from the dispatcher kernels in
// Ops.cpp, this works for any backend routed through c10d ops -- including
// backends with no native FlightRecorder support (nccl2, custom backends) --
// and the traces are dumped with the existing
// _dump_fr_trace{,_json,_file} APIs or the "fr_dump_file" control plane
// handler. attach() also tells the recorder this process's global rank, which
// is what names the per-rank dump file.

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <c10/core/Event.h>
#include <c10/core/Stream.h>
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
  // A recorded op and the device events the hook owns on its behalf. The
  // recorder's entry only borrows raw pointers to the events, so they must
  // stay alive until retire_id() has cleared those pointers.
  struct InflightOp {
    FlightRecorder<c10::Event>::TraceIdentifier trace_id;
    std::unique_ptr<c10::Event> start;
    std::unique_ptr<c10::Event> end;
    std::optional<c10::Stream> stream;
  };

  explicit FlightRecorderHook(c10::intrusive_ptr<ProcessGroup> pg);
  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args);
  // Retires op's entry, optionally recording its end event first. Caller must
  // hold mutex_ and must not free op's events before this returns.
  void retire(InflightOp& op, bool record_end);

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
  std::unordered_map<int64_t, InflightOp> inflight_;
};

} // namespace c10d
