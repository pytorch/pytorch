// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// FlightRecorderHook: backend-agnostic FlightRecorder integration built on the
// ProcessGroup pre/post collective hooks (Hooks.hpp). Port of torchcomms'
// hooks/fr FlightRecorderHook onto c10d.
//
// The pre-hook records an entry into the serving backend's own
// FlightRecorder<c10::Event> instance (getFlightRecorder, keyed by backend
// name). The post-hook does not retire it: it fires when the collective is
// *issued*, not when it finishes. The entry is retired from a callback on the
// Work's result future (Work::getFutureResult), which the backend completes
// where it establishes that the op really finished; the duration is the
// Work's own measurement. That is stock ProcessGroupNCCL's model -- its
// watchdog retires on real completion -- and it is what lets a dump tell a
// finished collective from a hung one: a collective that never completes is
// never retired and reads "scheduled", one that finished reads "completed". A
// hang needs no observer, since work that never completes is simply never
// reported.
//
// The post-hook is the only place a Work and its op_id are seen together, so
// it is where the callback is added. A future that already completed runs it
// at once, so a fast collective that finished before the post-hook is retired
// too.
//
// A Work with no result future (hasFutureResult() is false) falls back to
// retiring in the post-hook, i.e. at issue. Such an entry carries no duration
// and never reads "completed", only "scheduled" and retired -- degraded, but
// honest, because there is genuinely no moment at which this hook learns the op
// finished. Leaving those entries un-retired instead would make every healthy
// op look hung.
//
// The hook owns no device events. One it recorded itself could only say whether
// the *caller's* stream reached the op, which for an async_op collective is not
// even the stream the collective runs on (nccl2 uses its internal stream), it
// can never observe completion, and querying it is illegal under CUDA graph
// capture. Ops issued while a capture is active are not recorded at all,
// matching stock ProcessGroupNCCL, because their Work cannot be polled either.
//
// CPU-only ops are recorded the same way; whether they can report completion or
// a duration is up to the backend's Work.
//
// Because the hooks fire from the dispatcher kernels in Ops.cpp, this works for
// any backend routed through c10d ops -- including backends with no native
// FlightRecorder support (nccl2, gloo, custom backends) -- and the traces are
// dumped with the existing _dump_fr_trace{,_json,_file} APIs or the
// "fr_dump_file" control plane handler, naming the backend whose instance to
// read. attach() also tells the recorder this process's global rank, which is
// what names the per-rank dump file.
//
// Ops served by a backend with its own FlightRecorder are skipped, so the hook
// can be attached to a group of any composition without duplicating what a
// native backend already recorded.
//
// attach() registers an abort hook on backends that support one. The abort hook
// writes the trace to disk when the backend detects a timeout or an error, and
// is the single trigger for a dump-on-failure: backends do not dump themselves,
// they only run their abort hooks.

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <c10/core/Event.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

class TORCH_API FlightRecorderHook
    : public std::enable_shared_from_this<FlightRecorderHook> {
 public:
  // Attaches a hook to the process group and returns it. The hook stays
  // attached until remove() is called or the returned handle is destroyed.
  //
  // global_ranks is the group's members' ranks in the world, indexed by group
  // rank; it names the dump file and is published as the group's membership.
  // Empty means the caller does not know, in which case the backend's
  // Options::global_ranks_in_group is used if it has one. A group whose
  // mapping cannot be established either way publishes neither -- fabricating
  // 0..size-1 for a subgroup makes several ranks claim the same global rank,
  // so their dump files collide and one post-mortem is lost.
  static std::shared_ptr<FlightRecorderHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg,
      std::vector<uint64_t> global_ranks = {});

  ~FlightRecorderHook();

  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  // Detach from the process group. Idempotent.
  void remove();

 private:
  // The backend serving one device type of the group. A group can mix
  // backends ("cpu:gloo,cuda:nccl2"), so the recorder instance, the comm_lib
  // field of profiling_name and the timeout are all per device, resolved once
  // at attach. A null recorder means the backend records into a FlightRecorder
  // itself and the hook must leave its ops alone.
  struct BackendTarget {
    std::string name{"c10d"};
    FlightRecorder<c10::Event>* recorder = nullptr;
    std::chrono::milliseconds timeout{kBackendDefaultTimeout};
    // Owned by pg_; only read from hook calls, which the group makes.
    Backend* backend = nullptr;
  };

  // A recorded op whose collective the hook is still waiting on. If the
  // backend threw between the pre-hook and the post-hook it is never retired,
  // which is the honest report: it was issued and never seen to finish. work
  // is null unless the backend timed the op; then it is held until the op's
  // result is published, so the duration can be read after the caller drops
  // the handle, and is then parked in released_works_.
  struct InflightOp {
    FlightRecorder<c10::Event>::TraceIdentifier trace_id;
    FlightRecorder<c10::Event>* recorder = nullptr;
    HookOpName name = HookOpName::UNKNOWN;
    bool timed = false;
    c10::intrusive_ptr<Work> work;
  };

  FlightRecorderHook(
      c10::intrusive_ptr<ProcessGroup> pg,
      std::vector<uint64_t> global_ranks);
  // The target serving an op, or the group's default backend for an op that
  // names no device.
  const BackendTarget& targetFor(std::optional<c10::Device> device) const;
  // Whether a graph capture is active on the stream the op would be issued on.
  // nullopt when that cannot be established, which onPre reads as "capturing":
  // recording an op under capture and then querying its Work is what
  // invalidates the capture. An op with no device of its own falls back to the
  // devices the group serves.
  std::optional<bool> captureActive(std::optional<c10::Device> device) const;
  void onPre(const PreHookArgs& args);
  void onPost(const PostHookArgs& args);
  // Called from the result future of the op's Work. Retires the entry if the
  // op succeeded; a failed op stays un-retired.
  void onResult(int64_t op_id, bool success);
  // Claims the entry of an op out of inflight_, so exactly one caller retires
  // it. The caller drops the Work it holds, outside mutex_.
  std::optional<InflightOp> takeInflightLocked(int64_t op_id);
  // Dumps the trace to disk, at most once per process. Deliberately takes no
  // hook lock; see the definition.
  void onAbort();

  c10::intrusive_ptr<ProcessGroup> pg_;
  int64_t hook_id_;
  std::map<c10::DeviceType, BackendTarget> targets_;
  BackendTarget default_target_;
  // Whether attach() got an abort hook registered; false for backends that
  // have none (gloo), whose unregister call would throw just as loudly.
  bool abort_hook_registered_{false};
  size_t pg_id_;
  // Cap on inflight_, see onPre.
  size_t max_inflight_{0};
  std::shared_ptr<ProcessGroupStatus> pg_status_;

  // Sequencing and the op_id -> in-flight-op map. The mutex guards against
  // concurrent collectives from multiple threads (the hooks fire on the issuing
  // thread), against a completion arriving on a backend thread, and against
  // remove() running while an op is in flight.
  //
  // Lock order: nothing that can block on the GIL, and nothing that calls into
  // a backend, may run under this mutex. remove() runs with the GIL held -- the
  // pybind method does not release it, and the destructor runs when Python
  // drops the handle -- and blocks on this mutex, so a collective thread that
  // held it while waiting for the GIL would deadlock the process with no
  // timeout. That is why onPre() calls the recorder, which gathers a traceback
  // under the GIL, outside the lock, and why no Work is released under it: a
  // Work's destructor may take the GIL (PyWorkHolder). Nor does remove()
  // unregister under it: a hook may be blocked on it while the backend holds
  // the lock that guards its own hook table.
  std::mutex mutex_;
  size_t collective_seq_{0};
  size_t p2p_seq_{0};
  // Ordered by op_id, which is monotonic per process group, so the front is the
  // oldest op still awaited and eviction is in issue order.
  std::map<int64_t, InflightOp> inflight_;
  // Works of ops whose result was published, released by the next onPost or
  // remove() so the last reference is normally not dropped on a backend
  // thread; see onResult.
  std::vector<c10::intrusive_ptr<Work>> released_works_;
  // getFutureResult() failed unexpectedly; warn only once.
  std::atomic<bool> warned_future_result_{false};
};

} // namespace c10d
