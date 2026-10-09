#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

// Identifies the operation a pre/post hook is firing for. One value per
// dispatcher op rather than per collective family: the flight recorder writes
// the name into the trace and the analyzer keys its size rules off it, so
// folding e.g. every allgather variant into ALLGATHER would report an
// _allgather_base as a plain all_gather and apply the wrong check to it.
// Consumers that only care about the family (NanCheckHook) can ignore the
// extra values.
enum class HookOpName : uint8_t {
  SEND = 0,
  RECV,
  BROADCAST,
  ALLREDUCE,
  REDUCE,
  ALLGATHER,
  REDUCE_SCATTER,
  ALLTOALL,
  BARRIER,
  SCATTER,
  GATHER,
  SPLIT,
  NEW_WINDOW,
  ALLREDUCE_COALESCED,
  ALLGATHER_BASE,
  ALLGATHER_COALESCED,
  ALLGATHER_INTO_TENSOR_COALESCED,
  REDUCE_SCATTER_BASE,
  REDUCE_SCATTER_TENSOR_COALESCED,
  ALLTOALL_BASE,
  UNKNOWN,
};

// Arguments passed to a pre-hook, fired before an operation is issued.
struct PreHookArgs {
  HookOpName name = HookOpName::UNKNOWN;
  bool async_op = false;
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;
  int64_t root = -1;
  // Correlates a pre-hook call with its matching post-hook call.
  int64_t op_id = 0;
};

using PreHook = std::function<void(const PreHookArgs&)>;

// Arguments passed to a post-hook, fired after an operation is issued.
struct PostHookArgs {
  HookOpName name = HookOpName::UNKNOWN;
  bool async_op = false;
  // Work handle for the issued operation; may be null for synchronous ops.
  c10::intrusive_ptr<Work> work;
  int64_t op_id = 0;
};

using PostHook = std::function<void(const PostHookArgs&)>;

// Abort hook - called before aborting when a collective times out or fails.
// This allows users to capture debug information before the abort.
//
// A single failure may invoke the hook more than once: it is observed by the
// backend's watchdog and again by the next collective, and a backend may run
// its hooks both where the failure is detected and where it tears down. Make
// the hook idempotent. Deduplicating at the call site instead is unsafe -- the
// thread that terminates the process is not the thread that detects the
// failure, so a hook that returned early on "already ran" would let the process
// die mid-capture. The hook's own one-shot has to block, not skip.
using AbortHook = std::function<void()>;

// Arguments passed to a completion hook, fired when the backend establishes
// that an operation has finished.
struct CompletionHookArgs {
  // Correlates completion with PostHookArgs::work.
  uint64_t completionKey = 0;
  // The backend's own measurement of the op in ms, or nullopt if it does not
  // time its collectives. The backend supplies it because only the backend
  // knows whether timing is on -- Work::getDuration() throws when it is not,
  // so a consumer could only find out by provoking an exception per op.
  std::optional<float> duration_ms;
};

// Completion hook - called by a backend when it establishes that an operation
// completed *successfully*. A backend already learns this to garbage collect
// its work queue, so pushing it out saves every consumer from re-deriving the
// same fact by polling Work::isCompleted() on everything it has issued.
//
// Failed and timed-out work deliberately fires nothing. Work::isCompleted() is
// true for those too, and a consumer that took them for completed would erase
// the one fact a post-mortem needs: that the collective never finished.
//
// Fires once per Work. May run on the backend's watchdog thread and with
// backend locks held, so the hook must be short, must not take the GIL, and
// must not call back into the backend.
using CompletionHook = std::function<void(const CompletionHookArgs&)>;

} // namespace c10d
