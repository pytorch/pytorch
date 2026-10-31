// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// CompletionCallbackHook: backend-agnostic per-op completion callbacks built on
// the ProcessGroup post and completion hooks (Hooks.hpp).
//
// The post-hook is the only place an op's name, op_id and Work are seen
// together, so it maps the Work's completion key to the op. The backend's
// completion hook then reports the key (and the backend's own duration), and
// the matched op is queued for a dedicated dispatch thread that runs the
// callback. The backend thread never runs the callback, takes the GIL or
// waits on the callback: a slow callback only grows the queue.
//
// Callbacks run one at a time, in the order the backend reports completions.
// That is not issue order: nccl2 retires work per CUDA stream (async_op ops run
// on an internal stream, synchronous ones on the current stream) and may report
// from more than one thread. Never reported:
// - failed, timed-out and aborted ops (see CompletionHook);
// - ops issued before attach();
// - ops on a group's non-default backends (e.g. CPU ops of a gloo+nccl2
//   group): completion hooks are registered on the default backend only;
// - nccl2 p2p inside a coalescing window (batch_isend_irecv), which completes
//   only as part of the window, which the post-hook never sees;
// - ops captured into a CUDA graph (their op_ids still advance last_op_id),
//   which the backend does not track, and graph replays, which issue no op.
// Collectives inside a window are reported individually; *_coalesced ops
// once, under their *_COALESCED name. There is no drain API: to wait for
// specific ops, record their op_ids and wait for those.
//
// Requires a default backend that supports completion hooks (nccl2, nccl-lazy)
// and whose completion keys are unique per Work; attach() throws for backends
// without completion hooks, so a caller can fall back rather than silently
// never being called. Key uniqueness is the backend's contract and is not
// checked.
//
// The process group owns the hook and the callback, and releases the callback
// when it is destroyed, even if the handle is still referenced. A callback
// that references its group forms a cycle that keeps both alive until
// remove().

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

struct CompletedOpInfo {
  HookOpName name = HookOpName::UNKNOWN;
  // The op's PostHookArgs::op_id: per process group, monotonic in issue order,
  // comparable with CompletionCallbackHook::lastOpId().
  int64_t op_id = 0;
  // The backend's measurement, or nullopt if it does not time ops. nccl2 times
  // ops only with TORCH_NCCL_ENABLE_TIMING=1 or after
  // _enable_collectives_timing() (nccl2 only, not nccl-lazy).
  std::optional<float> duration_ms;
};

using CompletionCallback = std::function<void(const CompletedOpInfo&)>;

class TORCH_API CompletionCallbackHook
    : public std::enable_shared_from_this<CompletionCallbackHook> {
 public:
  // Attaches the callback to the process group, which owns it from then on.
  // The returned handle only needs to be kept to remove() it early or to read
  // lastOpId(). Like hook registration in general, must not race with ops
  // issued on the group, so throws if called from a completion callback, or
  // after stopAll().
  static std::shared_ptr<CompletionCallbackHook> attach(
      const c10::intrusive_ptr<ProcessGroup>& pg,
      CompletionCallback callback);

  CompletionCallbackHook(const CompletionCallbackHook&) = delete;
  CompletionCallbackHook(CompletionCallbackHook&&) = delete;
  CompletionCallbackHook& operator=(const CompletionCallbackHook&) = delete;
  CompletionCallbackHook& operator=(CompletionCallbackHook&&) = delete;
  ~CompletionCallbackHook();

  // Detaches from the process group and drops callbacks not yet delivered.
  // Waits for a running callback to return. Idempotent. Like attach(), must
  // not race with ops issued on the group. From a completion callback (of any
  // hook), which may run concurrently with such ops, it neither waits nor
  // touches the group's hook tables: the hooks stay registered but inert until
  // a later remove() from another thread or the group is destroyed. In a
  // forked child it does nothing: the child has no dispatch thread, and locks
  // here or in the backend may have been held by a parent thread at fork.
  void remove();

  // For interpreter exit: drops queued callbacks of every hook, waits for
  // running callbacks to return and releases the callbacks, so that no
  // dispatch thread enters the interpreter once it finalizes. Hooks keep
  // tracking but deliver nothing afterwards, and attach() throws.
  static void stopAll();

  // op_id of the most recently issued op seen since attach(), or -1.
  int64_t lastOpId() const {
    return last_op_id_.load();
  }

 private:
  struct Dispatcher;

  CompletionCallbackHook(
      const c10::intrusive_ptr<ProcessGroup>& pg,
      CompletionCallback callback);
  void onPost(const PostHookArgs& args);
  void onCompletion(const CompletionHookArgs& args);

  struct PendingOp {
    int64_t op_id = -1;
    HookOpName name = HookOpName::UNKNOWN;
  };

  // Weak: the group owns the hook.
  c10::weak_intrusive_ptr<ProcessGroup> pg_;
  int64_t hook_id_;
  std::atomic<int64_t> last_op_id_{-1};
  std::shared_ptr<Dispatcher> dispatcher_;

  // Lock order: nothing that calls into the backend or waits on the dispatch
  // thread runs under mutex_.
  std::mutex mutex_;
  bool attached_ = true;
  // Issued ops by completion key, and the same keys in issue order so the
  // oldest can be evicted. Ops that never complete successfully (failures,
  // aborts, ops a backend never reports) would otherwise accumulate.
  std::unordered_map<uint64_t, PendingOp> pending_;
  std::map<int64_t, uint64_t> pending_order_;
  // Completions reported before the post-hook registered their key: a fast op
  // can finish before the post-hook runs. Bounded for the same reason; keys
  // of Works the post-hook never sees (coalescing windows) land here too.
  std::unordered_map<uint64_t, std::optional<float>> early_;
  std::deque<uint64_t> early_order_;
};

} // namespace c10d
