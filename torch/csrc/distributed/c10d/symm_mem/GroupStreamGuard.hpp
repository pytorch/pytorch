#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace c10d {
class ProcessGroup;
} // namespace c10d

namespace c10d::symmetric_memory {

// RAII guard that orders the built-in symmetric-memory operations that
// synchronize ranks through a group's signal pad or NCCL device communicator,
// within a (process group, device) pair: barrier, put_signal and wait_signal,
// the collectives in CUDASymmetricMemoryOps.cu, the NCCL backend's barrier,
// the signal ops in nccl_extension.cu, and nccl_reduce_scatter_offset and
// nccl_all_to_all_nd, whose LSA barriers belong to the group's device
// communicator. Not covered: ncclPutSignal/ncclWaitSignal, the NVSHMEM
// backend, and user kernels on a raw get_signal_pad() tensor.
//
// One guarded scope is one pad operation: construct, launch the kernel, let
// the guard go out of scope. Construction takes the group's mutex and, when
// the current stream differs from the previous operation's, waits on the
// event marking the end of that operation. Destruction records that event on
// the current stream, just after the launch. The mutex spans the launch, so
// two host threads cannot interleave reading the stream with launching.
//
// State is keyed by (ProcessGroup identity, device), with a weak reference to
// the group as the liveness check.
//
// Under graph capture the record and wait become graph nodes, ordering
// streams forked inside one capture. An event can only be waited on from the
// capture context it was recorded in, so the ordering is per context: eager
// operations are ordered among themselves, and so are the operations of the
// capture in progress, each keeping its own previous stream and event. A
// captured operation therefore never drops the ordering between two eager
// operations, even when a capture starts without synchronizing the device
// first (a raw CUDAGraph::capture_begin). Not ordered: captured operations
// against eager ones, and a graph replay against eager operations or other
// replays; callers order replays themselves. One capture is tracked at a
// time, so captures interleaved from several threads are not ordered within
// themselves.
class TORCH_API GroupStreamGuard {
 public:
  struct State;

  explicit GroupStreamGuard(const std::string& group_name);
  ~GroupStreamGuard();
  GroupStreamGuard(const GroupStreamGuard&) = delete;
  GroupStreamGuard& operator=(const GroupStreamGuard&) = delete;
  GroupStreamGuard(GroupStreamGuard&&) = delete;
  GroupStreamGuard& operator=(GroupStreamGuard&&) = delete;

 private:
  struct LastOp;

  std::shared_ptr<State> state_;
  // Held until destruction, which is after the kernel launch at every call
  // site. Do not shorten this scope: see the class comment.
  std::unique_lock<std::mutex> lock_;
  // This operation's context in state_, whose stream is the one it launched
  // on. The event is recorded there on destruction.
  LastOp* last_ = nullptr;
};

class SignalPad;

// The internal signal pad of (pg, device), built by `create` on the first call
// and kept in the state GroupStreamGuard orders the group's ops by, so the pad
// and its ordering share one key. `create` is collective: every rank must reach
// the first call for a (pg, device) together. The state drops its reference
// when the group is gone and a later group's state is created.
TORCH_API std::shared_ptr<const SignalPad> get_or_create_signal_pad(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    c10::DeviceIndex device,
    const std::function<std::shared_ptr<const SignalPad>()>& create);

} // namespace c10d::symmetric_memory
