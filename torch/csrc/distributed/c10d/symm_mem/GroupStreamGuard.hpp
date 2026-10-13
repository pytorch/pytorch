#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace c10d {
class ProcessGroup;
} // namespace c10d

namespace c10d::symmetric_memory {

// RAII guard that orders the built-in CUDA-backend symmetric-memory
// operations touching the signal pad within a (process group, device) pair:
// barrier, put_signal and wait_signal, the collectives in
// CUDASymmetricMemoryOps.cu, the NCCL backend's barrier, and the signal ops in
// nccl_extension.cu. Not covered: ncclPutSignal/ncclWaitSignal, the NVSHMEM
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
// the group as the liveness check. The group_name arguments only resolve the
// group when the caller has not already.
//
// Under graph capture the record and wait become graph nodes, ordering
// streams forked inside one capture. The event belongs to the capture it was
// recorded in, so the wait is skipped when the next operation runs in a
// different capture context. A program that warms up on one stream and then
// captures on another hits that on its first captured operation, which is the
// normal pattern and not something a caller can act on, so the skip is logged
// at debug level rather than warned about.
class TORCH_API GroupStreamGuard {
 public:
  struct State;

  explicit GroupStreamGuard(const std::string& group_name);
  GroupStreamGuard(
      const std::string& group_name,
      const c10::intrusive_ptr<c10d::ProcessGroup>& pg);
  ~GroupStreamGuard();
  GroupStreamGuard(const GroupStreamGuard&) = delete;
  GroupStreamGuard& operator=(const GroupStreamGuard&) = delete;
  GroupStreamGuard(GroupStreamGuard&&) = delete;
  GroupStreamGuard& operator=(GroupStreamGuard&&) = delete;

 private:
  void init_(
      const std::string& group_name,
      const c10::intrusive_ptr<c10d::ProcessGroup>& pg);

  std::shared_ptr<State> state_;
  // Held until destruction, which is after the kernel launch at every call
  // site. Do not shorten this scope: see the class comment.
  std::unique_lock<std::mutex> lock_;
  // The stream this operation launched on; the event is recorded here on
  // destruction.
  std::optional<c10::cuda::CUDAStream> stream_;
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
