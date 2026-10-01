#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

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
// streams forked inside one capture. An event can only be waited on from the
// capture context it was recorded in, so the ordering is per context: eager
// operations are ordered among themselves, and so are the operations of one
// capture, each context keeping its own previous stream and event. An
// operation in another context therefore never drops the ordering between
// two eager operations, even when a capture starts without synchronizing the
// device first (a raw CUDAGraph::capture_begin).
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
  struct Frontier;

  void init_(const c10::intrusive_ptr<c10d::ProcessGroup>& pg);

  std::shared_ptr<State> state_;
  // Held until destruction, which is after the kernel launch at every call
  // site. Do not shorten this scope: see the class comment.
  std::unique_lock<std::mutex> lock_;
  // This operation's capture context in state_, whose last_stream is the
  // stream it launched on. The event is recorded there on destruction.
  Frontier* frontier_ = nullptr;
};

} // namespace c10d::symmetric_memory
