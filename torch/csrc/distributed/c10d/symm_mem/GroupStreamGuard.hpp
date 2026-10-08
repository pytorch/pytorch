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

// RAII guard that orders the built-in symmetric-memory operations that pair up
// across ranks within a (process group, device): barrier, put_signal and
// wait_signal on the CUDA, NCCL and NVSHMEM backends, the collectives in
// CUDASymmetricMemoryOps.cu and nvshmem_extension.cu, and the signal ops in
// nccl_extension.cu. Two of these issued concurrently on different streams can
// pair with the wrong counterpart on a peer, because the rendezvous they use --
// the signal pad for the pad ops, an NVSHMEM team for the NVSHMEM collectives
// -- carries no tag saying which operation an arrival belongs to.
//
// Not covered: point-to-point data movement (nvshmem_put/nvshmem_get), the
// signal ops taking a caller-supplied signal word and value (ncclPutSignal,
// ncclWaitSignal, nvshmem_put_with_signal, nvshmem_wait_for_signal), where that
// address and value identify the handshake so there is nothing to mismatch, and
// user kernels on a raw get_signal_pad() tensor.
//
// One guarded scope is one operation: construct, launch the kernel, let the
// guard go out of scope. An operation that spans several launches, as the
// all_to_all_vdev family does, holds one guard across all of them.
// Construction takes the group's mutex and, when the current stream differs
// from the previous operation's, waits on the event marking the end of that
// operation. Destruction records that event on the current stream, just after
// the launch. The mutex spans the launch, so two host threads cannot
// interleave reading the stream with launching.
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

} // namespace c10d::symmetric_memory
