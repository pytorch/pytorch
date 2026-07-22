// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WindowNCCL.hpp>

#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

namespace {
// NCCL's RMA APIs currently only accept (sigIdx=0, ctx=0, flags=0).
constexpr int kSigIdx = 0;
constexpr int kCtx = 0;
constexpr unsigned int kFlags = 0;
} // namespace

WindowNCCL::WindowNCCL(c10::intrusive_ptr<ProcessGroupNCCL> pg)
    : pg_(std::move(pg)) {
  TORCH_CHECK(pg_, "WindowNCCL: null ProcessGroupNCCL");
  nccl_api_ = pg_->getNcclApi();
  nccl_comm_ = pg_->nccl_comm_;
  TORCH_CHECK(
      nccl_comm_ != nullptr && nccl_api_ != nullptr,
      "WindowNCCL: ProcessGroupNCCL is not initialized");
}

void WindowNCCL::tensor_register(const at::Tensor& tensor, bool owning) {
  TORCH_CHECK(tensor.defined(), "WindowNCCL: a valid tensor is required");
  checkDeviceAndThrow(tensor);
  TORCH_CHECK(win_ == nullptr, "WindowNCCL: double registration");
  TORCH_CHECK(tensor.is_contiguous(), "WindowNCCL: contiguous tensor required");

  // Each segment from the NCCL mempool is tracked by the allocator hook.
  // Register the underlying segment as a NCCL_WIN_COLL_SYMMETRIC window
  // (collective). All ranks must reach this point with matching segments --
  // which holds for symmetric allocation patterns (the standard usage).
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      pg_->ensureSegmentWindow(tensor.data_ptr()),
      "WindowNCCL: ncclCommWindowRegister failed; the tensor must be "
      "allocated from the NCCL mempool (e.g. "
      "torch.cuda.MemPool(backend.mem_allocator))");
  auto [seg_win, seg_offset] = pg_->lookupSegmentWindow(tensor.data_ptr());
  TORCH_CHECK(
      seg_win != nullptr,
      "WindowNCCL: window registration succeeded but segment lookup "
      "returned null (internal error)");

  win_ = seg_win;
  peer_win_offset_ = seg_offset;
  win_size_ = tensor.numel() * tensor.element_size();
  if (owning) {
    buf_tensor_ = tensor;
  }
}

void WindowNCCL::tensor_deregister() {
  // Segment-level deregistration happens automatically when the mempool frees
  // the segment (via NcclCachingAllocatorHook). Here we just forget the
  // tensor; the barriers keep ranks aligned around the collective contract.
  pg_->barrier();
  TORCH_CHECK(win_ != nullptr, "WindowNCCL: double deregistration");
  win_ = nullptr;
  peer_win_offset_ = 0;
  win_size_ = 0;
  buf_tensor_.reset();
  pg_->barrier();
}

c10::intrusive_ptr<::c10d::Work> WindowNCCL::put(
    const at::Tensor& tensor,
    int64_t dstRank,
    int64_t targetOffsetNelems,
    bool asyncOp,
    const ::c10d::PutOptions& opts) {
  checkWindowAndThrow();
  checkDeviceAndThrow(tensor);
  TORCH_CHECK(
      tensor.is_contiguous(), "WindowNCCL: source tensor must be contiguous");

  const size_t elem_size = tensor.element_size();
  const size_t put_bytes = tensor.numel() * elem_size;
  const size_t target_offset_bytes =
      static_cast<size_t>(targetOffsetNelems) * elem_size;
  TORCH_CHECK(
      put_bytes + target_offset_bytes <= win_size_,
      "WindowNCCL: requested size (",
      put_bytes + target_offset_bytes,
      " bytes) exceeds the window size (",
      win_size_,
      " bytes)");

  // Ensure the source tensor's underlying segment is registered as a
  // symmetric window -- NCCL's ncclPutSignal looks it up internally. This is
  // a no-op on segments already registered, so hot-path puts on the same
  // buffer are zero-overhead.
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      pg_->ensureSegmentWindow(tensor.data_ptr()),
      "WindowNCCL: the source tensor must be allocated from the NCCL "
      "mempool (e.g. torch.cuda.MemPool(backend.mem_allocator))");

  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto work =
      pg_->createWork(stream, pg_->operationTimeout(opts.timeout), tensor);
  work->recordStart("put");
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->putSignal(
          tensor.data_ptr(),
          tensor.numel(),
          pg_->getNcclDataType(tensor),
          static_cast<int>(dstRank),
          win_,
          peer_win_offset_ + target_offset_bytes,
          kSigIdx,
          kCtx,
          kFlags,
          nccl_comm_,
          stream),
      "WindowNCCL::put ncclPutSignal failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<::c10d::Work> WindowNCCL::signal(
    int64_t peerRank,
    bool asyncOp,
    const ::c10d::SignalOptions& opts) {
  checkWindowAndThrow();
  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto work = pg_->createWork(stream, pg_->operationTimeout(opts.timeout));
  work->recordStart("signal");
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->signal(
          static_cast<int>(peerRank),
          kSigIdx,
          kCtx,
          kFlags,
          nccl_comm_,
          stream),
      "WindowNCCL::signal ncclSignal failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<::c10d::Work> WindowNCCL::wait_signal(
    int64_t peerRank,
    bool asyncOp,
    const ::c10d::WaitSignalOptions& opts) {
  checkWindowAndThrow();
  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto work = pg_->createWork(stream, pg_->operationTimeout(opts.timeout));
  work->recordStart("wait_signal");
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->waitSignal(
          static_cast<int>(peerRank),
          kSigIdx,
          kCtx,
          /*opCnt=*/1,
          nccl_comm_,
          stream),
      "WindowNCCL::wait_signal ncclWaitSignal failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

at::Tensor WindowNCCL::map_remote_tensor(int64_t rank) {
  checkWindowAndThrow();
  // Upstream NCCL only exposes the local user pointer via ncclWinGetUserPtr --
  // there is no direct mapping of peer windows. For self-rank we return the
  // local backing tensor; cross-rank mapping is not supported.
  TORCH_CHECK(
      rank == pg_->getRank(),
      "WindowNCCL: map_remote_tensor(rank=",
      rank,
      ") is only supported for the local rank (",
      pg_->getRank(),
      ") -- upstream NCCL does not expose peer window pointers");
  TORCH_CHECK(
      buf_tensor_.has_value(),
      "WindowNCCL: map_remote_tensor on the local rank requires an owning "
      "tensor_register (the default)");
  return *buf_tensor_;
}

::c10d::WindowAttr WindowNCCL::get_attr(int64_t /*peerRank*/) {
  checkWindowAndThrow();
  // Upstream NCCL does not expose per-peer window access metadata. Report
  // SEPARATE so callers fall back to put/signal rather than expecting a
  // direct NVLink mapping.
  return ::c10d::WindowAttr{.access_type = ::c10d::WindowAccessType::SEPARATE};
}

void WindowNCCL::checkWindowAndThrow() const {
  TORCH_CHECK(
      win_ != nullptr,
      "WindowNCCL: window not registered (call tensor_register first)");
}

void WindowNCCL::checkDeviceAndThrow(const at::Tensor& tensor) const {
  TORCH_CHECK(
      tensor.device() == pg_->getDevice(),
      "WindowNCCL: device mismatch: process group on device ",
      pg_->getDevice(),
      ", tensor on device ",
      tensor.device());
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
