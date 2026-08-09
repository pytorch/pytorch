// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WindowNCCL.hpp>

#include <array>
#include <cstring>
#include <limits>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/safe_numerics.h>
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

void WindowNCCL::exchangePeerMetadata(
    size_t local_offset,
    size_t local_size,
    at::ScalarType local_dtype) {
  const std::array<uint64_t, 3> metadata = {
      static_cast<uint64_t>(local_offset),
      static_cast<uint64_t>(local_size),
      static_cast<uint64_t>(local_dtype)};
  auto local_metadata = at::empty({3}, at::TensorOptions().dtype(at::kLong));
  std::memcpy(local_metadata.data_ptr(), metadata.data(), sizeof(metadata));
  local_metadata = local_metadata.to(pg_->getDevice());
  auto gathered_metadata = at::empty(
      {pg_->getSize(), local_metadata.numel()}, local_metadata.options());
  ::c10d::AllgatherOptions opts;
  opts.asyncOp = false;
  opts.timeout = pg_->options_c10d_->timeout;
  pg_->_allgather_base(gathered_metadata, local_metadata, opts)
      ->wait(opts.timeout);
  const auto gathered_metadata_cpu = gathered_metadata.cpu();
  const auto* gathered_data = gathered_metadata_cpu.const_data_ptr<int64_t>();

  std::vector<size_t> peer_offsets;
  std::vector<size_t> peer_sizes;
  std::vector<at::ScalarType> peer_dtypes;
  const auto world_size = static_cast<size_t>(pg_->getSize());
  peer_offsets.reserve(world_size);
  peer_sizes.reserve(world_size);
  peer_dtypes.reserve(world_size);
  for (const auto rank : c10::irange(world_size)) {
    std::array<uint64_t, 3> peer_metadata{};
    std::memcpy(
        peer_metadata.data(),
        gathered_data + rank * peer_metadata.size(),
        sizeof(peer_metadata));
    TORCH_CHECK(
        peer_metadata[0] <= std::numeric_limits<size_t>::max() &&
            peer_metadata[1] <= std::numeric_limits<size_t>::max() &&
            peer_metadata[2] <
                static_cast<uint64_t>(at::ScalarType::NumOptions),
        "WindowNCCL: invalid peer window metadata");
    peer_offsets.push_back(static_cast<size_t>(peer_metadata[0]));
    peer_sizes.push_back(static_cast<size_t>(peer_metadata[1]));
    peer_dtypes.push_back(static_cast<at::ScalarType>(peer_metadata[2]));
  }
  peer_win_offsets_ = std::move(peer_offsets);
  peer_win_sizes_ = std::move(peer_sizes);
  peer_dtypes_ = std::move(peer_dtypes);
}

void WindowNCCL::tensor_register(const at::Tensor& tensor, bool owning) {
  TORCH_CHECK(tensor.defined(), "WindowNCCL: a valid tensor is required");
  checkDeviceAndThrow(tensor);
  TORCH_CHECK(win_ == nullptr, "WindowNCCL: double registration");
  TORCH_CHECK(tensor.is_contiguous(), "WindowNCCL: contiguous tensor required");
  c10::cuda::CUDAGuard device_guard(pg_->getDevice());

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

  size_t win_size = 0;
  const auto numel = static_cast<size_t>(tensor.numel());
  TORCH_CHECK(
      !c10::mul_overflows(numel, tensor.element_size(), &win_size),
      "WindowNCCL: tensor size overflows size_t");
  exchangePeerMetadata(seg_offset, win_size, tensor.scalar_type());
  win_ = seg_win;
  if (owning) {
    buf_tensor_ = tensor;
  }
}

void WindowNCCL::tensor_deregister() {
  // Segment-level deregistration happens automatically when the mempool frees
  // the segment (via NCCLCachingAllocatorHook). Here we just forget the
  // tensor; the barriers keep ranks aligned around the collective contract.
  pg_->barrier();
  TORCH_CHECK(win_ != nullptr, "WindowNCCL: double deregistration");
  win_ = nullptr;
  peer_win_offsets_.clear();
  peer_win_sizes_.clear();
  peer_dtypes_.clear();
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
  checkPeerRankAndThrow(dstRank);
  TORCH_CHECK(
      tensor.is_contiguous(), "WindowNCCL: source tensor must be contiguous");
  TORCH_CHECK(
      targetOffsetNelems >= 0,
      "WindowNCCL: target offset must be nonnegative, got ",
      targetOffsetNelems);

  const auto peer = static_cast<size_t>(dstRank);
  TORCH_CHECK(
      tensor.scalar_type() == peer_dtypes_[peer],
      "WindowNCCL: source dtype ",
      tensor.scalar_type(),
      " does not match destination window dtype ",
      peer_dtypes_[peer],
      " on rank ",
      dstRank);

  const size_t elem_size = tensor.element_size();
  size_t put_bytes = 0;
  size_t target_offset_bytes = 0;
  TORCH_CHECK(
      !c10::mul_overflows(
          static_cast<size_t>(tensor.numel()), elem_size, &put_bytes) &&
          !c10::mul_overflows(
              static_cast<size_t>(targetOffsetNelems),
              elem_size,
              &target_offset_bytes),
      "WindowNCCL: requested put size or target offset overflows size_t");
  TORCH_CHECK(
      target_offset_bytes <= peer_win_sizes_[peer] &&
          put_bytes <= peer_win_sizes_[peer] - target_offset_bytes,
      "WindowNCCL: requested size (",
      put_bytes,
      " bytes at offset ",
      target_offset_bytes,
      " bytes) exceeds the window size (",
      peer_win_sizes_[peer],
      " bytes)");
  size_t peer_offset = 0;
  TORCH_CHECK(
      !c10::add_overflows(
          peer_win_offsets_[peer], target_offset_bytes, &peer_offset),
      "WindowNCCL: target byte offset overflows size_t");
  c10::cuda::CUDAGuard device_guard(pg_->getDevice());

  // ncclPutSignal requires its source to already belong to a symmetric window.
  // Registration is collective, so put must only validate this rank's state.
  const auto source_win = pg_->lookupSegmentWindow(tensor.data_ptr()).first;
  TORCH_CHECK(
      source_win != nullptr,
      "WindowNCCL: source tensor is not in a registered symmetric window; "
      "register it collectively before calling put");

  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto timeout = pg_->operationTimeout(opts.timeout);
  auto work = pg_->createWork(stream, timeout, tensor);
  work->recordStart("put");
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
  try {
    NCCL_CHECK_NONBLOCKING(
        nccl_api_,
        nccl_comm_,
        nccl_api_->putSignal(
            tensor.data_ptr(),
            tensor.numel(),
            pg_->getNcclDataType(tensor),
            static_cast<int>(dstRank),
            win_,
            peer_offset,
            kSigIdx,
            kCtx,
            kFlags,
            nccl_comm_,
            stream),
        "WindowNCCL::put ncclPutSignal failed");
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }
  pg_->waitForNcclOperation(
      nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<::c10d::Work> WindowNCCL::signal(
    int64_t peerRank,
    bool asyncOp,
    const ::c10d::SignalOptions& opts) {
  checkWindowAndThrow();
  checkPeerRankAndThrow(peerRank);
  c10::cuda::CUDAGuard device_guard(pg_->getDevice());
  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto timeout = pg_->operationTimeout(opts.timeout);
  auto work = pg_->createWork(stream, timeout);
  work->recordStart("signal");
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
  try {
    NCCL_CHECK_NONBLOCKING(
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
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }
  pg_->waitForNcclOperation(
      nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<::c10d::Work> WindowNCCL::wait_signal(
    int64_t peerRank,
    bool asyncOp,
    const ::c10d::WaitSignalOptions& opts) {
  checkWindowAndThrow();
  checkPeerRankAndThrow(peerRank);
  c10::cuda::CUDAGuard device_guard(pg_->getDevice());
  cudaStream_t stream = pg_->getOperationStream(asyncOp);
  auto timeout = pg_->operationTimeout(opts.timeout);
  auto work = pg_->createWork(stream, timeout);
  work->recordStart("wait_signal");
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
  try {
    NCCL_CHECK_NONBLOCKING(
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
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }
  pg_->waitForNcclOperation(
      nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
  work->recordEnd();
  pg_->enqueueWork(work, stream);
  return work;
}

at::Tensor WindowNCCL::map_remote_tensor(int64_t rank) {
  checkWindowAndThrow();
  checkPeerRankAndThrow(rank);
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

::c10d::WindowAttr WindowNCCL::get_attr(int64_t peerRank) {
  checkWindowAndThrow();
  checkPeerRankAndThrow(peerRank);
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

void WindowNCCL::checkPeerRankAndThrow(int64_t rank) const {
  TORCH_CHECK(
      rank >= 0 && rank < pg_->getSize(),
      "WindowNCCL: peer rank ",
      rank,
      " is outside [0, ",
      pg_->getSize(),
      ")");
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
