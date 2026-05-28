// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/nccl/TorchCommWindowNCCL.hpp>

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>
#include <torch/csrc/comms/utils/Logging.hpp>

namespace torch::comms {

namespace {

// NCCL's RMA APIs currently only accept (sigIdx=0, ctx=0, flags=0).
constexpr int kSigIdx = 0;
constexpr int kCtx = 0;
constexpr unsigned int kFlags = 0;

ncclDataType_t torchToNcclDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kChar:
      return ncclInt8;
    case at::kByte:
      return ncclUint8;
    case at::kInt:
      return ncclInt32;
    case at::kLong:
      return ncclInt64;
    case at::kHalf:
      return ncclFloat16;
    case at::kFloat:
      return ncclFloat32;
    case at::kDouble:
      return ncclFloat64;
    case at::kBFloat16:
      return ncclBfloat16;
    default:
      throw std::runtime_error(fmt::format(
          "[TorchCommWindowNCCL]: unsupported dtype {}", c10::toString(dtype)));
  }
}

} // namespace

TorchCommWindowNCCL::TorchCommWindowNCCL(
    std::shared_ptr<TorchCommNCCL> torchComm)
    : torch_comm_(std::move(torchComm)) {
  if (!torch_comm_) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: null TorchCommNCCL communicator");
  }
  nccl_api_ = torch_comm_->getNcclApi();
  nccl_comm_ = torch_comm_->nccl_comm_;
  comm_device_ = torch_comm_->getDevice();
  if (nccl_comm_ == nullptr || nccl_api_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: TorchCommNCCL is not initialized");
  }
}

void TorchCommWindowNCCL::tensor_register(
    const at::Tensor& tensor,
    bool owning) {
  checkCommAndThrow();

  if (!tensor.defined()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL][register]: a valid tensor is required.");
  }
  checkDeviceAndThrow(tensor);
  if (win_ != nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL][register]: Double registration error.");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL][register]: contiguous tensor required.");
  }

  // Each segment from the NCCL mempool is tracked by the allocator hook.
  // Register the underlying segment as a NCCL_WIN_COLL_SYMMETRIC window
  // (collective). All ranks must reach this point with matching segments —
  // which holds for symmetric allocation patterns (the standard usage).
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      torch_comm_->ensureSegmentWindow(tensor.data_ptr()),
      "[TorchCommWindowNCCL][register]: ncclCommWindowRegister failed; "
      "tensor must be allocated from the NCCL mempool (e.g. "
      "`torch.cuda.use_mem_pool(comm.get_mem_allocator())`).");
  auto [seg_win, seg_offset] =
      torch_comm_->lookupSegmentWindow(tensor.data_ptr());
  if (seg_win == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL][register]: window registration succeeded but "
        "segment lookup returned null (internal error).");
  }

  buf_dtype_ = tensor.scalar_type();
  win_size_ = tensor.numel() * tensor.element_size();
  buf_shape_.assign(tensor.sizes().begin(), tensor.sizes().end());

  win_ = seg_win;
  peer_win_offset_ = seg_offset;

  if (owning) {
    buf_tensor_ = tensor;
  }
  buf_device_ = tensor.device();
}

void TorchCommWindowNCCL::tensor_deregister() {
  checkCommAndThrow();
  // Segment-level deregistration happens automatically when the mempool frees
  // the segment (via NcclCachingAllocatorHookImpl). Here we just forget the
  // tensor.
  torch_comm_->barrier(false);

  if (win_ == nullptr) {
    throw std::runtime_error("[TorchCommWindowNCCL]: Double deregistration.");
  }
  win_ = nullptr;
  peer_win_offset_ = 0;
  win_size_ = 0;
  buf_tensor_.reset();

  torch_comm_->barrier(false);
}

std::shared_ptr<TorchCommWindow> TorchCommWindowNCCL::clone() {
  auto new_window = std::make_shared<TorchCommWindowNCCL>(torch_comm_);
  if (buf_tensor_.has_value()) {
    new_window->tensor_register(buf_tensor_->clone());
  }
  return new_window;
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCL::put(
    const at::Tensor& tensor,
    int dstRank,
    size_t targetOffsetNelems,
    bool asyncOp,
    const PutOptions& options) {
  checkCommAndThrow();
  checkWindowAndThrow();
  checkDeviceAndThrow(tensor);
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL][put]: source tensor must be contiguous.");
  }

  const size_t elem_size = tensor.element_size();
  const size_t put_bytes = tensor.numel() * elem_size;
  const size_t target_offset_bytes = targetOffsetNelems * elem_size;
  checkRequestSizeAndThrow(put_bytes + target_offset_bytes);

  // Ensure the source tensor's underlying segment is registered as a
  // symmetric window — NCCL's ncclPutSignal looks it up internally via
  // ncclDevrFindWindow. This is a no-op on segments already registered, so
  // hot-path puts on the same buffer are zero-overhead.
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      torch_comm_->ensureSegmentWindow(tensor.data_ptr()),
      "[TorchCommWindowNCCL][put]: source tensor must be allocated from the "
      "NCCL mempool (e.g. "
      "`torch.cuda.use_mem_pool(comm.get_mem_allocator())`).");

  cudaStream_t stream = torch_comm_->getOperationStream(asyncOp);
  auto work = torch_comm_->createWork(stream, options.timeout, {tensor});
  work->recordStart("put");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->putSignal(
          tensor.data_ptr(),
          tensor.numel(),
          torchToNcclDtype(tensor.scalar_type()),
          dstRank,
          win_,
          peer_win_offset_ + target_offset_bytes,
          kSigIdx,
          kCtx,
          kFlags,
          nccl_comm_,
          stream),
      "TorchCommWindowNCCL::put ncclPutSignal failed");

  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCL::signal(
    int peerRank,
    bool asyncOp,
    const SignalOptions& options) {
  checkCommAndThrow();
  checkWindowAndThrow();

  cudaStream_t stream = torch_comm_->getOperationStream(asyncOp);
  auto work = torch_comm_->createWork(stream, options.timeout);
  work->recordStart("signal");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->signal(peerRank, kSigIdx, kCtx, kFlags, nccl_comm_, stream),
      "TorchCommWindowNCCL::signal ncclSignal failed");

  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCL::wait_signal(
    int peerRank,
    bool asyncOp,
    const WaitSignalOptions& options) {
  checkCommAndThrow();
  checkWindowAndThrow();

  cudaStream_t stream = torch_comm_->getOperationStream(asyncOp);
  auto work = torch_comm_->createWork(stream, options.timeout);
  work->recordStart("wait_signal");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->waitSignal(
          peerRank, kSigIdx, kCtx, /*opCnt=*/1, nccl_comm_, stream),
      "TorchCommWindowNCCL::wait_signal ncclWaitSignal failed");

  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

at::Tensor TorchCommWindowNCCL::map_remote_tensor(int rank) {
  checkCommAndThrow();
  checkWindowAndThrow();

  // Upstream NCCL only exposes the local user pointer via ncclWinGetUserPtr —
  // there is no direct mapping of peer windows. For self-rank we return the
  // local backing tensor; cross-rank mapping is not supported.
  if (rank != torch_comm_->getRank()) {
    throw std::runtime_error(fmt::format(
        "[TorchCommWindowNCCL]: map_remote_tensor(rank={}) is only "
        "supported for the local rank ({}) — upstream NCCL does not "
        "expose peer window pointers.",
        rank,
        torch_comm_->getRank()));
  }

  if (!buf_tensor_.has_value()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: map_remote_tensor on the local rank requires "
        "an owning tensor_register (the default).");
  }
  return *buf_tensor_;
}

std::shared_ptr<TorchCommWindowAttr> TorchCommWindowNCCL::get_attr(
    int /*peerRank*/) {
  checkCommAndThrow();
  checkWindowAndThrow();
  // Upstream NCCL does not expose per-peer window access metadata. Report
  // SEPARATE so callers fall back to put/get/signal rather than expecting a
  // direct NVLink mapping.
  auto attr = std::make_shared<TorchCommWindowAttr>();
  attr->accessType = TorchCommWinAccessType::WIN_ACCESS_TYPE_SEPARATE;
  return attr;
}

void TorchCommWindowNCCL::checkCommAndThrow() const {
  if (nccl_comm_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: NCCL communicator not initialized");
  }
  if (torch_comm_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: TorchComm not initialized");
  }
}

void TorchCommWindowNCCL::checkWindowAndThrow() const {
  if (win_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCL]: window not registered (call tensor_register "
        "first)");
  }
}

void TorchCommWindowNCCL::checkDeviceAndThrow(const at::Tensor& tensor) const {
  auto data_device_type = tensor.device().type();
  if (comm_device_.type() == at::kCUDA && data_device_type == at::kCUDA) {
    auto data_device_idx = tensor.device().index();
    if (comm_device_.index() != data_device_idx) {
      throw std::runtime_error(fmt::format(
          "[TorchCommWindowNCCL]: Device mismatch: torchcomm on device {}, "
          "tensor on device {}",
          comm_device_.index(),
          data_device_idx));
    }
  }
}

void TorchCommWindowNCCL::checkRequestSizeAndThrow(size_t input_size) const {
  if (input_size > win_size_) {
    throw std::runtime_error(fmt::format(
        "[TorchCommWindowNCCL]: Requested size ({} bytes) exceeds the "
        "window size ({} bytes)",
        input_size,
        win_size_));
  }
}

} // namespace torch::comms
