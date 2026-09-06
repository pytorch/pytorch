// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <optional>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include <torch/csrc/distributed/c10d/Window.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>

namespace c10d::nccl2 {

class ProcessGroupNCCL;

// One-sided window backed by NCCL 2.29+ RMA APIs (ncclPutSignal/ncclSignal/
// ncclWaitSignal). Port of torchcomms' TorchCommWindowNCCL rebased onto
// c10d::Window. Zero-copy: relies on the NCCL mempool hook
// (NCCLCachingAllocatorHook + ProcessGroupNCCL::register_address) to register
// every allocated segment. tensor_register / new_window collectively upgrades
// the destination segment to a symmetric window. Source tensors passed to put
// must already reside in a collectively registered symmetric segment.
//
// signal() / wait_signal() use sigIdx=0, ctx=0 (the only values currently
// accepted by NCCL). put() also emits a signal (ncclPutSignal cannot suppress
// it); the explicit signal() adds another increment. wait_signal() consumes
// one fresh signal per call -- accumulated extras stay buffered and are
// drained on the next wait.
class WindowNCCL : public ::c10d::Window {
 public:
  explicit WindowNCCL(c10::intrusive_ptr<ProcessGroupNCCL> pg);
  ~WindowNCCL() override = default;

  WindowNCCL(const WindowNCCL&) = delete;
  WindowNCCL(WindowNCCL&&) = delete;
  WindowNCCL& operator=(const WindowNCCL&) = delete;
  WindowNCCL& operator=(WindowNCCL&&) = delete;

  void tensor_register(const at::Tensor& tensor, bool owning = true) override;
  void tensor_deregister() override;

  c10::intrusive_ptr<::c10d::Work> put(
      const at::Tensor& tensor,
      int64_t dstRank,
      int64_t targetOffsetNelems,
      bool asyncOp,
      const ::c10d::PutOptions& opts = ::c10d::PutOptions()) override;
  c10::intrusive_ptr<::c10d::Work> signal(
      int64_t peerRank,
      bool asyncOp,
      const ::c10d::SignalOptions& opts = ::c10d::SignalOptions()) override;
  c10::intrusive_ptr<::c10d::Work> wait_signal(
      int64_t peerRank,
      bool asyncOp,
      const ::c10d::WaitSignalOptions& opts =
          ::c10d::WaitSignalOptions()) override;

  at::Tensor map_remote_tensor(int64_t rank) override;
  ::c10d::WindowAttr get_attr(int64_t peerRank) override;

 private:
  void checkWindowAndThrow() const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkPeerRankAndThrow(int64_t rank) const;
  void exchangePeerMetadata(
      size_t local_offset,
      size_t local_size,
      at::ScalarType local_dtype);

  c10::intrusive_ptr<ProcessGroupNCCL> pg_;
  NcclApi* nccl_api_{nullptr};
  ncclComm_t nccl_comm_{nullptr};

  // Destination window for this rank and each peer's logical tensor metadata.
  // The NCCL window is collective, but tensor offsets and sizes may differ.
  ncclWindow_t win_{nullptr};
  std::vector<size_t> peer_win_offsets_;
  std::vector<size_t> peer_win_sizes_;
  std::vector<at::ScalarType> peer_dtypes_;
  std::optional<at::Tensor> buf_tensor_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
