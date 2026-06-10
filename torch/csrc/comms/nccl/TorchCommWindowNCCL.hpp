// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <nccl.h> // @manual=fbsource//third-party/nccl:nccl

#include <torch/csrc/comms/TorchCommWindow.hpp>
#include <torch/csrc/comms/nccl/NcclApi.hpp>

namespace torch::comms {

class TorchCommNCCL;

// Host-side window backed by NCCL 2.29+ RMA APIs (ncclPutSignal/ncclSignal/
// ncclWaitSignal). Zero-copy: relies on the NCCL mempool hook
// (NcclCachingAllocatorHookImpl + TorchCommNCCL::register_address) to register
// every allocated VMM segment as a NCCL_WIN_COLL_SYMMETRIC window. Both the
// destination tensor (passed to tensor_register / new_window) and the source
// tensors (passed to put) must therefore be allocated from the NCCL mempool —
// `torch.cuda.use_mem_pool(comm.get_mem_allocator())`.
//
// signal() / wait_signal() use sigIdx=0, ctx=0 (the only values currently
// accepted by NCCL). put() also emits a signal (ncclPutSignal cannot suppress
// it); the explicit signal() adds another increment. wait_signal() consumes
// one fresh signal per call — accumulated extras stay buffered and are
// drained on the next wait, which is safe for the patterns this class
// supports.
class TorchCommWindowNCCL : public TorchCommWindow {
 public:
  TorchCommWindowNCCL() = delete;
  explicit TorchCommWindowNCCL(std::shared_ptr<TorchCommNCCL> torchComm);
  ~TorchCommWindowNCCL() noexcept override = default;

  TorchCommWindowNCCL(const TorchCommWindowNCCL&) = delete;
  TorchCommWindowNCCL(TorchCommWindowNCCL&&) = delete;
  TorchCommWindowNCCL& operator=(const TorchCommWindowNCCL&) = delete;
  TorchCommWindowNCCL& operator=(TorchCommWindowNCCL&&) = delete;

  void tensor_register(const at::Tensor& tensor, bool owning = true) override;
  void tensor_deregister() override;

  std::shared_ptr<TorchCommWindow> clone() override;

  c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& tensor,
      int dstRank,
      size_t targetOffsetNelems,
      bool asyncOp,
      const PutOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> signal(
      int peerRank,
      bool asyncOp,
      const SignalOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> wait_signal(
      int peerRank,
      bool asyncOp,
      const WaitSignalOptions& options = {}) override;

  at::Tensor map_remote_tensor(int rank) override;
  std::shared_ptr<TorchCommWindowAttr> get_attr(int peerRank) override;

 private:
  void checkCommAndThrow() const;
  void checkWindowAndThrow() const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkRequestSizeAndThrow(size_t input_size) const;

  std::shared_ptr<TorchCommNCCL> torch_comm_;
  NcclApi* nccl_api_{nullptr};
  ncclComm_t nccl_comm_{nullptr};
  at::Device comm_device_{at::kCUDA};

  // Destination window for this rank — looked up from the mempool's segment
  // registration table. `peer_win_offset_` is the byte offset of the user's
  // tensor within the segment's window.
  ncclWindow_t win_{nullptr};
  size_t peer_win_offset_{0};
};

} // namespace torch::comms
