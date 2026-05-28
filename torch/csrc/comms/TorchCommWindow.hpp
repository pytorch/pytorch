// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/comms/TorchCommOptions.hpp>
#include <torch/csrc/comms/TorchCommTypes.hpp>
namespace torch::comms {

// Forward declaration
class TorchWork;

enum class TorchCommWinAccessType {
  WIN_ACCESS_TYPE_UNIFIED = 0,
  WIN_ACCESS_TYPE_SEPARATE = 1,
};

class TorchCommWindowAttr {
 public:
  TorchCommWinAccessType accessType;
};

class TorchCommWindow {
 public:
  TorchCommWindow() = default;
  virtual ~TorchCommWindow() = default;

  // Disable copy and move semantics
  TorchCommWindow(const TorchCommWindow&) = delete;
  TorchCommWindow& operator=(const TorchCommWindow&) = delete;
  TorchCommWindow(TorchCommWindow&&) = delete;
  TorchCommWindow& operator=(TorchCommWindow&&) = delete;

  // tensor_register and tensor_deregister are collective operations - all
  // ranks must call them together.
  //
  // When owning=true (default), the window holds a reference to the tensor,
  // keeping its storage alive. When owning=false, the window does NOT hold a
  // reference — the caller must ensure the tensor remains alive for the
  // window's lifetime. Use owning=false in CUDA graph capture mode to allow
  // tensor memory reuse within the graph.
  virtual void tensor_register(
      const at::Tensor& tensor,
      bool owning = true) = 0;
  virtual void tensor_deregister() = 0;

  // Creates a new window with the same backend/comm configuration.
  // If a tensor is registered, the clone will have a cloned tensor registered.
  virtual std::shared_ptr<TorchCommWindow> clone() = 0;

  // APIs exposed to users
  virtual c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& tensor,
      int dstRank,
      size_t targetOffsetNelems,
      bool asyncOp,
      const PutOptions& options = {}) = 0;
  virtual at::Tensor map_remote_tensor(int rank) = 0;
  virtual c10::intrusive_ptr<TorchWork> signal(
      int peerRank,
      bool asyncOp,
      const SignalOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> wait_signal(
      int peerRank,
      bool asyncOp,
      const WaitSignalOptions& options = {}) = 0;

  virtual std::shared_ptr<TorchCommWindowAttr> get_attr(int peerRank) = 0;

  // Get the registered buffer's dtype (for torch.compile meta kernel)
  at::ScalarType getDtype() const {
    return buf_dtype_;
  }

  // Get the registered buffer's shape (for torch.compile meta kernel)
  std::vector<int64_t> getShape() const {
    return buf_shape_;
  }

  // Get the registered buffer's device (for torch.compile meta kernel)
  c10::Device getDevice() const {
    return buf_device_;
  }

  size_t get_size() const {
    return win_size_;
  }

  // Returns the registered tensor buffer, or nullopt if owning=false was used.
  std::optional<at::Tensor> get_tensor() const {
    return buf_tensor_;
  }

 protected:
  // device_: The device where the window is allocated.
  //  The device where the window is allocated may differ from the device used
  //  by the communicator. For example, the window could be allocated on the CPU
  //  while the communicator operates on the GPU. However, if both are using the
  //  GPU, they should reside on the same device.
  size_t win_size_{0};
  // Holds a reference to the registered tensor to keep its storage alive.
  // Nullopt when owning=false was passed to tensor_register().
  std::optional<at::Tensor> buf_tensor_;
  at::ScalarType buf_dtype_{at::kFloat};
  c10::Device buf_device_{c10::kCUDA};
  // Cached buffer shape to avoid repeated calls to tensor.sizes()
  std::vector<int64_t> buf_shape_;
};

} // namespace torch::comms
