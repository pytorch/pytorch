#pragma once

#include <cstdint>
#include <optional>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

// Memory model exposed by a one-sided window.
// UNIFIED: a single coherent view of the window across host and device.
// SEPARATE: distinct public/private copies requiring explicit synchronization.
enum class WindowAccessType : uint8_t {
  UNIFIED = 0,
  SEPARATE = 1,
};

struct WindowAttr {
  WindowAccessType access_type = WindowAccessType::UNIFIED;
};

// Window - one-sided (RMA) communication handle.
//
// Backends that support one-sided operations return a concrete subclass from
// Backend::new_window(). This base class is an interface only: every method
// defaults to throwing so that backends can override just the operations they
// support and unsupported calls fail loudly.
class TORCH_API Window : public torch::CustomClassHolder {
 public:
  ~Window() override = default;

  // tensor_register / tensor_deregister are collective: all ranks must call
  // them together. When owning is true the window keeps the tensor's storage
  // alive for the window's lifetime.
  virtual void tensor_register(
      const at::Tensor& /* tensor */,
      bool /* owning */ = true) {
    TORCH_CHECK(false, "This window does not support tensor_register");
  }

  virtual void tensor_deregister() {
    TORCH_CHECK(false, "This window does not support tensor_deregister");
  }

  // Write tensor into the window of dstRank at the given element offset.
  virtual c10::intrusive_ptr<Work> put(
      const at::Tensor& /* tensor */,
      int64_t /* dstRank */,
      int64_t /* targetOffsetNelems */,
      bool /* asyncOp */,
      const PutOptions& /* opts */ = PutOptions()) {
    TORCH_CHECK(false, "This window does not support put");
  }

  // Returns a tensor view of a peer rank's window memory, if mappable.
  virtual at::Tensor map_remote_tensor(int64_t /* rank */) {
    TORCH_CHECK(false, "This window does not support map_remote_tensor");
  }

  // signal / wait_signal implement point-to-point synchronization over the
  // window.
  virtual c10::intrusive_ptr<Work> signal(
      int64_t /* peerRank */,
      bool /* asyncOp */,
      const SignalOptions& /* opts */ = SignalOptions()) {
    TORCH_CHECK(false, "This window does not support signal");
  }

  virtual c10::intrusive_ptr<Work> wait_signal(
      int64_t /* peerRank */,
      bool /* asyncOp */,
      const WaitSignalOptions& /* opts */ = WaitSignalOptions()) {
    TORCH_CHECK(false, "This window does not support wait_signal");
  }

  virtual WindowAttr get_attr(int64_t /* peerRank */) {
    TORCH_CHECK(false, "This window does not support get_attr");
  }
};

} // namespace c10d
