#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace torch { namespace autograd {

// This legacy function squashes a tensor's Device into an integer
// index which identifies which autograd thread it refers to.
inline int assign_tensor_to_autograd_thread(const at::Tensor& tensor) {
  if (tensor.device().type() == at::DeviceType::CPU) {
    return -1;
  } else {
    return tensor.device().index();
  }
}

/// A tensor's type and shape. Each Function records the required type and
/// shape of its inputs. If is_valid() is false, then the corresponding input
/// is not used and may be an undefined tensor.
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(const at::Type& type, at::IntArrayRef shape, const int64_t device)
  : type_{&type} , shape_{shape}, device_{device} { }

  InputMetadata(const at::Tensor& t)
  : InputMetadata(t.type(), t.sizes(), assign_tensor_to_autograd_thread(t)) { }

  bool is_valid() const {
    return type_ != nullptr;
  }

  const at::Type& type() const {
    AT_ASSERT(type_);
    return *type_;
  }

  at::IntArrayRef shape() const {
    return shape_;
  }

  int64_t device() const {
    return device_;
  }

  at::Tensor zeros_like() const {
    return at::zeros(shape_, type_->options(static_cast<int32_t>(device_)));
  }

private:
  const at::Type* type_ = nullptr;
  at::DimVector shape_;
  // This should be a 'Device'.  However, it is not for hysterical raisins,
  // when we only had CPU and CUDA device (so we let -1 be CPU and other
  // numbers be CUDA).  We haven't had time to refactor this properly,
  // but we needed to support other devices like XLA, so what this really
  // is, is an approximation of the Device, but with the device type dropped.
  // This is good enough for us to assign this to autograd threads, but
  // it's not good enough for us to do completely correct checks that device
  // lines up.  When we fix this to be 'Device', you can delete this comment.
  const int64_t device_ = -1;
};

}}
