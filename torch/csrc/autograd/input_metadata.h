#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace torch { namespace autograd {

// This legacy function squashes a tensor's Device into an integer
// index which identifies which autograd thread it refers to.
inline int assign_tensor_to_autograd_thread(const at::Tensor& tensor) {
  if (tensor.is_cpu()) {
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
  // NB: This looks like a "device" but it actually is an assignment to
  // an autograd thread; e.g., both CUDA device 1 and XLA device 1 live
  // on the same thread.  We should change this to *actually* be a Device
  // but we have to fix all of the sites to adjust this way.  (Why is
  // it this way?  In the old days we only had CUDA devices, and so
  // we could just say -1 was CPU and everything else was CUDA.  This clearly
  // doesn't hold anymore.)
  const int64_t device_ = -1;
};

}}
