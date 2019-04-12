#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace torch { namespace autograd {

/// A tensor's type and shape. Each Function records the required type and
/// shape of its inputs. If is_valid() is false, then the corresponding input
/// is not used and may be an undefined tensor.
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(const at::Type& type, at::IntArrayRef shape, at::Device device)
  : type_{&type} , shape_{shape}, device_{device} { }

  InputMetadata(const at::Tensor& t)
  : InputMetadata(t.dispatch_type(), t.sizes(), t.device()) { }

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

  at::Device device() const {
    return device_;
  }

  at::Tensor zeros_like() const {
    return at::zeros(shape_, type_->options(device_));
  }

private:
  const at::Type* type_ = nullptr;
  at::DimVector shape_;
  at::Device device_ = at::kCPU;
};

}}
