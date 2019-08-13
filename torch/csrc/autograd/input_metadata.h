#pragma once

#include <ATen/ATen.h>
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "c10/core/Stream.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"

#include <cstdint>

namespace torch { namespace autograd {

/**
 * Records type, shape, and device of tensor and, where applicable,
 * the stream the correspondingoperation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(const at::DeprecatedTypeProperties& type, at::IntArrayRef shape, at::Device device)
  : type_{&type}, shape_{shape}, device_{device} {
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
  : InputMetadata(t.type(), t.sizes(), t.device()) { }

  bool is_valid() const {
    return type_ != nullptr;
  }

  const at::DeprecatedTypeProperties& type() const {
    AT_ASSERT(type_);
    return *type_;
  }

  at::IntArrayRef shape() const {
    return shape_;
  }

  at::Device device() const {
    return device_;
  }

  c10::Stream stream() const {
    return stream_;
  }

  at::Tensor zeros_like() const {
    return at::zeros(shape_, type_->options(device_));
  }

private:
  const at::DeprecatedTypeProperties* type_ = nullptr;
  at::DimVector shape_;
  at::Device device_ = at::kCPU;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device_);
};

}} // torch::autograd
