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

  InputMetadata(const at::TensorOptions options, at::IntArrayRef shape, at::Device device, bool is_lazy)
  : options_{options}, shape_{shape}, device_{device}, is_lazy_(is_lazy) {
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
  : InputMetadata(t.options(), t.sizes(), t.device(), t.is_lazy()) { }

  const at::TensorOptions options() const {
    return options_;
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
    return at::zeros(shape_, options_);
  }

  bool is_lazy() const {
    return is_lazy_;
  }

private:
  const at::TensorOptions options_;
  at::DimVector shape_;
  at::Device device_ = at::kCPU;
  bool is_lazy_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device_);
};

}} // torch::autograd
