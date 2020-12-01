#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <cstdint>

namespace torch {
namespace autograd {

/**
 * Records type, shape, and device of tensor and, where applicable,
 * the stream the correspondingoperation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(
      const at::TensorOptions options,
      c10::optional<std::vector<int64_t>> shape,
      at::Device device,
      bool is_nested_tensor,
      at::Tensor nested_size_tensor)
      : options_{options},
        shape_{shape},
        device_{device},
        is_nested_tensor_(is_nested_tensor),
        nested_size_(
            nested_size_tensor.data_ptr<int64_t>(),
            nested_size_tensor.data_ptr<int64_t>() + 
                     nested_size_tensor.numel()) {
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
      : InputMetadata(
            t.options(),
            at::is_nested_tensor_impl(t) 
              ? c10::nullopt
              : c10::optional<std::vector<int64_t>>(t.sizes().vec()),
            t.device(),
            at::is_nested_tensor_impl(t),
            at::serialize_nested_size(t)) {}

  const at::TensorOptions options() const {
    return options_;
  }

  at::IntArrayRef shape() const {
    TORCH_CHECK(!is_nested_tensor(), "NestedTensor doesn't have a shape.");
    TORCH_CHECK(shape_, "internal error: Expected non-NestedTensor to have shape.");
    return at::IntArrayRef(*shape_);
  }

  at::IntArrayRef nested_size() const {
    TORCH_CHECK(is_nested_tensor(), "Only available to NestedTensors.");
    return at::IntArrayRef(nested_size_);
  }

  at::Device device() const {
    return device_;
  }

  c10::Stream stream() const {
    return stream_;
  }

  at::Tensor zeros_like() const {
    TORCH_CHECK(shape_, "zeros_like is only supported if not a NestedTensor.");
    return at::zeros(*shape_, options_);
  }

  bool is_nested_tensor() const {
    return is_nested_tensor_;
  }

 private:
  const at::TensorOptions options_;
  c10::optional<std::vector<int64_t>> shape_;
  at::Device device_ = at::kCPU;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device_);
  bool is_nested_tensor_;
  std::vector<int64_t> nested_size_;
};

} // namespace autograd
} // namespace torch
