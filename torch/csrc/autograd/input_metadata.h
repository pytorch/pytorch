#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/DimVector.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/SmallVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

#include <cstdint>

namespace torch { namespace autograd {

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python dispatch key is set (tensor subclass),
 * and, where applicable, the stream the corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(const at::TensorOptions options, c10::SymIntArrayRef shape, bool is_tensor_subclass)
  : options_{options}, shape_{shape.begin(), shape.end()}, is_tensor_subclass_{is_tensor_subclass} {
    auto device_ = options.device();
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
  : InputMetadata(t.options(), t.sym_sizes(), t.unsafeGetTensorImpl()->is_python_dispatch()) { }

  const at::TensorOptions options() const {
    return options_;
  }

  c10::SymIntArrayRef shape() const {
    return c10::SymIntArrayRef(shape_.data(), shape_.size());
  }

  caffe2::TypeMeta dtype() const {
    return options_.dtype();
  }

  at::Device device() const {
    return options_.device();
  }

  at::Layout layout() const {
    return options_.layout();
  }

  c10::Stream stream() const {
    return stream_;
  }

  bool is_tensor_subclass() const {
    return is_tensor_subclass_;
  }

  at::Tensor zeros_like() const {
    // TODO: add the SymInt at::zeros overload
    return at::zeros(asIntArrayRefSlow(shape()), options_);
  }

private:
  const at::TensorOptions options_;
  c10::SmallVector<c10::SymInt, c10::kDimVectorStaticSize> shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
};

}} // torch::autograd