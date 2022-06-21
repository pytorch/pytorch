#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/variant.h>
#include <c10/util/DimVector.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/SmallVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

#include <cstdint>
#include <utility>

namespace torch {
namespace autograd {

using SymIntSmallVec = c10::SmallVector<c10::SymInt, c10::kDimVectorStaticSize>;
using MetadataShape = c10::variant<SymIntSmallVec, at::Tensor>;

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python
 * dispatch key is set (tensor subclass), and, where applicable, the stream the
 * corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(
      const at::TensorOptions options,
      MetadataShape input_shape,
      bool is_tensor_subclass)
      : options_{options},
        shape_{input_shape},
        is_tensor_subclass_{is_tensor_subclass} {
    auto device_ = options.device();
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
      : InputMetadata(
            t.options(),
            compute_variant_shape(t),
            t.unsafeGetTensorImpl()->is_python_dispatch()) {}

  const at::TensorOptions options() const {
    return options_;
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
    TORCH_CHECK(
        !is_nested_tensor(),
        "Zeros is not currently supported for nested tensors.")
    return at::zeros_symint(shape_as_dim_vector(), options_);
  }

 private:
  bool is_nested_tensor() const {
    return (c10::holds_alternative<at::Tensor>(shape_));
  }
  MetadataShape compute_variant_shape(const at::Tensor& input) {
    if (input.is_nested()) {
      auto nested_size = at::native::get_nested_size_tensor(input);
      return MetadataShape{c10::in_place_type<at::Tensor>, nested_size};
    }
    return MetadataShape{c10::in_place_type<SymIntSmallVec>, input.sym_sizes()};
  }

  c10::SymIntArrayRef shape_as_dim_vector() const {
    const auto& dim_shape = c10::get<SymIntSmallVec>(shape_);
   return c10::SymIntArrayRef(dim_shape.data(), dim_shape.size());
  }

  at::Tensor shape_as_tensor() const {
    return c10::get<at::Tensor>(shape_);
  }

  const at::TensorOptions options_;
  MetadataShape shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
};
} // namespace autograd
} // namespace torch
