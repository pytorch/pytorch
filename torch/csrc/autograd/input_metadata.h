#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

namespace torch::autograd {

using SymIntSmallVec = c10::SmallVector<c10::SymInt, c10::kDimVectorStaticSize>;
using MetadataShape = std::variant<SymIntSmallVec, at::Tensor>;

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python
 * dispatch key is set (tensor subclass), and, where applicable, the stream the
 * corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct TORCH_API InputMetadata {
  InputMetadata() = default;
  InputMetadata(
      const at::TensorOptions& options,
      MetadataShape input_shape,
      bool is_tensor_subclass,
      bool is_nested);
  InputMetadata(const at::Tensor& t);

  const at::TensorOptions& options() const {
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

  at::Tensor zeros_like() const;

  bool is_same_shape(const at::Tensor& grad) const;

  bool is_expandable_to_shape(const at::Tensor& grad) const;

  at::Tensor reduce_grad(at::Tensor& grad) const;

  at::Tensor maybe_reduce(
      const size_t index,
      at::Tensor grad,
      const std::function<std::string(const std::string&)>& format_error) const;

  std::stringstream incompatible_shape_error_message(
      const size_t index,
      const at::Tensor& grad) const;

  bool was_default_constructed() const {
    return was_default_constructed_;
  }

  bool is_cpp_nested_tensor() const;

  bool is_nested_tensor() const {
    return is_nested_;
  }

  c10::SymIntArrayRef shape_as_dim_vector() const;

  // Danger: not thread safe, caller must protect with lock
  SymIntSmallVec& mutable_shape_as_dim_vector();

 private:
  at::Tensor shape_as_tensor() const;
  bool is_nestedness_same(const at::Tensor& grad) const;
  bool maybe_expandable_to(const at::Tensor& grad) const;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  at::TensorOptions options_;
  MetadataShape shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
  bool is_nested_ = false;
  bool was_default_constructed_ = true;
};
} // namespace torch::autograd
