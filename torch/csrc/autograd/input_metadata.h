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
#include <c10/util/variant.h>

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
using MetadataShape = std::variant<SymIntSmallVec, at::Tensor>;

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
      const at::TensorOptions& options,
      MetadataShape input_shape,
      bool is_tensor_subclass,
      bool is_nested)
      : options_{options},
        shape_{std::move(input_shape)},
        is_tensor_subclass_{is_tensor_subclass},
        is_nested_{is_nested},
        was_default_constructed_{false} {
    auto device_ = options.device();
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
      : InputMetadata(
            t.options(),
            compute_variant_shape(t),
            t.unsafeGetTensorImpl()->is_python_dispatch(),
            t.is_nested()) {}

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

  at::Tensor zeros_like() const {
    TORCH_CHECK(
        !is_nested_, "Zeros is not currently supported for nested tensors.")
    return at::zeros_symint(shape_as_dim_vector(), options_);
  }

  void check_nestedness_same(const at::Tensor& grad) const {
    bool grad_is_subclass = grad.unsafeGetTensorImpl()->is_python_dispatch();
    bool grad_is_nested = grad.is_nested();
    bool grad_is_cpp_nested = grad_is_nested && !grad_is_subclass;
    TORCH_CHECK(
        grad_is_cpp_nested == is_cpp_nested_tensor() &&
            grad_is_nested == is_nested_,
        "grad and the input wrt the gradient that is being computed for need to be "
        "either both nested or both non-nested tensors. Also note that nested "
        "tensors with different layouts do not compose currently.");
  }

  bool is_same_shape(const at::Tensor& grad) const {
    check_nestedness_same(grad);
    if (is_cpp_nested_tensor()) {
      return grad._nested_tensor_size().is_same_size(shape_as_tensor());
    }
    return grad.sym_sizes().equals(shape_as_dim_vector());
  }
  bool is_expandable_to_shape(const at::Tensor& grad) const {
    // Currently NestedTensors are not expandable. If this support is added then
    // updates to reduce_grad will be needed
    check_nestedness_same(grad);
    return grad.is_nested()
        ? false
        : at::is_expandable_to(shape_as_dim_vector(), grad.sym_sizes());
  }

  at::Tensor reduce_grad(at::Tensor& grad) const {
    // Currently reduce_grad is only called if is_expandable_to_shape returns
    // true For nested tensors this always returns False, so this check
    // shouldn't fail
    TORCH_INTERNAL_ASSERT(!grad.is_nested() && !is_nested_)
    return at::sum_to(std::move(grad), shape_as_dim_vector());
  }

  std::stringstream incompatible_shape_error_message(
      const size_t index,
      const at::Tensor& grad) const {
    std::stringstream ss;
    ss << "invalid gradient at index " << index << " - got ";
    if (grad.is_nested() && !grad.unsafeGetTensorImpl()->is_python_dispatch()) {
      ss << grad._nested_tensor_size();
    } else {
      ss << grad.sym_sizes();
    }
    ss << " but expected shape compatible with ";
    if (is_cpp_nested_tensor()) {
      ss << shape_as_tensor();
    } else {
      ss << shape_as_dim_vector();
    }
    return ss;
  }

  bool was_default_constructed() const {
    return was_default_constructed_;
  }

  bool is_cpp_nested_tensor() const {
    bool ret = std::holds_alternative<at::Tensor>(shape_);
    TORCH_INTERNAL_ASSERT(ret == (is_nested_ && !is_tensor_subclass_))
    return ret;
  }

  bool is_nested_tensor() const {
    return is_nested_;
  }

  c10::SymIntArrayRef shape_as_dim_vector() const {
    const auto& dim_shape = std::get<SymIntSmallVec>(shape_);
    return c10::SymIntArrayRef(dim_shape.data(), dim_shape.size());
  }

  // Danger: not thread safe, caller must protect with lock
  SymIntSmallVec& mutable_shape_as_dim_vector() {
    return std::get<SymIntSmallVec>(shape_);
  }

 private:
  MetadataShape compute_variant_shape(const at::Tensor& input) {
    if (input.is_nested() &&
        !input.unsafeGetTensorImpl()->is_python_dispatch()) {
      auto nested_size = input._nested_tensor_size();
      return MetadataShape{std::in_place_type<at::Tensor>, nested_size};
    }
    return MetadataShape{std::in_place_type<SymIntSmallVec>, input.sym_sizes()};
  }

  at::Tensor shape_as_tensor() const {
    return std::get<at::Tensor>(shape_);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::TensorOptions options_;
  MetadataShape shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
  bool is_nested_ = false;
  bool was_default_constructed_ = true;
};
} // namespace autograd
} // namespace torch
