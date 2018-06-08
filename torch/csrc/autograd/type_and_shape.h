#pragma once

#include <ATen/ATen.h>
#include "torch/csrc/assertions.h"

namespace torch { namespace autograd {

/// A tensor's type and shape. Each Function records the required type and
/// shape of its inputs. If is_valid() is false, then the corresponding input
/// is not used and may be an undefined tensor.
struct TypeAndShape {
  TypeAndShape() : type_(nullptr) {}

  TypeAndShape(const at::Type& type, at::IntList shape)
    : type_(&type) , shape_(shape) {}

  bool is_valid() const {
    return type_ != nullptr;
  }

  const at::Type& type() const {
    TORCH_ASSERT(type_);
    return *type_;
  }

  at::IntList shape() const {
    return shape_;
  }

  const at::Type* type_;
  at::DimVector shape_;
};

}}
