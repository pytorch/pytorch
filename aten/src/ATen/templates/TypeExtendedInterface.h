#pragma once
#include <ATen/core/Tensor.h>

namespace at {

struct CAFFE2_API TypeExtendedInterface : public Type {
  explicit TypeExtendedInterface()
      : Type() {}
};

} // namespace at
