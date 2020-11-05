#pragma once

#include <ATen/core/Dimname.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/C++17.h>

namespace at {

static inline bool is_nested_tensor_impl(at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::NestedTensor) ||
      tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::AutogradNestedTensor);
}

}
