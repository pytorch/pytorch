#pragma once

#include "torch/csrc/stable/c/shim.h"

AOTI_TORCH_EXPORT AtenTensorHandle aoti_torch_empty_like(AtenTensorHandle self,
                                                         int32_t device_type,
                                                         int32_t device_index) {
  return torch_call_dispatcher("aten::empty_like", "", self, device_type, device_index);
}

AOTI_TORCH_EXPORT AtenTensorHandle aoti_torch_transpose_int(AtenTensorHandle self, int64_t dim0, int64_t dim1) {
  return torch_call_dispatcher("aten::transpose", "int", self, dim0, dim1);
}

AOTI_TORCH_EXPORT AtenTensorHandle aoti_torch_clone(AtenTensorHandle self) {
  return torch_call_dispatcher("aten::clone", "", self);
}

AOTI_TORCH_EXPORT void aoti_torch_zero_(AtenTensorHandle self) {
  torch_call_dispatcher("aten::zero_", "", self);
}

AOTI_TORCH_EXPORT void aoti_torch_copy_(AtenTensorHandle self, AtenTensorHandle src) {
  torch_call_dispatcher("aten::copy_", "", self, src);
}
