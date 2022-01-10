#include <ATen/TypeDefault.h>

// ${generated_comment}

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>
#include <torch/library.h>

namespace at {
TORCH_LIBRARY(aten, m) {
  ${schema_registrations};

  // Distributed Ops
  // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
}
}  // namespace at
