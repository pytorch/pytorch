// ${generated_comment}
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/library.h>

namespace at {
TORCH_LIBRARY(aten, m) {
  ${aten_schema_registrations};
  // Distributed Ops
  // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
}
${schema_registrations}
}  // namespace at
