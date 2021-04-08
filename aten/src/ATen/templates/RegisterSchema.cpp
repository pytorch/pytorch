#include <ATen/TypeDefault.h>

// ${generated_comment}

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/SparseTensorUtils.h>
#include <torch/library.h>

namespace at {
TORCH_LIBRARY(aten, m) {
  ${schema_registrations};

  // String Ops
  // Implementations located in torch/csrc/jit/runtime/register_prim_ops.cpp
  m.def(TORCH_SELECTIVE_SCHEMA("aten::splitlines(str self, bool keepends=False) -> str[]"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "aten::slice.str(str string, int? start=0, int? end=9223372036854775807, int step=1) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::isupper(str self) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::islower(str self) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::capitalize(str self) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::title(str self) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::center(str self, int width, str fillchar=' ') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::count(str self, str substr, int start=0, int end=-1) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::endswith(str self, str substr, int start=0, int end=-1) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::startswith(str self, str substr, int start=0, int end=-1) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::expandtabs(str self, int tabsize=8) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::find(str self, str substr, int start=0, int end=-1) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rfind(str self, str substr, int start=0, int end=-1) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::index.str(str self, str substr, int start=0, int end=-1) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rindex(str self, str substr, int start=0, int end=-1) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::isidentifier(str self) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::istitle(str self) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::isprintable(str self) -> bool"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::ljust(str self, int width, str fillchar=' ') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rjust(str self, int width, str fillchar=' ') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::zfill(str self, int width) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::lstrip(str self, str chars=' \\n\\t\\f\\v') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rstrip(str self, str chars=' \\n\\t\\f\\v') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::strip(str self, str chars=' \\n\\t\\f\\v') -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::replace(str self, str old, str new, int max=-1) -> str"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::partition(str self, str separator) -> (str, str, str)"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rpartition(str self, str separator) -> (str, str, str)"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::split.str(str self, str? separator=None, int max=-1) -> str[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::rsplit(str self, str separator=' ', int max=-1) -> str[]"));
  m.def(TORCH_SELECTIVE_SCHEMA("aten::join(str self, str[] values) -> str"));

  // Distributed Ops
  // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
}
}  // namespace at
