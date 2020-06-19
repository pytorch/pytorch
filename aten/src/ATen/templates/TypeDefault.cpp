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

namespace {
static const char* named_tensors_unsupported_error =
  " is not yet supported with named tensors. Please drop names via "
  "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
  "and set names on the result of the operation.";
}

namespace at {
namespace TypeDefault {

${type_method_definitions}

}  // namespace TypeDefault

TORCH_LIBRARY(aten, m) {
  ${function_registrations};

  // String Ops
  // Implementations located in torch/csrc/jit/runtime/register_string_ops.cpp
  m.def("splitlines(str self, bool keepends=False) -> str[]");
  m.def(
      "slice.str(str string, int start, int end=9223372036854775807, int step=1) -> str");
  m.def("isupper(str self) -> bool");
  m.def("islower(str self) -> bool");
  m.def("capitalize(str self) -> str");
  m.def("title(str self) -> str");
  m.def("center(str self, int width, str fillchar=' ') -> str");
  m.def("count(str self, str substr, int start=0, int end=-1) -> int");
  m.def("endswith(str self, str substr, int start=0, int end=-1) -> bool");
  m.def("startswith(str self, str substr, int start=0, int end=-1) -> bool");
  m.def("expandtabs(str self, int tabsize=8) -> str");
  m.def("find(str self, str substr, int start=0, int end=-1) -> int");
  m.def("rfind(str self, str substr, int start=0, int end=-1) -> int");
  m.def("index.str(str self, str substr, int start=0, int end=-1) -> int");
  m.def("rindex(str self, str substr, int start=0, int end=-1) -> int");
  m.def("isidentifier(str self) -> bool");
  m.def("istitle(str self) -> bool");
  m.def("isprintable(str self) -> bool");
  m.def("ljust(str self, int width, str fillchar=' ') -> str");
  m.def("rjust(str self, int width, str fillchar=' ') -> str");
  m.def("zfill(str self, int width) -> str");
  m.def("lstrip(str self, str chars=' \\n\\t\\f\\v') -> str");
  m.def("rstrip(str self, str chars=' \\n\\t\\f\\v') -> str");
  m.def("strip(str self, str chars=' \\n\\t\\f\\v') -> str");
  m.def("replace(str self, str old, str new, int max=-1) -> str");
  m.def("partition(str self, str separator) -> (str, str, str)");
  m.def("rpartition(str self, str separator) -> (str, str, str)");
  m.def("split.str(str self, str? separator=None, int max=-1) -> str[]");
  m.def("rsplit(str self, str separator=' ', int max=-1) -> str[]");
  m.def("join(str self, str[] values) -> str");

  // Integer Ops
  // Implementations located in torch/csrc/jit/runtime/register_prim_ops_c10.cp
  m.def("Int.Tensor(Tensor a) -> int");
  m.def("Int.bool(bool a) -> int");
  m.def("Int.float(float a) -> int");
  m.def("Int.Scalar(Scalar a) -> int");
  m.def("Int.str(str a) -> int");

  // Distributed Ops
  // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
}

}  // namespace at
