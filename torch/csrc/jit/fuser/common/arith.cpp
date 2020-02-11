#include <c10/util/Exception.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/type.h>

namespace torch {
namespace jit {
namespace fuser {
// Return new value of type that v1 and v2 promotes to
TORCH_API Val* promote_new(Val* v1, Val* v2) {
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  switch (out_vtype) {
    case (ValType::Tensor):
      return new Tensor(out_dtype); // TODO add dtype here.
    case (ValType::Scalar):
      switch (out_dtype) {
        case (DataType::Float):
          return new Float();
        case (DataType::Int):
          return new Int();
        default:
          break;
      }
    default:
      break;
  }
  std::runtime_error("Did not recognize out type.");
  return new Int(-1);
}

TORCH_API Val* add(Val* v1, Val* v2) {
  Val* out = promote_new(v1, v2);
  Statement* expr = new Add(out, v1, v2);
  return out;
}

} // namespace fuser
} // namespace jit
} // namespace torch
