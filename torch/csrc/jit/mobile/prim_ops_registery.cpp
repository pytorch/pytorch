#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>

namespace torch {
namespace jit {
namespace mobile {

std::unordered_map<std::string, std::function<void(Stack&)>>& primOpsFnTable() {
  static std::unordered_map<std::string, std::function<void(Stack&)>>
      prim_ops_fn;
  return prim_ops_fn;
}

void registerPrimOpsFunction(
    const std::string& name,
    const std::function<void(Stack&)>& fn) {
  primOpsFnTable()[name] = fn;
}

bool hasPrimOpsFn(const std::string& name) {
  return primOpsFnTable().count(name);
}

std::function<void(Stack&)>& getPrimOpsFn(const std::string& name) {
  TORCH_CHECK(
      hasPrimOpsFn(name),
      "Prim Ops Function for ",
      name,
      " is not promoted yet.");
  return primOpsFnTable()[name];
}

void add_functions() {
  // TODO: (@pavithran)
  // to remove schema argument if it is not going to be used
  registerPrimOpsFunction("prim::TupleIndex", tupleIndex);

  registerPrimOpsFunction("aten::Bool.Tensor", boolTensor);

  registerPrimOpsFunction("aten::format", aten_format);

  registerPrimOpsFunction("prim::NumToTensor.Scalar", numToTensorScalar);

  registerPrimOpsFunction("prim::RaiseException", raiseException);

  // TODO: (@pavithran) size is overloaded with int[] and Tensor
  // so this throws error expecting int not Tensor
  // registerPrimOpsFunction(
  //     "aten::size", size);

  registerPrimOpsFunction("prim::device", device);

  registerPrimOpsFunction("prim::dtype", dtype);

  registerPrimOpsFunction("aten::__not__", _not);

  registerPrimOpsFunction("aten::__is__", is);

  registerPrimOpsFunction("aten::__isnot__", isNot);

  registerPrimOpsFunction("aten::dim", dim);

  registerPrimOpsFunction("prim::Uninitialized", unInitialized);

  registerPrimOpsFunction("aten::to.prim_dtype", toPrimDType);

  registerPrimOpsFunction("prim::is_cuda", isCuda);
}

} // namespace mobile
} // namespace jit
} // namespace torch
