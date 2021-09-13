#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>

namespace torch {
namespace jit {
namespace mobile {

std::unordered_map<std::string,
    std::pair<std::string, std::function<void(Stack&)>>>& primOpsFnTable() {
  static std::unordered_map<std::string,
                            std::pair<std::string, std::function<void(Stack&)>>>
      prim_ops_fn;
  return prim_ops_fn;
}

void registerPrimOpsFunction(
    const std::string& name,
    const std::string& schema,
    const std::function<void(Stack&)>& fn) {
  primOpsFnTable()[name] = {schema, fn};
}

bool hasPrimOpsFn(const std::string& name) {
  return primOpsFnTable().count(name);
}

std::pair<std::string, std::function<void(Stack&)>>& getPrimOpsFn(const std::string& name) {
  TORCH_CHECK(
      hasPrimOpsFn(name),
      "Prim Ops Function for ",
      name,
      " is not promoted yet.");
  return primOpsFnTable()[name];
}

void add_functions() {
  registerPrimOpsFunction("prim::TupleIndex","prim::TupleIndex(Any tup, int i) -> Any", tupleIndex);
}

} // namespace mobile
} // namespace jit
} // namespace torch
