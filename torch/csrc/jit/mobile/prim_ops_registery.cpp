#include <torch/csrc/jit/mobile/prim_ops_registery.h>

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

} // namespace mobile
} // namespace jit
} // namespace torch
