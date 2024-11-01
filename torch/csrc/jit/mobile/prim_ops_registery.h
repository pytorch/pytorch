#pragma once

#include <ATen/core/ivalue.h>
#include <functional>
#include <vector>

namespace torch::jit::mobile {

using Stack = std::vector<c10::IValue>;

void registerPrimOpsFunction(
    const std::string& name,
    const std::function<void(Stack&)>& fn);

bool hasPrimOpsFn(const std::string& name);

std::function<void(Stack&)>& getPrimOpsFn(const std::string& name);

class prim_op_fn_register {
 public:
  prim_op_fn_register(
      const std::string& name,
      const std::function<void(Stack&)>& fn) {
    registerPrimOpsFunction(name, fn);
  }
};

} // namespace torch::jit::mobile
