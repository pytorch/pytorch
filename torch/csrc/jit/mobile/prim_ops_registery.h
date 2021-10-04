#pragma once

#include <ATen/core/ivalue.h>
#include <vector>

namespace torch {
namespace jit {
namespace mobile {

using Stack = std::vector<c10::IValue>;
using StackFn = void (*)(Stack&);

void registerPrimOpsFunction(const std::string& name, const StackFn& fn);

bool hasPrimOpsFn(const std::string& name);

StackFn& getPrimOpsFn(const std::string& name);

class prim_op_fn_register {
 public:
  prim_op_fn_register(const std::string& name, const StackFn fn) {
    registerPrimOpsFunction(name, fn);
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
