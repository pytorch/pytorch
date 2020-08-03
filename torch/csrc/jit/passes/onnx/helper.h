#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch {
namespace jit {
using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

using ParamMap = std::map<std::string, IValue>;

ValueToParamPairMap buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);

} // namespace jit
} // namespace torch
