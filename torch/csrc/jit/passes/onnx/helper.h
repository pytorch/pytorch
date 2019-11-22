#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

#include <memory>

namespace torch {
namespace jit {
using ValueToParamPairMap =
    std::map<Value*, std::pair<std::string, at::Tensor>>;

using ParamMap = std::map<std::string, at::Tensor>;

ValueToParamPairMap buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);

} // namespace jit
} // namespace torch
