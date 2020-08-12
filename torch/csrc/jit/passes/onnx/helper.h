#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch {
namespace jit {

namespace onnx {
static const int OPSET_VERSION_1 = 1;
static const int OPSET_VERSION_9 = 9;
static const int OPSET_VERSION_10 = 10;
static const int OPSET_VERSION_11 = 11;
static const int OPSET_VERSION_12 = 12;
} // namespace onnx

using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

using ParamMap = std::map<std::string, IValue>;

ValueToParamPairMap buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);
void eraseUnusedBlockInputs(Block* b);
void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

} // namespace jit
} // namespace torch
