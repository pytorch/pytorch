#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API const std::string& GetSerializedShapeFunctions();

TORCH_API const OperatorMap<std::string>& GetShapeFunctionMappings();

TORCH_API const OperatorMap<std::pair<std::string, std::string>>&
GetBoundedShapeMappings();

} // namespace torch::jit
