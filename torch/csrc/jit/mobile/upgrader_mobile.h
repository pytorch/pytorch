#pragma once

// #include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
struct Instruction;
struct Upgrader {
  int min_version;
  int max_version;
  std::string upgrader_name;
  int index;
};

// From operator_versions.yaml
TORCH_API const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile();

struct OperatorString {
  const std::string name;
  const std::string overload_name;
  const c10::optional<int> num_specified_args;
};

struct ByteCodeFunctionWithOperator {
  mobile::Function& function;
  std::vector<OperatorString> operators;
};

TORCH_API const std::vector<ByteCodeFunctionWithOperator>&
getUpgraderBytecodeList();

} // namespace jit
} // namespace torch
