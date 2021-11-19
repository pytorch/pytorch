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
const std::unordered_map<std::string, std::vector<Upgrader>> kOperatorVersionMap(
    {{std::string("aten::div_Tensor"),
      std::vector<Upgrader>({Upgrader({0, 3, "div_Tensor_0_3", 0})})}});

struct OperatorString {
  const std::string name;
  const std::string overload_name;
  const c10::optional<int> num_specified_args;
};

struct MobileCodeData {
  std::string qualified_name;
  std::vector<Instruction> instructions;
  std::vector<OperatorString> operators;
  std::vector<c10::IValue> constants;
  std::vector<c10::TypePtr> types;
  size_t register_size;
};

static std::vector<MobileCodeData> kUpgraderByteCode({MobileCodeData({
    "div_Tensor_0_3",
    std::vector<Instruction>({
        Instruction{OpCode::STOREN, 1, 2}, Instruction{OpCode::LOAD, 1, 0},
        Instruction{OpCode::OP, 0, 0},     Instruction{OpCode::JF, 3, 0},
        Instruction{OpCode::LOADC, 1, 0},  Instruction{OpCode::JMP, 3, 0},
        Instruction{OpCode::LOAD, 2, 0},   Instruction{OpCode::OP, 0, 0},
        Instruction{OpCode::STORE, 3, 0},  Instruction{OpCode::MOVE, 3, 0},
        Instruction{OpCode::JF, 5, 0},     Instruction{OpCode::LOAD, 1, 0},
        Instruction{OpCode::LOAD, 2, 0},   Instruction{OpCode::OP, 1, 0},
        Instruction{OpCode::JMP, 5, 0},    Instruction{OpCode::LOAD, 1, 0},
        Instruction{OpCode::LOAD, 2, 0},   Instruction{OpCode::LOADC, 0, 0},
        Instruction{OpCode::OP, 2, 0},     Instruction{OpCode::STORE, 4, 0},
        Instruction{OpCode::DROPR, 2, 0},  Instruction{OpCode::DROPR, 1, 0},
        Instruction{OpCode::MOVE, 4, 0},   Instruction{OpCode::RET, 0, 0},
    }), // instructions_
    std::vector<OperatorString>({
        OperatorString({"aten::is_floating_point", "", 1}),
        OperatorString({"aten::div", "Tensor", 2}),
        OperatorString({"aten::div", "Tensor_mode", 3}),
    }), // op_names
    std::vector<c10::IValue>({
        c10::IValue("trunc"),
        c10::IValue(true),
    }), // constants
    std::vector<c10::TypePtr>(), // types
    4 // register_size_
})});

} // namespace jit
} // namespace torch
