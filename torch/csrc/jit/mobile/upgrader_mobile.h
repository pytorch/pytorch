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

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

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
    {
        {std::string("aten::div.Tensor"),
         std::vector<Upgrader>({Upgrader({0, 3, "div_Tensor_0_3", 0})})},
        {std::string("aten::div.Scalar"),
         std::vector<Upgrader>({Upgrader({0, 3, "div_Scalar_0_3", 1})})},
        {std::string("aten::div.out"),
         std::vector<Upgrader>({Upgrader({0, 3, "div_out_0_3", 2})})},
        {std::string("aten::div_.Tensor"),
         std::vector<Upgrader>({Upgrader({0, 3, "div__Tensor_0_3", 3})})},
        {std::string("aten::div_.Scalar"),
         std::vector<Upgrader>({Upgrader({0, 3, "div__Scalar_0_3", 4})})},
    });

struct OperatorString {
  const std::string name;
  const std::string overload_name;
  const c10::optional<int> num_specified_args;
};

struct ByteCodeFunctionWithOperator {
  mobile::Function& function;
  std::vector<OperatorString> operators;
};

static std::vector<ByteCodeFunctionWithOperator> kUpgraderByteCode({
    ByteCodeFunctionWithOperator({
        mobile::Function::registerFunc(
            "div_Tensor_0_3",
            std::vector<Instruction>({
                Instruction{OpCode::STOREN, 1, 2},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::STORE, 3, 0},
                Instruction{OpCode::MOVE, 3, 0},
                Instruction{OpCode::JF, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 1, 0},
                Instruction{OpCode::JMP, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::LOADC, 0, 0},
                Instruction{OpCode::OP, 2, 0},
                Instruction{OpCode::STORE, 4, 0},
                Instruction{OpCode::DROPR, 2, 0},
                Instruction{OpCode::DROPR, 1, 0},
                Instruction{OpCode::MOVE, 4, 0},
                Instruction{OpCode::RET, 0, 0},
            }), // instructions_
            std::vector<c10::IValue>({
                c10::IValue("trunc"),
                c10::IValue(true),
            }), // constants
            std::vector<c10::TypePtr>(), // types
            4 // register_size_
            ),
        std::vector<OperatorString>({
            OperatorString({"aten::is_floating_point", "", 1}),
            OperatorString({"aten::div", "Tensor", 2}),
            OperatorString({"aten::div", "Tensor_mode", 3}),
        }), // op_names
    }),
    ByteCodeFunctionWithOperator({
        mobile::Function::registerFunc(
            "div_Scalar_0_3",
            std::vector<Instruction>({
                Instruction{OpCode::STOREN, 1, 2},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::ISINSTANCE, 0, 1},
                Instruction{OpCode::STORE, 3, 0},
                Instruction{OpCode::MOVE, 3, 0},
                Instruction{OpCode::JF, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 1, 0},
                Instruction{OpCode::JMP, 6, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 2, 0},
                Instruction{OpCode::LOADC, 0, 0},
                Instruction{OpCode::OP, 3, 0},
                Instruction{OpCode::STORE, 4, 0},
                Instruction{OpCode::DROPR, 2, 0},
                Instruction{OpCode::DROPR, 1, 0},
                Instruction{OpCode::MOVE, 4, 0},
                Instruction{OpCode::RET, 0, 0},
            }), // instructions_
            std::vector<c10::IValue>({
                c10::IValue("trunc"),
                c10::IValue(true),
            }), // constants
            std::vector<c10::TypePtr>({c10::parseType("float")}), // types
            4 // register_size_
            ),
        std::vector<OperatorString>({
            OperatorString({"aten::is_floating_point", "", 1}),
            OperatorString({"aten::div", "Scalar", 2}),
            OperatorString({"prim::unchecked_cast", "", 1}),
            OperatorString({"aten::div", "Scalar_mode", 3}),
        }), // op_names
    }),
    ByteCodeFunctionWithOperator({
        mobile::Function::registerFunc(
            "div_out_0_3",
            std::vector<Instruction>({
                Instruction{OpCode::STOREN, 1, 3},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 3, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::STORE, 4, 0},
                Instruction{OpCode::MOVE, 4, 0},
                Instruction{OpCode::JF, 6, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::LOAD, 3, 0},
                Instruction{OpCode::OP, 1, 0},
                Instruction{OpCode::JMP, 6, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::LOADC, 0, 0},
                Instruction{OpCode::LOAD, 3, 0},
                Instruction{OpCode::OP, 2, 0},
                Instruction{OpCode::STORE, 5, 0},
                Instruction{OpCode::DROPR, 3, 0},
                Instruction{OpCode::DROPR, 2, 0},
                Instruction{OpCode::DROPR, 1, 0},
                Instruction{OpCode::MOVE, 5, 0},
                Instruction{OpCode::RET, 0, 0},
            }), // instructions_
            std::vector<c10::IValue>({
                c10::IValue("trunc"),
                c10::IValue(true),
            }), // constants
            std::vector<c10::TypePtr>(), // types
            5 // register_size_
            ),
        std::vector<OperatorString>({
            OperatorString({"aten::is_floating_point", "", 1}),
            OperatorString({"aten::div", "out", 3}),
            OperatorString({"aten::div", "out_mode", 4}),
        }), // op_names
    }),
    ByteCodeFunctionWithOperator({
        mobile::Function::registerFunc(
            "div__Tensor_0_3",
            std::vector<Instruction>({
                Instruction{OpCode::STOREN, 1, 2},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::STORE, 3, 0},
                Instruction{OpCode::MOVE, 3, 0},
                Instruction{OpCode::JF, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 1, 0},
                Instruction{OpCode::JMP, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::LOADC, 0, 0},
                Instruction{OpCode::OP, 2, 0},
                Instruction{OpCode::STORE, 4, 0},
                Instruction{OpCode::DROPR, 2, 0},
                Instruction{OpCode::DROPR, 1, 0},
                Instruction{OpCode::MOVE, 4, 0},
                Instruction{OpCode::RET, 0, 0},
            }), // instructions_
            std::vector<c10::IValue>({
                c10::IValue("trunc"),
                c10::IValue(true),
            }), // constants
            std::vector<c10::TypePtr>(), // types
            4 // register_size_
            ),
        std::vector<OperatorString>({
            OperatorString({"aten::is_floating_point", "", 1}),
            OperatorString({"aten::div_", "Tensor", 2}),
            OperatorString({"aten::div_", "Tensor_mode", 3}),
        }), // op_names
    }),
    ByteCodeFunctionWithOperator({
        mobile::Function::registerFunc(
            "div__Scalar_0_3",
            std::vector<Instruction>({
                Instruction{OpCode::STOREN, 1, 2},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::OP, 0, 0},
                Instruction{OpCode::JF, 3, 0},
                Instruction{OpCode::LOADC, 1, 0},
                Instruction{OpCode::JMP, 3, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::ISINSTANCE, 0, 1},
                Instruction{OpCode::STORE, 3, 0},
                Instruction{OpCode::MOVE, 3, 0},
                Instruction{OpCode::JF, 5, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 1, 0},
                Instruction{OpCode::JMP, 6, 0},
                Instruction{OpCode::LOAD, 1, 0},
                Instruction{OpCode::LOAD, 2, 0},
                Instruction{OpCode::OP, 2, 0},
                Instruction{OpCode::LOADC, 0, 0},
                Instruction{OpCode::OP, 3, 0},
                Instruction{OpCode::STORE, 4, 0},
                Instruction{OpCode::DROPR, 2, 0},
                Instruction{OpCode::DROPR, 1, 0},
                Instruction{OpCode::MOVE, 4, 0},
                Instruction{OpCode::RET, 0, 0},
            }), // instructions_
            std::vector<c10::IValue>({
                c10::IValue("trunc"),
                c10::IValue(true),
            }), // constants
            std::vector<c10::TypePtr>(), // types
            4 // register_size_
            ),
        std::vector<OperatorString>({
            OperatorString({"aten::is_floating_point", "", 1}),
            OperatorString({"aten::div_", "Scalar", 2}),
            OperatorString({"prim::unchecked_cast", "", 1}),
            OperatorString({"aten::div_", "Scalar_mode", 3}),
        }), // op_names
    }),
});

} // namespace jit
} // namespace torch
