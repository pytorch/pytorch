#pragma once

// #include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <unordered_map>

namespace torch {
namespace jit {

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

static const std::vector<std::pair<std::string, c10::IValue>> kUpgraderBytecode(
    {
        // The following are called from setup sections.
        {"div_Tensor_0_3",
         to_tuple(std::vector<c10::IValue>{
             //  instructions
             to_tuple(std::vector<c10::IValue>{
                 to_tuple({"STOREN", 1, 2}), to_tuple({"LOAD", 1, 0}),
                 to_tuple({"OP", 0, 0}),     to_tuple({"JF", 3, 0}),
                 to_tuple({"LOADC", 1, 0}),  to_tuple({"JMP", 3, 0}),
                 to_tuple({"LOAD", 2, 0}),   to_tuple({"OP", 0, 0}),
                 to_tuple({"STORE", 3, 0}),  to_tuple({"MOVE", 3, 0}),
                 to_tuple({"JF", 5, 0}),     to_tuple({"LOAD", 1, 0}),
                 to_tuple({"LOAD", 2, 0}),   to_tuple({"OP", 1, 0}),
                 to_tuple({"JMP", 5, 0}),    to_tuple({"LOAD", 1, 0}),
                 to_tuple({"LOAD", 2, 0}),   to_tuple({"LOADC", 0, 0}),
                 to_tuple({"OP", 2, 0}),     to_tuple({"STORE", 4, 0}),
                 to_tuple({"DROPR", 2, 0}),  to_tuple({"DROPR", 1, 0}),
                 to_tuple({"MOVE", 4, 0}),   to_tuple({"RET", 0, 0}),
             }),
             //  operators
             to_tuple(std::vector<c10::IValue>{
                 to_tuple({"aten::is_floating_point", "", 1}),
                 to_tuple({"aten::div", "Tensor", 2}),
                 to_tuple({"aten::div", "Tensor_mode", 3}),
             }),
             //  constants
             to_tuple(std::vector<c10::IValue>{
                 c10::IValue("trunc"),
                 c10::IValue(true),
             }),
             //  types
             to_tuple(std::vector<c10::IValue>{

             }),
             //  register_size
             to_tuple(std::vector<c10::IValue>{
                 4,
             }),
         })},
    });

} // namespace jit
} // namespace torch
