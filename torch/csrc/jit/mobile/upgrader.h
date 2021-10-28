#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <unordered_map>

namespace torch {
namespace jit {
namespace mobile {

// fix inline
inline c10::IValue to_tuple_2(std::vector<c10::IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

struct Upgrader {
  int min_version;
  int max_version;
  std::string upgrader_name;
  int index;
};

// From operator_versions.yaml
// const std::unordered_map<
//     std::string,
//     std::vector<std::pair<std::pair<int, int>, std::string>>>
//     operator_versions({
//         // The following are called from setup sections.
//         {std::string("aten::div_Tensor"),
//          std::vector{std::make_pair(
//              std::make_pair(0, 3),
//              std::string("aten::div_Tensor_0_3"))}},
//     });
const std::unordered_map<
    std::string,
    Upgrader>
    operator_versions({// The following are called from setup sections.
                       {std::string("aten::div_Tensor"),
                        Upgrader{0, 3, "aten::div_Tensor_0_3", 0}}});

// From upgraders.yaml
const std::vector<std::pair<std::string, c10::IValue>> upgraders = {
    // The following are called from setup sections.
    {"aten::div_Tensor_0_3",
     to_tuple_2(std::vector<c10::IValue>{
         //  instructions
         to_tuple_2(std::vector<c10::IValue>{
             to_tuple_2({"STOREN", 1, 2}), to_tuple_2({"LOAD", 1, 0}),
             to_tuple_2({"OP", 0, 0}),     to_tuple_2({"JF", 3, 0}),
             to_tuple_2({"LOADC", 1, 0}),  to_tuple_2({"JMP", 3, 0}),
             to_tuple_2({"LOAD", 2, 0}),   to_tuple_2({"OP", 0, 0}),
             to_tuple_2({"STORE", 3, 0}),  to_tuple_2({"MOVE", 3, 0}),
             to_tuple_2({"JF", 5, 0}),     to_tuple_2({"LOAD", 1, 0}),
             to_tuple_2({"LOAD", 2, 0}),   to_tuple_2({"OP", 1, 0}),
             to_tuple_2({"JMP", 5, 0}),    to_tuple_2({"LOAD", 1, 0}),
             to_tuple_2({"LOAD", 2, 0}),   to_tuple_2({"LOADC", 0, 0}),
             to_tuple_2({"OP", 2, 0}),     to_tuple_2({"STORE", 4, 0}),
             to_tuple_2({"DROPR", 2, 0}),  to_tuple_2({"DROPR", 1, 0}),
             to_tuple_2({"MOVE", 4, 0}),   to_tuple_2({"RET", 0, 0}),
         }),
         //  operators
         to_tuple_2(std::vector<c10::IValue>{
             to_tuple_2({"aten::is_floating_point", "", 1}),
             to_tuple_2({"aten::div", "Tensor", 2}),
             to_tuple_2({"aten::div", "Tensor_mode", 3}),
         }),
         //  constants
         to_tuple_2(std::vector<c10::IValue>{
             c10::IValue("trunc"),
             c10::IValue(true),
         }),
         //  types
         to_tuple_2(std::vector<c10::IValue>{

         }),
         //  register_size
         to_tuple_2(std::vector<c10::IValue>{
             4,
         }),
     })},
    //  to_tuple_2(
    //      std::vector<c10::IValue>{
    //      to_tuple_2({"STORE", 1, 0}),
    //      to_tuple_2({"LOAD", 1, 0}),
    //      to_tuple_2({"MOVE", 1, 0}),
    //      to_tuple_2({"LOADC", 0, 0}),
    //      to_tuple_2({"OP", 0, 0}),
    //      to_tuple_2({"RET", 0, 0}),
    //     }
    //  )},
};

} // namespace mobile
} // namespace jit
} // namespace torch
