#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/instruction.h>

#include <vector>

namespace torch{
namespace jit{
namespace mobile {
struct Method{
  std::vector<Instruction> instructions;
  std::vector<c10::OperatorName> op_names;
  std::vector<c10::IValue> constants;
};

struct Bytecode {
  std::unique_ptr<c10::ivalue::Object> object;
  std::vector<Method> methods;
};
}
}
}
