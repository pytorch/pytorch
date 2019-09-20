#pragma once
#include <torch/csrc/jit/lite_interpreter/bytecode.h>

namespace torch{
namespace jit{
namespace mobile {
using Stack = std::vector<c10::IValue>;

struct InterpreterState {
  TORCH_API InterpreterState(const Bytecode& bytecode);
  TORCH_API bool run(Stack& stack);

 private:
  c10::IValue& reg(size_t reg);
  std::vector<Instruction> instructions_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<c10::IValue> constants_;
  size_t register_size_; // Aggregated output size.
  std::vector<c10::IValue> registers_;
};

} // namespace mobile
} // namespace torch
} // namespace jit
