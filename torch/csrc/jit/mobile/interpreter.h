#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace torch{
namespace jit{
namespace mobile {
using Stack = std::vector<c10::IValue>;
struct Code {
  std::vector<Instruction> instructions_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<std::function<void(Stack&)>> operators_;
  std::vector<c10::IValue> constants_;
  std::vector<c10::TypePtr> types_;
  size_t register_size_; // Aggregated output size.
};

struct InterpreterState {
  TORCH_API explicit InterpreterState(std::shared_ptr<Code> code);
  TORCH_API bool run(Stack& stack);

 private:
  std::shared_ptr<Code> code_;
  c10::IValue& reg(size_t reg);
  std::vector<c10::IValue> registers_;
};

} // namespace mobile
} // namespace torch
} // namespace jit
