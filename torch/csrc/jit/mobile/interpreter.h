#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/instruction.h>
#include <aten/src/ATen/core/dispatch/Dispatcher.h>

namespace torch{
namespace jit{
namespace mobile {
using Stack = std::vector<c10::IValue>;

struct Code {
  std::vector<Instruction> instructions_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<c10::optional<c10::OperatorHandle>> operators_;
  std::vector<c10::IValue> constants_;
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
