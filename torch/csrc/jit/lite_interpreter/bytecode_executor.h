#pragma once

#include <torch/csrc/jit/generic_instruction.h>

namespace torch {
namespace jit {

class InstructionExecutor final {
 public:
  explicit InstructionExecutor(std::shared_ptr<GenericInstructionList> ins_list);
  IValue run(Stack& stack);

 private:
  void loadTensorsFromRegisters(const std::vector<Variable>& uses, Stack& stack);

  size_t pc = 0;
  std::shared_ptr<GenericInstructionList> ins_list;
  std::vector<IValue> registers;
};

}
}
