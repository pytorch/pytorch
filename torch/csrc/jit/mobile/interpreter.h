#pragma once

#include <vector>

#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/frame.h>

namespace torch {
namespace jit {
namespace mobile {

struct InterpreterState {
  TORCH_API explicit InterpreterState(const Code& code);
  TORCH_API bool run(Stack& stack);

 private:
  void enterFrame(const Code&);
  void leaveFrame();
  void saveExceptionDebugHandles();
  void callFunction(torch::jit::Function& f, Stack& stack);

  c10::IValue& reg(size_t reg);
  std::vector<c10::IValue> registers_;
  std::vector<Frame> frames_;
};

const std::vector<DebugHandle>& getInterpretersExceptionDebugHandles();
} // namespace mobile
} // namespace jit
} // namespace torch
