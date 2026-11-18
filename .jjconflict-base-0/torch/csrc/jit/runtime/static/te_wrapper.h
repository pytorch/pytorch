#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch::jit {

class TEWrapper {
 public:
  TEWrapper() = default;
  void call(const std::vector<void*>& args);

  template <typename ExpectedType>
  bool checkInput(const at::Tensor& t) {
#ifdef TORCH_ENABLE_LLVM
    return t.is_contiguous() && t.dtype().Match<ExpectedType>();
#else
    return false;
#endif
  }

#ifdef TORCH_ENABLE_LLVM
  void update(std::unique_ptr<tensorexpr::LLVMCodeGen>&& cg_);
#endif

 private:
#ifdef TORCH_ENABLE_LLVM
  std::unique_ptr<tensorexpr::LLVMCodeGen> cg;
#endif
};

std::shared_ptr<TEWrapper> createDiv();
std::shared_ptr<TEWrapper> createLogit();
std::shared_ptr<TEWrapper> createRelu();
std::shared_ptr<TEWrapper> createTanh();
std::shared_ptr<TEWrapper> createSigmoid();
std::shared_ptr<TEWrapper> createSignedLog1p();
std::shared_ptr<TEWrapper> createClamp();
std::shared_ptr<TEWrapper> createClampNanToNum();

} // namespace torch::jit
