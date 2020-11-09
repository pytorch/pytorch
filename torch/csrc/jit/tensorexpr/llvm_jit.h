#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>

namespace llvm {
namespace orc {

class PytorchLLVMJITImpl;

class TORCH_API PytorchLLVMJIT {
 public:
  PytorchLLVMJIT();
  ~PytorchLLVMJIT();

  Error addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C);

  JITSymbol findSymbol(const std::string Name);

  TargetMachine& getTargetMachine();

  const DataLayout& getDataLayout();

 private:
  // Use the PImpl idiom here to hide the no-rtti parts of the JIT structure.
  std::unique_ptr<PytorchLLVMJITImpl> impl_;
};

} // end namespace orc
} // end namespace llvm

#endif // ENABLE LLVM
