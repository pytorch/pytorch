#ifndef NNC_LIB_LLVM_JIT_H_
#define NNC_LIB_LLVM_JIT_H_

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>

namespace llvm {
namespace orc {

class PytorchLLVMJITImpl;

class PytorchLLVMJIT {
 public:
  PytorchLLVMJIT();
  ~PytorchLLVMJIT();
  TargetMachine& getTargetMachine();
  VModuleKey addModule(std::unique_ptr<Module> M);
  JITSymbol findSymbol(const std::string Name);
  JITTargetAddress getSymbolAddress(const std::string Name);
  void removeModule(VModuleKey K);
  
 private:
  // Use PImpl idiom here to hide the no-rtti parts of the JIT structure.
  std::unique_ptr<PytorchLLVMJITImpl> impl_;
};

}  // end namespace orc
}  // end namespace llvm

#endif  // NNC_LIB_LLVM_JIT_H_
