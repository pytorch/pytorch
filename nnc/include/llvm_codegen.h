#ifndef NNC_INCLUDE_LLVM_CODEGEN_H_
#define NNC_INCLUDE_LLVM_CODEGEN_H_

#include "ir_visitor.h"
#include "llvm_jit.h"

#include <llvm/IR/IRBuilder.h>

namespace nnc {

class LLVMCodegen : public IRVisitor {
 private:
  llvm::LLVMContext context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<PytorchLlvmJit> jit_;
  std::unique_ptr<llvm::Module> module_;
  
 public:
  LLVMCodegen();
  void visit(const Add *v) override;
  void visit(const Sub *v) override;
  void visit(const Mul *v) override;
  void visit(const Div *v) override;
  void visit(const IntImm *v) override;
  void visit(const FloatImm *v) override;
};

} // namespace nnc

#endif // NNC_INCLUDE_LLVM_CODEGEN_H_
