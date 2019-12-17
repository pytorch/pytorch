#ifndef NNC_INCLUDE_LLVM_CODEGEN_H_
#define NNC_INCLUDE_LLVM_CODEGEN_H_

#include "ir_visitor.h"
#include "llvm_jit.h"

#include <llvm/IR/IRBuilder.h>

namespace torch {
namespace jit {
namespace compiler {

class LLVMCodeGen : public IRVisitor {
 private:
  llvm::LLVMContext context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_;
  llvm::Type* int32Ty_;

 public:
  LLVMCodeGen();
  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const IntImm* v) override;
  void visit(const FloatImm* v) override;
  int value();
};

} // namespace compiler
} // namespace jit
} // namespace torch

#endif  // NNC_INCLUDE_LLVM_CODEGEN_H_
