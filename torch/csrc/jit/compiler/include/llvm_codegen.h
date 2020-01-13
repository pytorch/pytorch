#pragma once

#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/ir_visitor.h"
#include "torch/csrc/jit/compiler/include/llvm_jit.h"

#include <llvm/IR/IRBuilder.h>
#include <unordered_map>
#include <vector>

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
  llvm::Type* floatTy_;

  std::unordered_map<const BaseExprNode*, int> varToArg_;
  std::unordered_map<const Variable*, llvm::Value*> varToVal_;

 public:
  explicit LLVMCodeGen(const std::vector<Buffer*>& args);
  LLVMCodeGen();

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const IntImm* v) override;
  void visit(const FloatImm* v) override;
  void visit(const Cast* v) override;
  void visit(const Variable* v) override;
  void visit(const Let* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
  void visit(const Store* v) override;
  void visit(const Broadcast* v) override;

  void optimize(llvm::TargetMachine& TM, llvm::Module& M);


  template<typename T> T value() {
    std::vector<void*> args;
    return value<T>(args);
  }

  template<typename T>
  T value(std::vector<void*>& args) {
    irb_.CreateRet(value_);
    assert(!llvm::verifyFunction(*fn_, &llvm::outs()));
    optimize(jit_->getTargetMachine(), *module_);

  #if DEBUG_PRINT
    llvm::errs() << *module_;
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
    jit_->getTargetMachine().addPassesToEmitFile(
        PM,
        asmStream,
        nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
    PM.run(*module_);
    llvm::errs() << asmStream.str();
  #endif

    auto key = jit_->addModule(std::move(module_));
    auto sym = jit_->findSymbol("wrapper");
    auto addr = sym.getAddress();
    assert(addr);
    T (*fp)(void**) = (T (*)(void**))addr.get();
    T rv = fp(args.data());
    jit_->removeModule(key);
    return rv;
  }
};

} // namespace compiler
} // namespace jit
} // namespace torch
