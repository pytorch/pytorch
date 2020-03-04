#pragma once

#ifdef ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <unordered_map>
#include <vector>

#define DEBUG_PRINT 0

#if DEBUG_PRINT
#include <llvm/IR/LegacyPassManager.h>
#endif

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API LLVMCodeGen : public CodeGen, public IRVisitor {
 private:
  llvm::orc::ThreadSafeContext context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::TargetMachine> TM_;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_;
  llvm::JITTargetAddress kernelAddress_;

#define LLVM_TYPE_DECLARE(_1, Name) llvm::Type* Name##Ty_;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, LLVM_TYPE_DECLARE);
#undef LLVM_TYPE_DECLARE

  std::unordered_map<const Var*, int> varToArg_;
  std::unordered_map<const Var*, llvm::Value*> varToVal_;

  std::vector<void*> args_;

 private:
  llvm::LLVMContext& getContext();
  llvm::Type* dtypeToLLVM(Dtype dtype);
  llvm::Type* dtypeToLLVMPtr(Dtype dtype);
  void emitWrapper(const std::vector<llvm::Type*>& params);
  void emitKernel(Stmt* stmt, const std::vector<llvm::Type*>& params);

 public:
  explicit LLVMCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& args,
      Dtype dtype = kInt);
  explicit LLVMCodeGen(Stmt* stmt);

  ~LLVMCodeGen() override {}

  TORCH_API void call(const std::vector<CallArg>& args) override;

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Mod* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;

#define IMM_VISIT_DECLARE(_1, Name) void visit(const Name##Imm* v) override;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT_DECLARE);
#undef IMM_VISIT_DECLARE

  void visit(const Cast* v) override;
  void visit(const Var* v) override;
  void visit(const Let* v) override;
  void visit(const LetStmt* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
  void visit(const Store* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const BaseCallNode* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const FunctionCall* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Cond* v) override;

  llvm::Value* emitUnmaskedLoad(llvm::Value* addr, llvm::Value* idx);
  llvm::Value* emitMaskedLoad(
      llvm::Value* addr,
      llvm::Value* idx,
      llvm::Value* mask);
  void emitUnmaskedStore(llvm::Value* base, llvm::Value* idx, llvm::Value* val);
  void emitMaskedStore(
      llvm::Value* base,
      llvm::Value* idx,
      llvm::Value* mask,
      llvm::Value* val);

  void optimize(llvm::Module& M);

  template <typename T>
  T value() {
    std::vector<void*> args;
    return value<T>(args);
  }

  template <typename T>
  T value(std::vector<void*>& args) {
    T (*fp)(void**) = (T(*)(void**))kernelAddress_;
    T rv = fp(args.data());
    return rv;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
