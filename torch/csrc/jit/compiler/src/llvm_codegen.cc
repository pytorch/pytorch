#include "torch/csrc/jit/compiler/include/llvm_codegen.h"
#include "torch/csrc/jit/compiler/include/ir.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <memory>

using namespace torch::jit::compiler;

LLVMCodeGen::LLVMCodeGen() : irb_(context_) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>();
  module_ = std::make_unique<llvm::Module>("pytorch", context_);
  module_->setDataLayout(jit_->getTargetMachine().createDataLayout());
  module_->setTargetTriple(
      jit_->getTargetMachine().getTargetTriple().normalize());

  // Emit prototype.
  int32Ty_ = llvm::Type::getInt32Ty(context_);

  llvm::FunctionType* fntype = llvm::FunctionType::get(int32Ty_, {}, false);
  fn_ = llvm::Function::Create(
      fntype, llvm::Function::ExternalLinkage, "pytorch", module_.get());
  bb_ = llvm::BasicBlock::Create(context_, "entry", fn_);
  irb_.SetInsertPoint(bb_);
}

void LLVMCodeGen::visit(const Add* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  v->rhs().accept(this);
  auto rhs = this->value_;
  value_ = irb_.CreateAdd(lhs, rhs);
}

void LLVMCodeGen::visit(const Sub* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  v->rhs().accept(this);
  auto rhs = this->value_;
  value_ = irb_.CreateSub(lhs, rhs);
}

void LLVMCodeGen::visit(const Mul* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  v->rhs().accept(this);
  auto rhs = this->value_;
  value_ = irb_.CreateMul(lhs, rhs);
}

void LLVMCodeGen::visit(const Div* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  v->rhs().accept(this);
  auto rhs = this->value_;
  value_ = irb_.CreateSDiv(lhs, rhs);
}

void LLVMCodeGen::visit(const IntImm* v) {
  value_ =
      llvm::Constant::getIntegerValue(int32Ty_, llvm::APInt(32, v->value()));
}

void LLVMCodeGen::visit(const FloatImm* v) {
  assert(false && "Integer only now sorry");
}

int LLVMCodeGen::value() {
  irb_.CreateRet(value_);
  assert(!llvm::verifyFunction(*fn_, &llvm::outs()));

  auto key = jit_->addModule(std::move(module_));
  auto sym = jit_->findSymbol("pytorch");
  auto addr = sym.getAddress();
  assert(addr);
  int (*fp)() = (int (*)())addr.get();
  int rv = fp();
  jit_->removeModule(key);
  return rv;
}
