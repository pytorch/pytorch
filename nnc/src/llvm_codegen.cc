#include "llvm_codegen.h"

using namespace nnc;

LLVMCodegen::LLVMCodegen() 
  : irb_(context_),
    jit_(std::make_unique<PytorchLlvmJit>()),
    module_(std::make_unique<llvm::Module>("pytorch", context_))
{
  module_->setDataLayout(jit_->getTargetMachine().createDataLayout());
}

void LLVMCodegen::visit(const Add *v) {
}

void LLVMCodegen::visit(const Sub *v) {
}

void LLVMCodegen::visit(const Mul *v) {
}

void LLVMCodegen::visit(const Div *v) {
}

void LLVMCodegen::visit(const IntImm *v) {
}

void LLVMCodegen::visit(const FloatImm *v) {
}
