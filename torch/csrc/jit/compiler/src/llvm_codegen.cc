#include "torch/csrc/jit/compiler/include/llvm_codegen.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/types.h"

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <memory>

using namespace torch::jit::compiler;

LLVMCodeGen::LLVMCodeGen(const std::vector<Buffer*>& args, Dtype dtype) : irb_(context_) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>();
  module_ = std::make_unique<llvm::Module>("pytorch", context_);
  module_->setDataLayout(jit_->getTargetMachine().createDataLayout());
  module_->setTargetTriple(
      jit_->getTargetMachine().getTargetTriple().normalize());

  int32Ty_ = llvm::Type::getInt32Ty(context_);
  floatTy_ = llvm::Type::getFloatTy(context_);

  // Emit prototype.
  llvm::Type* ret_ty = nullptr;
  if (dtype == kInt32) {
    ret_ty = int32Ty_;
  } else if (dtype == kFloat32) {
    ret_ty = floatTy_;
  }
  std::vector<llvm::Type*> params;
  for (int i = 0; i < args.size(); i++) {
    auto const &arg = args[i];
    if (arg->dtype() == kInt32) {
      params.push_back(llvm::Type::getInt32PtrTy(context_));
    } else if (arg->dtype() == kFloat32) {
      params.push_back(llvm::Type::getFloatPtrTy(context_));
    }
    varToArg_[args[i]->data().node()] = i;
  }
  llvm::FunctionType* fntype = llvm::FunctionType::get(ret_ty, params, false);
  fn_ = llvm::Function::Create(
      fntype, llvm::Function::PrivateLinkage, "pytorch", module_.get());
  for (int i = 0; i < args.size(); i++) {
    fn_->addParamAttr(i, llvm::Attribute::NoAlias);
  }

  // Emit wrapper to unpack argument vector.
  auto voidPP = llvm::Type::getVoidTy(context_)->getPointerTo()->getPointerTo();
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(int32Ty_, {voidPP}, false),
      llvm::Function::ExternalLinkage,
      "wrapper",
      module_.get());
  auto wrapBB = llvm::BasicBlock::Create(context_, "wrapBB", wrapper);
  irb_.SetInsertPoint(wrapBB);
  llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
  for (size_t i = 0; i < args.size(); i++) {
    auto argp = irb_.CreateGEP(
        wrapper->arg_begin(),
        llvm::Constant::getIntegerValue(int32Ty_, llvm::APInt(32, i)));
    auto arg = irb_.CreateLoad(argp);
    wrappedArgs.push_back(arg);
  }
  auto cc = irb_.CreateCall(fn_, wrappedArgs);
  irb_.CreateRet(cc);

  // Set insert point to the real function.
  bb_ = llvm::BasicBlock::Create(context_, "entry", fn_);
  irb_.SetInsertPoint(bb_);
}

LLVMCodeGen::LLVMCodeGen() : LLVMCodeGen({}) {}

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
  value_ = 
    llvm::ConstantFP::get(floatTy_, v->value());
}

void LLVMCodeGen::visit(const Cast* v) {
  v->src_value().accept(this);

  if (v->dtype().lanes() == 1) {
    if (v->dtype() == kInt32 &&
        v->src_value().dtype() == kFloat32) {
      value_ = irb_.CreateFPToSI(value_, int32Ty_);
      return;
    }

    if (v->dtype() == kFloat32 &&
        v->src_value().dtype() == kInt32) {
      value_ = irb_.CreateSIToFP(value_, floatTy_);
      return;
    }
  }

  assert(0 && "Unhandled cast");
}

void LLVMCodeGen::visit(const Variable* v) {
  if (varToArg_.count(v)) {
    auto idx = varToArg_.at(v);
    auto arg = fn_->arg_begin() + idx;
    value_ = arg;
  } else if (varToVal_.count(v)) {
    value_ = varToVal_.at(v);
  }
}

void LLVMCodeGen::visit(const Let* v) {}
void LLVMCodeGen::visit(const Ramp* v) {}

void LLVMCodeGen::visit(const Load* v) {
  v->base_handle().accept(this);
  auto base = this->value_;
  v->index().accept(this);
  auto idx = this->value_;
  auto addr = irb_.CreateGEP(base, idx);
  value_ = irb_.CreateLoad(addr);
}

void LLVMCodeGen::visit(const For* v) {
  // Create "start" value.
  v->start().accept(this);
  auto start = this->value_;

  // Create loop preheader and body.
  auto preheader = irb_.GetInsertBlock();
  auto loop = llvm::BasicBlock::Create(context_, "loop", fn_);
  irb_.CreateBr(loop);
  irb_.SetInsertPoint(loop);

  // Set up phi node for index variable.
  auto idx = irb_.CreatePHI(int32Ty_, 2);
  idx->addIncoming(start, preheader);
  varToVal_.emplace(v->var().node(), idx);

  // Codegen the body.
  v->body().accept(this);

  // Create the stop condition. and "after" block.
  auto inc = irb_.CreateAdd(
      idx, llvm::Constant::getIntegerValue(int32Ty_, llvm::APInt(32, 1)));
  v->stop().accept(this);
  auto stop = this->value_;
  auto cond = irb_.CreateICmpSLT(inc, stop);

  // Branch back to top of loop and finish phi for index variable.
  auto end_loop = irb_.GetInsertBlock();
  auto after = llvm::BasicBlock::Create(context_, "after", fn_);
  irb_.CreateCondBr(cond, loop, after);
  irb_.SetInsertPoint(after);
  idx->addIncoming(inc, end_loop);
  value_ = llvm::Constant::getIntegerValue(int32Ty_, llvm::APInt(32, 0));
}

void LLVMCodeGen::visit(const Block* v) {
  for (int i = 0; i < v->nstmts(); i++) {
    v->stmt(i).accept(this);
  }
}

void LLVMCodeGen::visit(const Store* v) {
  v->base_handle().accept(this);
  auto base = this->value_;
  v->index().accept(this);
  auto idx = this->value_;
  v->value().accept(this);
  auto val = this->value_;
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
  value_ = llvm::Constant::getIntegerValue(int32Ty_, llvm::APInt(32, 0));
}

void LLVMCodeGen::visit(const Broadcast* v) {}

void LLVMCodeGen::optimize(llvm::TargetMachine& TM, llvm::Module& M) {
  llvm::legacy::FunctionPassManager FPM(&M);
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(llvm::createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  FPM.add(llvm::createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  TM.adjustPassManager(PMB);

  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);
  FPM.doInitialization();
  PM.run(M);
  for (auto& FF : M) {
    FPM.run(FF);
  }
  FPM.doFinalization();
  PM.run(M);
}
