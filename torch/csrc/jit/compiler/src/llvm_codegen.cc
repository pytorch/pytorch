#include "torch/csrc/jit/compiler/include/llvm_codegen.h"

#include <memory>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "torch/csrc/jit/compiler/include/buffer.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/types.h"

using namespace torch::jit::compiler;

LLVMCodeGen::LLVMCodeGen() : LLVMCodeGen(std::vector<Buffer*>()) {}

LLVMCodeGen::LLVMCodeGen(const std::vector<Buffer*>& args, Dtype dtype)
    : context_(std::make_unique<llvm::LLVMContext>()),
      irb_(*context_.getContext()) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

#if 0
  // FIXME: Switch to using detectHost() rather than setting up the JTMB manually
  // once LLVM 10 is available.
  auto JTMB = llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
#else
  llvm::orc::JITTargetMachineBuilder JTMB(
      (llvm::Triple(llvm::sys::getProcessTriple())));

  // Retrieve host CPU name and sub-target features and add them to builder.
  // Relocation model, code model and codegen opt level are kept to default
  // values.
  llvm::SubtargetFeatures SubtargetFeatures;
  llvm::StringMap<bool> FeatureMap;
  llvm::sys::getHostCPUFeatures(FeatureMap);
  for (auto& Feature : FeatureMap) {
    SubtargetFeatures.AddFeature(Feature.first(), Feature.second);
  }

  JTMB.setCPU(llvm::sys::getHostCPUName());
  JTMB.addFeatures(SubtargetFeatures.getFeatures());
#endif

  TM = llvm::cantFail(JTMB.createTargetMachine());

  jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>();
  module_ = std::make_unique<llvm::Module>("pytorch", *context_.getContext());
  module_->setDataLayout(cantFail(JTMB.getDefaultDataLayoutForTarget()));
  module_->setTargetTriple(JTMB.getTargetTriple().str());

  int32Ty_ = llvm::Type::getInt32Ty(*context_.getContext());
  floatTy_ = llvm::Type::getFloatTy(*context_.getContext());

  // Emit prototype.
  llvm::Type* ret_ty = nullptr;
  if (dtype == kInt32) {
    ret_ty = int32Ty_;
  } else if (dtype == kFloat32) {
    ret_ty = floatTy_;
  }
  std::vector<llvm::Type*> params;
  for (int i = 0; i < args.size(); i++) {
    auto const& arg = args[i];
    if (arg->dtype() == kInt32) {
      params.push_back(llvm::Type::getInt32PtrTy(*context_.getContext()));
    } else if (arg->dtype() == kFloat32) {
      params.push_back(llvm::Type::getFloatPtrTy(*context_.getContext()));
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
  auto voidPP = llvm::Type::getVoidTy(*context_.getContext())
                    ->getPointerTo()
                    ->getPointerTo();
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(int32Ty_, {voidPP}, false),
      llvm::Function::ExternalLinkage,
      "wrapper",
      module_.get());
  auto wrapBB =
      llvm::BasicBlock::Create(*context_.getContext(), "wrapBB", wrapper);
  irb_.SetInsertPoint(wrapBB);
  llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
  for (size_t i = 0; i < args.size(); i++) {
    auto argp = irb_.CreateGEP(
        wrapper->arg_begin(), llvm::ConstantInt::getSigned(int32Ty_, i));
    auto arg = irb_.CreateLoad(argp);
    wrappedArgs.push_back(arg);
  }
  auto cc = irb_.CreateCall(fn_, wrappedArgs);
  irb_.CreateRet(cc);

  // Set insert point to the real function.
  bb_ = llvm::BasicBlock::Create(*context_.getContext(), "entry", fn_);
  irb_.SetInsertPoint(bb_);
}

// TODO: The binary ops are copypasta.

void LLVMCodeGen::visit(const Add* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs().accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFAdd(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateAdd(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch add arg types";
  }
}

void LLVMCodeGen::visit(const Sub* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs().accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFSub(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSub(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch sub arg types";
  }
}

void LLVMCodeGen::visit(const Mul* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs().accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFMul(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateMul(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch mul arg types";
  }
}

void LLVMCodeGen::visit(const Div* v) {
  v->lhs().accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs().accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFDiv(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSDiv(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch div arg types";
  }
}

void LLVMCodeGen::visit(const IntImm* v) {
  value_ = llvm::ConstantInt::getSigned(int32Ty_, v->value());
}

void LLVMCodeGen::visit(const FloatImm* v) {
  value_ = llvm::ConstantFP::get(floatTy_, v->value());
}

void LLVMCodeGen::visit(const Cast* v) {
  v->src_value().accept(this);

  llvm::Type* dstType = nullptr;
  if (v->dtype().scalar_type() == kInt32) {
    dstType = int32Ty_;
  } else if (v->dtype().scalar_type() == kFloat32) {
    dstType = floatTy_;
  }

  if (v->dtype().lanes() > 1) {
    dstType = llvm::VectorType::get(dstType, v->dtype().lanes());
  }

  // Scalar casts
  if (v->dtype() == kInt32 &&
      v->src_value().dtype() == kFloat32) {
    value_ = irb_.CreateFPToSI(value_, dstType);
    return;
  }

  if (v->dtype() == kFloat32 &&
      v->src_value().dtype() == kInt32) {
    value_ = irb_.CreateSIToFP(value_, dstType);
    return;
  }

  LOG(FATAL) << "Unsupported cast!";
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
  v->mask().accept(this);
  auto mask = this->value_;

  // Create block structure for the masked load.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(*context_.getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(*context_.getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::getTrue(int32Ty_));
  irb_.CreateCondBr(cond, condblock, tailblock);

  // Do the load
  irb_.SetInsertPoint(condblock);
  auto addr = irb_.CreateGEP(base, idx);
  auto load = irb_.CreateLoad(addr);
  irb_.CreateBr(tailblock);

  // Merge the masked and unmasked CFG edges
  irb_.SetInsertPoint(tailblock);
  auto phi = irb_.CreatePHI(load->getType(), 2);
  phi->addIncoming(llvm::UndefValue::get(load->getType()), preheader);
  phi->addIncoming(load, condblock);

  value_ = phi;
}

void LLVMCodeGen::visit(const For* v) {
  // Create "start" value.
  v->start().accept(this);
  auto start = this->value_;

  // Create loop preheader and body.
  auto preheader = irb_.GetInsertBlock();
  auto loop = llvm::BasicBlock::Create(*context_.getContext(), "loop", fn_);
  irb_.CreateBr(loop);
  irb_.SetInsertPoint(loop);

  // Set up phi node for index variable.
  auto idx = irb_.CreatePHI(int32Ty_, 2);
  idx->addIncoming(start, preheader);
  varToVal_.emplace(v->var().node(), idx);

  // Codegen the body.
  v->body().accept(this);

  // Create the stop condition. and "after" block.
  auto inc = irb_.CreateAdd(idx, llvm::ConstantInt::getSigned(int32Ty_, 1));
  v->stop().accept(this);
  auto stop = this->value_;
  auto cond = irb_.CreateICmpSLT(inc, stop);

  // Branch back to top of loop and finish phi for index variable.
  auto end_loop = irb_.GetInsertBlock();
  auto after = llvm::BasicBlock::Create(*context_.getContext(), "after", fn_);
  irb_.CreateCondBr(cond, loop, after);
  irb_.SetInsertPoint(after);
  idx->addIncoming(inc, end_loop);
  value_ = llvm::ConstantInt::get(int32Ty_, 0);
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
  v->mask().accept(this);
  auto mask = this->value_;
  v->value().accept(this);
  auto val = this->value_;

  // Create block structure for the masked store.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(*context_.getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(*context_.getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::getTrue(int32Ty_));
  irb_.CreateCondBr(cond, condblock, tailblock);

  // Do the store
  irb_.SetInsertPoint(condblock);
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
  irb_.CreateBr(tailblock);

  // Merge the masked and unmasked CFG edges
  irb_.SetInsertPoint(tailblock);

  value_ = llvm::ConstantInt::get(int32Ty_, 0);
}

void LLVMCodeGen::visit(const Broadcast* v) {
  v->value().accept(this);
  Dtype dtype = v->value().dtype();
  int lanes = v->lanes();
  value_ = irb_.CreateVectorSplat(lanes, value_);
}

void LLVMCodeGen::optimize(llvm::Module& M) {
  llvm::legacy::FunctionPassManager FPM(&M);
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(llvm::createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
  FPM.add(
      llvm::createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  TM->adjustPassManager(PMB);

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
