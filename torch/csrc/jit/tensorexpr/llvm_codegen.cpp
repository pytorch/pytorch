#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"

#include <memory>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/execution_counter.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/types.h"

using namespace torch::jit::tensorexpr;

DEFINE_TRIGGER(llvm_codegen_created);
DEFINE_TRIGGER(llvm_codegen_executed);

static llvm::orc::JITTargetMachineBuilder makeTargetMachineBuilder() {
#if 0
  // FIXME: Switch to using detectHost() rather than setting up the JTMB manually
  // once LLVM 10 is available.
  return llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
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

  JTMB.setCodeGenOptLevel(llvm::CodeGenOpt::Default);
  JTMB.setCPU(llvm::sys::getHostCPUName());
  JTMB.addFeatures(SubtargetFeatures.getFeatures());

  return JTMB;
#endif
}

LLVMCodeGen::LLVMCodeGen(Stmt* stmt)
    : LLVMCodeGen(stmt, std::vector<BufferArg>()) {}

LLVMCodeGen::LLVMCodeGen(
    Stmt* stmt,
    const std::vector<BufferArg>& args,
    Dtype dtype)
    : CodeGen(stmt, args),
      context_(std::make_unique<llvm::LLVMContext>()),
      irb_(getContext()),
      int32Ty_(llvm::Type::getInt32Ty(getContext())),
      floatTy_(llvm::Type::getFloatTy(getContext())) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto JTMB = makeTargetMachineBuilder();
  TM_ = llvm::cantFail(JTMB.createTargetMachine());

  jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>();
  module_ = std::make_unique<llvm::Module>("pytorch", getContext());
  module_->setDataLayout(cantFail(JTMB.getDefaultDataLayoutForTarget()));
  module_->setTargetTriple(JTMB.getTargetTriple().str());

  // Emit prototype and bind argument Vars to parameter indices.
  llvm::Type* retTy = dtypeToLLVM(dtype);
  std::vector<llvm::Type*> params;
  for (int i = 0; i < args.size(); i++) {
    auto const& arg = args[i];
    if (arg.isVar()) {
      params.push_back(dtypeToLLVM(arg.dtype()));
    } else {
      params.push_back(dtypeToLLVMPtr(arg.dtype()));
    }
    varToArg_[arg.var()] = i;
  }
  llvm::FunctionType* fntype = llvm::FunctionType::get(retTy, params, false);
  fn_ = llvm::Function::Create(
      fntype, llvm::Function::PrivateLinkage, "pytorch", module_.get());
  for (int i = 0; i < args.size(); i++) {
    if (!args[i].isVar()) {
      fn_->addParamAttr(i, llvm::Attribute::NoAlias);
    }
  }

  emitWrapper(params);
  emitKernel(stmt, params);

  cantFail(jit_->addModule(
      llvm::orc::ThreadSafeModule(std::move(module_), context_)));
  auto sym = jit_->findSymbol("wrapper");
  kernelAddress_ = cantFail(sym.getAddress());

  USE_TRIGGER(llvm_codegen_created);
}

llvm::LLVMContext& LLVMCodeGen::getContext() {
  return *context_.getContext();
}

llvm::Type* LLVMCodeGen::dtypeToLLVM(Dtype dtype) {
  if (dtype == kInt32) {
    return int32Ty_;
  } else if (dtype == kFloat32) {
    return floatTy_;
  }
  LOG(FATAL) << "Unhandled dtype: " << dtype;
  return nullptr;
}

llvm::Type* LLVMCodeGen::dtypeToLLVMPtr(Dtype dtype) {
  return dtypeToLLVM(dtype)->getPointerTo();
}

void LLVMCodeGen::emitWrapper(const std::vector<llvm::Type*>& params) {
  auto voidPtrPtrTy = llvm::Type::getInt8PtrTy(getContext())->getPointerTo();
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(int32Ty_, {voidPtrPtrTy}, false),
      llvm::Function::ExternalLinkage,
      "wrapper",
      module_.get());
  auto wrapBB = llvm::BasicBlock::Create(getContext(), "wrapBB", wrapper);
  irb_.SetInsertPoint(wrapBB);
  llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
  for (size_t i = 0; i < params.size(); i++) {
    auto argp = irb_.CreateGEP(
        wrapper->arg_begin(), llvm::ConstantInt::getSigned(int32Ty_, i));
    if (params[i]->isPointerTy()) {
      auto arg = irb_.CreatePointerCast(irb_.CreateLoad(argp), params[i]);
      wrappedArgs.push_back(arg);
    } else {
      auto p = irb_.CreatePointerCast(
          irb_.CreateLoad(argp), params[i]->getPointerTo());
      auto arg = irb_.CreateLoad(p);
      wrappedArgs.push_back(arg);
    }
  }
  auto cc = irb_.CreateCall(fn_, wrappedArgs);
  irb_.CreateRet(cc);
}

void LLVMCodeGen::emitKernel(
    Stmt* stmt,
    const std::vector<llvm::Type*>& params) {
  // Set insert point to the real function.
  bb_ = llvm::BasicBlock::Create(getContext(), "entry", fn_);
  irb_.SetInsertPoint(bb_);

  // Compile the kernel.
  stmt->accept(this);
  irb_.CreateRet(value_);

#if DEBUG_PRINT
  llvm::errs() << *module_;
#endif
  CHECK(!llvm::verifyFunction(*fn_, &llvm::outs()))
      << "Function verification failed";
  optimize(*module_);

#if DEBUG_PRINT
  llvm::errs() << *module_;
  llvm::SmallVector<char, 0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  llvm::legacy::PassManager PM;
  TM_->addPassesToEmitFile(
      PM,
      asmStream,
      nullptr,
      llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
  PM.run(*module_);
  llvm::errs() << asmStream.str();
#endif
}

static void* argToPtr(
    const CodeGen::BufferArg& bufferArg,
    const CodeGen::CallArg& callArg) {
  if (!bufferArg.isVar()) {
    return callArg.data();
  }
  if (bufferArg.dtype() == kInt32) {
    return callArg.intPtr();
  }
  if (bufferArg.dtype() == kFloat32) {
    return callArg.floatPtr();
  }
  LOG(FATAL) << "Unhandled dtype for arg: " << bufferArg.var()->name_hint()
             << "dtype=" << bufferArg.var()->dtype();
  return nullptr;
}

void LLVMCodeGen::call(const std::vector<CallArg>& args) {
  CHECK_EQ(args.size(), buffer_args().size())
      << "args: " << args.size() << ", buffers: " << buffer_args().size();
  for (size_t i = 0; i < buffer_args().size(); i++) {
    auto const& bufferArg = buffer_args()[i];
    auto const& callArg = args[i];
    args_.push_back(argToPtr(bufferArg, callArg));
  }
  value<float>(args_);
  args_.clear();
  USE_TRIGGER(llvm_codegen_executed);
}

// TODO: The binary ops are copypasta.

void LLVMCodeGen::visit(const Add* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
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
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
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
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
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
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
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

void LLVMCodeGen::visit(const And* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateAnd(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch And arg types";
  }
}

void LLVMCodeGen::visit(const Xor* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateXor(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch And arg types";
  }
}

void LLVMCodeGen::visit(const Lshift* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateShl(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch And arg types";
  }
}

void LLVMCodeGen::visit(const Rshift* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFloatingPointTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFloatingPointTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateLShr(lhs, rhs);
  } else {
    LOG(FATAL) << "Unhandled mismatch And arg types";
  }
}

void LLVMCodeGen::visit(const Mod* v) {
  throw std::runtime_error("Mod unsupported in LLVM codegen yet");
}

void LLVMCodeGen::visit(const Max* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;

  if (v->dtype() == kInt32) {
    auto icmp = irb_.CreateICmpSGT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  if (v->propagate_nans()) {
    value_ = irb_.CreateBinaryIntrinsic(llvm::Intrinsic::maximum, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(llvm::FCmpInst::FCMP_OGT, lhs, rhs), lhs, rhs);
}

void LLVMCodeGen::visit(const Min* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;

  if (v->dtype() == kInt32) {
    auto icmp = irb_.CreateICmpSLT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  if (v->propagate_nans()) {
    value_ = irb_.CreateBinaryIntrinsic(llvm::Intrinsic::minimum, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(llvm::FCmpInst::FCMP_OLT, lhs, rhs), lhs, rhs);
}

void LLVMCodeGen::visit(const CompareSelect* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;
  v->ret_val1()->accept(this);
  auto retval1 = this->value_;
  v->ret_val2()->accept(this);
  auto retval2 = this->value_;

  auto type_used = v->lhs()->dtype();

  llvm::Value* cmp_;
  CompareSelectOperation cmp_op_ = v->compare_select_op();

  if (type_used == kInt32) {
    switch (cmp_op_) {
      case CompareSelectOperation::kEQ:
        cmp_ = irb_.CreateICmpEQ(lhs, rhs);
        break;
      case CompareSelectOperation::kNE:
        cmp_ = irb_.CreateICmpNE(lhs, rhs);
        break;
      case CompareSelectOperation::kGT:
        cmp_ = irb_.CreateICmpSGT(lhs, rhs);
        break;
      case CompareSelectOperation::kGE:
        cmp_ = irb_.CreateICmpSGE(lhs, rhs);
        break;
      case CompareSelectOperation::kLT:
        cmp_ = irb_.CreateICmpSLT(lhs, rhs);
        break;
      case CompareSelectOperation::kLE:
        cmp_ = irb_.CreateICmpSLE(lhs, rhs);
        break;
      default:
        // TODO: change to a proper error report
        throw std::runtime_error("invalid operator type");
    }
  } else { // FP32
    switch (cmp_op_) {
      case CompareSelectOperation::kEQ:
        cmp_ = irb_.CreateFCmpOEQ(lhs, rhs);
        break;
      case CompareSelectOperation::kNE:
        cmp_ = irb_.CreateFCmpONE(lhs, rhs);
        break;
      case CompareSelectOperation::kGT:
        cmp_ = irb_.CreateFCmpOGT(lhs, rhs);
        break;
      case CompareSelectOperation::kGE:
        cmp_ = irb_.CreateFCmpOGE(lhs, rhs);
        break;
      case CompareSelectOperation::kLT:
        cmp_ = irb_.CreateFCmpOLT(lhs, rhs);
        break;
      case CompareSelectOperation::kLE:
        cmp_ = irb_.CreateFCmpOLE(lhs, rhs);
        break;
      default:
        // TODO: change to a proper error report
        throw std::runtime_error("invalid operator type");
    }
  }

  value_ = irb_.CreateSelect(cmp_, retval1, retval2);
  return;
}

void LLVMCodeGen::visit(const IntImm* v) {
  value_ = llvm::ConstantInt::getSigned(int32Ty_, v->value());
}

void LLVMCodeGen::visit(const FloatImm* v) {
  value_ = llvm::ConstantFP::get(floatTy_, v->value());
}

void LLVMCodeGen::visit(const Cast* v) {
  v->src_value()->accept(this);

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
  if (v->dtype() == kInt32 && v->src_value()->dtype() == kFloat32) {
    value_ = irb_.CreateFPToSI(value_, dstType);
    return;
  }

  if (v->dtype() == kFloat32 && v->src_value()->dtype() == kInt32) {
    value_ = irb_.CreateSIToFP(value_, dstType);
    return;
  }

  LOG(FATAL) << "Unsupported cast!";
}

void LLVMCodeGen::visit(const Var* v) {
  if (varToArg_.count(v)) {
    auto idx = varToArg_.at(v);
    auto arg = fn_->arg_begin() + idx;
    value_ = arg;
  } else if (varToVal_.count(v)) {
    value_ = varToVal_.at(v);
  }
}

void LLVMCodeGen::visit(const Let* v) {
  const Var* var = dynamic_cast<const Var*>(v->var());
  CHECK(var != nullptr);
  v->value()->accept(this);
  auto value = value_;
  if (!varToVal_.count(var)) {
    varToVal_.emplace(var, value);
  } else {
    throw std::runtime_error("var should not exist before");
  }
  v->body()->accept(this);
  if (varToVal_.count(var)) {
    varToVal_.erase(var);
  } else {
    throw std::runtime_error("erasing var that doesn't exist");
  }
}

// TODO: refactor this and merge with Let
void LLVMCodeGen::visit(const LetStmt* v) {
  const Var* var = v->var();
  CHECK(var != nullptr);
  v->value()->accept(this);
  auto value = value_;
  if (!varToVal_.count(var)) {
    varToVal_.emplace(var, value);
  } else {
    throw std::runtime_error("var should not exist before");
  }
  v->body()->accept(this);
  if (varToVal_.count(var)) {
    varToVal_.erase(var);
  } else {
    throw std::runtime_error("erasing var that doesn't exist");
  }
}

void LLVMCodeGen::visit(const Ramp* v) {
  v->base()->accept(this);
  auto base = this->value_;
  v->stride()->accept(this);
  auto stride = this->value_;
  int lanes = v->lanes();

  llvm::Type* vecType = nullptr;
  if (v->dtype().scalar_type() == kInt32) {
    vecType = llvm::VectorType::get(int32Ty_, lanes);
  } else if (v->dtype().scalar_type() == kFloat32) {
    vecType = llvm::VectorType::get(floatTy_, lanes);
  }

  value_ = llvm::UndefValue::get(vecType);
  for (int i = 0; i < lanes; ++i) {
    value_ = irb_.CreateInsertElement(value_, base, i);
    base = irb_.CreateAdd(base, stride);
  }
}

llvm::Value* LLVMCodeGen::emitUnmaskedLoad(
    llvm::Value* base,
    llvm::Value* idx) {
  auto addr = irb_.CreateGEP(base, idx);
  return irb_.CreateLoad(addr);
}

llvm::Value* LLVMCodeGen::emitMaskedLoad(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask) {
  // Create block structure for the masked load.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(int32Ty_, 1));
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

  return phi;
}

void LLVMCodeGen::visit(const Load* v) {
  v->base_handle()->accept(this);
  auto base = this->value_;
  v->index()->accept(this);
  auto idx = this->value_;
  v->mask()->accept(this);
  auto mask = this->value_;

  if (v->dtype().lanes() == 1) {
    auto* maskimm = dynamic_cast<const IntImm*>(v->mask());
    if (maskimm && maskimm->value() == 1) {
      value_ = emitUnmaskedLoad(base, idx);
    } else {
      value_ = emitMaskedLoad(base, idx, mask);
    }
    return;
  }

  llvm::Type* loadType = nullptr;
  if (v->dtype().scalar_type() == kInt32) {
    loadType = llvm::VectorType::get(int32Ty_, v->dtype().lanes());
  } else if (v->dtype().scalar_type() == kFloat32) {
    loadType = llvm::VectorType::get(floatTy_, v->dtype().lanes());
  }

  // Detect whether the vector mask is all true
  bool unmasked_load = false;
  auto* mask_broadcast = dynamic_cast<const Broadcast*>(v->mask());
  if (mask_broadcast) {
    auto* broadcast_imm = dynamic_cast<const IntImm*>(mask_broadcast->value());
    if (broadcast_imm && broadcast_imm->value() == 1) {
      unmasked_load = true;
    }
  }

  // Handle the case where the load is contiguous and unmasked efficiently
  auto* idx_ramp = dynamic_cast<const Ramp*>(v->index());
  if (unmasked_load && idx_ramp) {
    auto* stride_imm = dynamic_cast<const IntImm*>(idx_ramp->stride());
    if (stride_imm && stride_imm->value() == 1) {
      auto first_idx = irb_.CreateExtractElement(idx, uint64_t{0ULL});
      auto addr = irb_.CreateGEP(base, first_idx);
      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(loadType, 0));
      value_ = irb_.CreateAlignedLoad(loadType, vaddr, 4);
      return;
    }
  }

  // Fallback to a scalar implementation
  llvm::Value* load = llvm::UndefValue::get(loadType);
  for (int i = 0; i < v->dtype().lanes(); ++i) {
    auto sub_idx = irb_.CreateExtractElement(idx, i);
    llvm::Value* sub_load = nullptr;
    if (unmasked_load) {
      sub_load = emitUnmaskedLoad(base, sub_idx);
    } else {
      auto sub_mask = irb_.CreateExtractElement(mask, i);
      sub_load = emitMaskedLoad(base, sub_idx, sub_mask);
    }
    load = irb_.CreateInsertElement(load, sub_load, i);
  }

  value_ = load;
}

void LLVMCodeGen::visit(const For* v) {
  // Create "start" value.
  v->start()->accept(this);
  auto start = this->value_;

  // Create loop preheader and body.
  auto preheader = irb_.GetInsertBlock();
  auto loop = llvm::BasicBlock::Create(getContext(), "loop", fn_);
  irb_.CreateBr(loop);
  irb_.SetInsertPoint(loop);

  // Set up phi node for index variable.
  auto idx = irb_.CreatePHI(int32Ty_, 2);
  idx->addIncoming(start, preheader);
  varToVal_.emplace(v->var(), idx);

  // Codegen the body.
  if (v->body()) {
    v->body()->accept(this);
  }

  // Create the stop condition. and "after" block.
  auto inc = irb_.CreateAdd(idx, llvm::ConstantInt::getSigned(int32Ty_, 1));
  v->stop()->accept(this);
  auto stop = this->value_;
  auto cond = irb_.CreateICmpSLT(inc, stop);

  // Branch back to top of loop and finish phi for index variable.
  auto end_loop = irb_.GetInsertBlock();
  auto after = llvm::BasicBlock::Create(getContext(), "after", fn_);
  irb_.CreateCondBr(cond, loop, after);
  irb_.SetInsertPoint(after);
  idx->addIncoming(inc, end_loop);
  value_ = llvm::ConstantInt::get(int32Ty_, 0);
}

void LLVMCodeGen::visit(const Block* v) {
  for (int i = 0; i < v->nstmts(); i++) {
    v->stmt(i)->accept(this);
  }
}

void LLVMCodeGen::emitUnmaskedStore(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* val) {
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
}

void LLVMCodeGen::emitMaskedStore(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask,
    llvm::Value* val) {
  // Create block structure for the masked store.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(int32Ty_, 1));
  irb_.CreateCondBr(cond, condblock, tailblock);

  // Do the store
  irb_.SetInsertPoint(condblock);
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
  irb_.CreateBr(tailblock);

  // Merge the masked and unmasked CFG edges
  irb_.SetInsertPoint(tailblock);
}

void LLVMCodeGen::visit(const Store* v) {
  v->base_handle()->accept(this);
  auto base = this->value_;
  v->index()->accept(this);
  auto idx = this->value_;
  v->mask()->accept(this);
  auto mask = this->value_;
  v->value()->accept(this);
  auto val = this->value_;

  value_ = llvm::ConstantInt::get(int32Ty_, 0);

  if (v->value()->dtype().lanes() == 1) {
    auto* maskimm = dynamic_cast<const IntImm*>(v->mask());
    if (maskimm && maskimm->value() == 1) {
      emitUnmaskedStore(base, idx, val);
    } else {
      emitMaskedStore(base, idx, mask, val);
    }
    return;
  }

  // Detect whether the vector mask is all true
  bool unmasked_store = false;
  auto* mask_broadcast = dynamic_cast<const Broadcast*>(v->mask());
  if (mask_broadcast) {
    auto* broadcast_imm = dynamic_cast<const IntImm*>(mask_broadcast->value());
    if (broadcast_imm && broadcast_imm->value() == 1) {
      unmasked_store = true;
    }
  }

  // Handle the case where the store is contiguous and unmasked efficiently
  auto* idx_ramp = dynamic_cast<const Ramp*>(v->index());
  if (unmasked_store && idx_ramp) {
    auto* stride_imm = dynamic_cast<const IntImm*>(idx_ramp->stride());
    if (stride_imm && stride_imm->value() == 1) {
      auto first_idx = irb_.CreateExtractElement(idx, uint64_t{0});
      auto addr = irb_.CreateGEP(base, first_idx);
      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(val->getType(), 0));
      irb_.CreateAlignedStore(val, vaddr, 4);
      return;
    }
  }

  // Fallback to a scalar implementation
  for (int i = 0; i < v->value()->dtype().lanes(); ++i) {
    auto sub_idx = irb_.CreateExtractElement(idx, i);
    auto sub_val = irb_.CreateExtractElement(val, i);
    if (unmasked_store) {
      emitUnmaskedStore(base, sub_idx, sub_val);
    } else {
      auto sub_mask = irb_.CreateExtractElement(mask, i);
      emitMaskedStore(base, sub_idx, sub_mask, sub_val);
    }
  }
}

void LLVMCodeGen::visit(const Broadcast* v) {
  v->value()->accept(this);
  int lanes = v->lanes();
  value_ = irb_.CreateVectorSplat(lanes, value_);
}

void LLVMCodeGen::visit(const IfThenElse* v) {
  v->condition()->accept(this);
  llvm::Value* condition = value_;
  llvm::Value* c =
      irb_.CreateICmpNE(condition, llvm::ConstantInt::get(int32Ty_, 0));

  auto then_block = llvm::BasicBlock::Create(getContext(), "then", fn_);
  auto else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
  auto end_block = llvm::BasicBlock::Create(getContext(), "block", fn_);
  irb_.CreateCondBr(c, then_block, else_block);

  irb_.SetInsertPoint(then_block);
  v->true_value()->accept(this);
  llvm::Value* then_val = value_;
  then_block = irb_.GetInsertBlock();
  irb_.CreateBr(end_block);

  irb_.SetInsertPoint(else_block);
  v->false_value()->accept(this);
  llvm::Value* else_val = value_;
  else_block = irb_.GetInsertBlock();
  irb_.CreateBr(end_block);

  irb_.SetInsertPoint(end_block);
  llvm::PHINode* phi = irb_.CreatePHI(then_val->getType(), 2);
  phi->addIncoming(then_val, then_block);
  phi->addIncoming(else_val, else_block);
  value_ = phi;
}

void LLVMCodeGen::visit(const BaseCallNode* v) {
  LOG(FATAL) << "Unimplemented: BaseCall";
}

static void applyMathFunctionAttributes(llvm::Function* f) {
  f->addFnAttr(llvm::Attribute::ReadNone);
  f->addFnAttr(llvm::Attribute::NoFree);
  f->addFnAttr(llvm::Attribute::NoUnwind);
  f->addFnAttr(llvm::Attribute::Speculatable);
  f->addFnAttr(llvm::Attribute::WillReturn);
}

void LLVMCodeGen::visit(const Intrinsics* v) {
  llvm::FunctionType* call_ty = nullptr;
  llvm::Value* call_fn = nullptr;

  switch (v->op_type()) {
#define UNARY_INTRIN_CASE(enum, intrin)                 \
  case enum: {                                          \
    v->params().front()->accept(this);                   \
    value_ = irb_.CreateUnaryIntrinsic(intrin, value_); \
    return;                                             \
  } break;
    UNARY_INTRIN_CASE(kLog10, llvm::Intrinsic::log10)
    UNARY_INTRIN_CASE(kLog, llvm::Intrinsic::log)
    UNARY_INTRIN_CASE(kLog2, llvm::Intrinsic::log2)
    UNARY_INTRIN_CASE(kExp, llvm::Intrinsic::exp)
    UNARY_INTRIN_CASE(kCos, llvm::Intrinsic::cos)
    UNARY_INTRIN_CASE(kSin, llvm::Intrinsic::sin)
    UNARY_INTRIN_CASE(kSqrt, llvm::Intrinsic::sqrt)
    UNARY_INTRIN_CASE(kFabs, llvm::Intrinsic::fabs)
    UNARY_INTRIN_CASE(kFloor, llvm::Intrinsic::floor)
    UNARY_INTRIN_CASE(kCeil, llvm::Intrinsic::ceil)
    UNARY_INTRIN_CASE(kTrunc, llvm::Intrinsic::trunc)
    UNARY_INTRIN_CASE(kRound, llvm::Intrinsic::round)
#undef UNARY_INTRIN_CASE

    case kRsqrt: {
      v->params().front()->accept(this);
      value_ = irb_.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, value_);
      llvm::Value* constant = llvm::ConstantFP::get(floatTy_, 1.0);
      if (v->dtype().lanes() > 1) {
        constant = irb_.CreateVectorSplat(v->dtype().lanes(), constant);
      }
      value_ = irb_.CreateFDiv(constant, value_);
      return;
    } break;

#define UNARY_MATH_CASE(enum, name, type)                             \
  case enum: {                                                        \
    auto callee = module_->getOrInsertFunction(                       \
        name, llvm::FunctionType::get(type, {type}, false), {});      \
    call_ty = callee.getFunctionType();                               \
    call_fn = callee.getCallee();                                     \
    applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn)); \
  } break;
      UNARY_MATH_CASE(kErf, "erff", floatTy_)
      UNARY_MATH_CASE(kErfc, "erfcf", floatTy_)
      UNARY_MATH_CASE(kTan, "tanf", floatTy_)
      UNARY_MATH_CASE(kAcos, "acosf", floatTy_)
      UNARY_MATH_CASE(kAsin, "asinf", floatTy_)
      UNARY_MATH_CASE(kAtan, "atanf", floatTy_)
      UNARY_MATH_CASE(kCosh, "coshf", floatTy_)
      UNARY_MATH_CASE(kSinh, "sinhf", floatTy_)
      UNARY_MATH_CASE(kTanh, "tanhf", floatTy_)
      UNARY_MATH_CASE(kExpm1, "expm1f", floatTy_)
      UNARY_MATH_CASE(kLgamma, "lgammaf", floatTy_)
#undef UNARY_MATH_CASE

#define BINARY_MATH_CASE(enum, name, type)                             \
  case enum: {                                                         \
    auto callee = module_->getOrInsertFunction(                        \
        name, llvm::FunctionType::get(type, {type, type}, false), {}); \
    call_ty = callee.getFunctionType();                                \
    call_fn = callee.getCallee();                                      \
    applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn));  \
  } break;
      BINARY_MATH_CASE(kRemainder, "remainderf", floatTy_)
      BINARY_MATH_CASE(kAtan2, "atan2f", floatTy_)
      BINARY_MATH_CASE(kPow, "powf", floatTy_)
      BINARY_MATH_CASE(kFmod, "fmodf", floatTy_)
#undef BINARY_MATH_CASE

    default: {
      LOG(FATAL) << "Unimplemented: Intrinsics: " << ExprHandle(v);
    } break;
  }

  std::vector<llvm::Value*> params;
  for (auto& p : v->params()) {
    p->accept(this);
    params.push_back(value_);
  }

  if (v->dtype().lanes() == 1) {
    value_ = irb_.CreateCall(call_ty, call_fn, params);
  } else {
    llvm::Type* vecType = llvm::VectorType::get(floatTy_, v->dtype().lanes());
    value_ = llvm::UndefValue::get(vecType);
    for (int i = 0; i < v->dtype().lanes(); ++i) {
      std::vector<llvm::Value*> call_operands;
      for (auto p : params) {
        call_operands.push_back(irb_.CreateExtractElement(p, i));
      }

      llvm::Value* val = irb_.CreateCall(call_ty, call_fn, call_operands);
      value_ = irb_.CreateInsertElement(value_, val, i);
    }
  }
}

void LLVMCodeGen::visit(const FunctionCall* v) {
  LOG(FATAL) << "Unimplemented: FunctionCall";
}

void LLVMCodeGen::visit(const Allocate* v) {
  LOG(FATAL) << "Unimplemented: Allocate";
}

void LLVMCodeGen::visit(const Free* v) {
  LOG(FATAL) << "Unimplemented: Free";
}

void LLVMCodeGen::visit(const Cond* v) {
  LOG(FATAL) << "Unimplemented: Cond";
}

void LLVMCodeGen::optimize(llvm::Module& M) {
  llvm::legacy::FunctionPassManager FPM(&M);
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(
      llvm::createTargetTransformInfoWrapperPass(TM_->getTargetIRAnalysis()));
  FPM.add(
      llvm::createTargetTransformInfoWrapperPass(TM_->getTargetIRAnalysis()));

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  TM_->adjustPassManager(PMB);

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

RegisterCodeGen<LLVMCodeGen> reg("llvm_codegen");

#endif // ENABLE_LLVM
