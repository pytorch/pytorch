#ifdef TORCH_ENABLE_LLVM

#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>

#include <aten/src/ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/llvm_jit.h>

#include <memory>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>

#if LLVM_VERSION_MAJOR >= 10
#include <llvm/Support/CodeGen.h>
#else
#include <llvm/Target/TargetMachine.h>
#endif

#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>

#if LLVM_VERSION_MAJOR >= 11
#include <llvm/Support/TypeSize.h>
#endif

#include <torch/csrc/jit/tensorexpr/execution_counter.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/half_support.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <torch/csrc/jit/jit_log.h>

using namespace torch::jit::tensorexpr;

C10_DEFINE_bool(
    torch_jit_llvm_use_fast_intrinsics,
    false,
    "Use fast (but slightly less accurate) implementations of tanh and sigmoid");

DEFINE_TRIGGER(llvm_codegen_created);
DEFINE_TRIGGER(llvm_codegen_executed);

namespace torch {
namespace jit {
namespace tensorexpr {
DEFINE_TRIGGER(llvm_codegen_parallel_dispatched);
namespace {

llvm::CmpInst::Predicate llvm_comparison_predicate(
    CompareSelectOperation compare_op,
    const ScalarType& type) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::ICmpInst::ICMP_EQ;
    case CompareSelectOperation::kNE:
      return llvm::ICmpInst::ICMP_NE;
    case CompareSelectOperation::kGT:
      return is_signed(type) ? llvm::ICmpInst::ICMP_SGT
                             : llvm::ICmpInst::ICMP_UGT;
    case CompareSelectOperation::kGE:
      return is_signed(type) ? llvm::ICmpInst::ICMP_SGE
                             : llvm::ICmpInst::ICMP_UGE;
    case CompareSelectOperation::kLT:
      return is_signed(type) ? llvm::ICmpInst::ICMP_SLT
                             : llvm::ICmpInst::ICMP_ULT;
    case CompareSelectOperation::kLE:
      return is_signed(type) ? llvm::ICmpInst::ICMP_SLE
                             : llvm::ICmpInst::ICMP_ULE;
    default:
      // TODO: change to a proper error report
      throw std::runtime_error("invalid operator type");
  }
}

llvm::CmpInst::Predicate llvm_fp_comparison_predicate(
    CompareSelectOperation compare_op) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::FCmpInst::FCMP_OEQ;
    case CompareSelectOperation::kNE:
      return llvm::FCmpInst::FCMP_ONE;
    case CompareSelectOperation::kGT:
      return llvm::FCmpInst::FCMP_OGT;
    case CompareSelectOperation::kGE:
      return llvm::FCmpInst::FCMP_OGE;
    case CompareSelectOperation::kLT:
      return llvm::FCmpInst::FCMP_OLT;
    case CompareSelectOperation::kLE:
      return llvm::FCmpInst::FCMP_OLE;
    default:
      // TODO: change to a proper error report
      throw std::runtime_error("invalid operator type");
  }
}

#if LLVM_VERSION_MAJOR <= 9
int ElementCount(int lanes) {
  return lanes;
}
#else
llvm::ElementCount ElementCount(int lanes) {
#if LLVM_VERSION_MAJOR <= 11
  return llvm::ElementCount(static_cast<unsigned>(lanes), false);
#elif LLVM_VERSION_MAJOR >= 12
  return llvm::ElementCount::getFixed(lanes);
#else
#error Only LLVM versions 8 and above are supported.
#endif
}
#endif

#if LLVM_VERSION_MAJOR >= 9

using FunctionCallee = llvm::FunctionCallee;

#elif LLVM_VERSION_MAJOR == 8 && LLVM_VERSION_PATCH == 20181009

struct FunctionCallee {
  FunctionCallee() {}

  FunctionCallee(llvm::Constant* fn)
      : v_(fn), ft_(cast<llvm::Function>(v_)->getFunctionType()) {}

  llvm::FunctionType* getFunctionType() {
    return ft_;
  }

  llvm::Value* getCallee() {
    return v_;
  }

 private:
  llvm::Value* v_{nullptr};
  llvm::FunctionType* ft_{nullptr};
};

#else
#error Only LLVM versions 8 and above are supported.
#endif
} // namespace

class LLVMCodeGenImpl : public IRVisitor {
 private:
  std::unique_ptr<llvm::LLVMContext> context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_{nullptr};
  llvm::JITTargetAddress kernelAddress_;

#define LLVM_TYPE_DECLARE(_1, Name) llvm::Type* Name##Ty_;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, LLVM_TYPE_DECLARE);
#undef LLVM_TYPE_DECLARE
  llvm::Type* Int8PtrTy_;
  llvm::Type* VoidTy_;

  std::unordered_map<const Var*, int> varToArg_;
  std::unordered_map<const Var*, llvm::Value*> varToVal_;
  std::unordered_map<const Block*, std::vector<const Var*>> scopeToVar_;
  const Block* scope_;

  std::string llvmCode;
  std::string asmCode;

 private:
  llvm::LLVMContext& getContext();
  llvm::Type* dtypeToLLVM(Dtype dtype);
  llvm::Type* dtypeToLLVMPtr(Dtype dtype);
  void emitWrapper(const std::vector<llvm::Type*>& params);
  void emitKernel(Stmt* stmt, const std::vector<llvm::Type*>& params);
  llvm::Value* toVec(llvm::Value* v, int lanes);

  enum Arity {
    Unary,
    Binary,
  };

  using SimdCallee = std::tuple<llvm::FunctionType*, llvm::Value*, bool>;
  SimdCallee getSimdFunction(
      const std::string& name,
      llvm::Type* type,
      Arity arity,
      int lanes);

  llvm::Value* varToValue(const Var* var);
  void replaceVarMapping(
      const std::vector<const Var*>& vars,
      const std::vector<llvm::Value*>& vals);
  llvm::Value* packFuncArgs(const std::vector<llvm::Value*>& func_args);
  std::vector<llvm::Value*> unpackFuncArgs(llvm::Value* packed, int arg_count);
  void processParallelFor(const For* v);

 public:
  LLVMCodeGenImpl(
      Stmt* stmt,
      const std::vector<CodeGen::BufferArg>& args,
      at::Device device,
      Dtype dtype);
  ~LLVMCodeGenImpl() = default;

  llvm::JITTargetAddress getKernelAddress() const;

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
  void visit(const BitCast* v) override;
  void visit(const Var* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
  void visit(const Store* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Let* v) override;
  void visit(const Cond* v) override;
  void visit(const ExternalCall* v) override;

  void emitIsNan(const Intrinsics* v);

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
  std::string getLLVMCodeText() {
    return llvmCode;
  }
  std::string getASMCodeText() {
    return asmCode;
  }
};

typedef void (*ParallelCallee)(int index, int8_t* packed_data);
void DispatchParallel(int8_t* func, int start, int stop, int8_t* packed_data) {
  // TODO: preserve the func type.
  ParallelCallee callee = reinterpret_cast<ParallelCallee>(func);
  at::parallel_for(start, stop, 1, [&](int64_t f_begin, int64_t f_end) {
    for (int index = f_begin; index < f_end; index++) {
      callee(index, packed_data);
    }
  });
  USE_TRIGGER(llvm_codegen_parallel_dispatched);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

LLVMCodeGen::~LLVMCodeGen() = default;

LLVMCodeGen::LLVMCodeGen(Stmt* stmt)
    : LLVMCodeGen(stmt, std::vector<CodeGen::BufferArg>()) {}

LLVMCodeGen::LLVMCodeGen(
    Stmt* stmt,
    const std::vector<BufferArg>& args,
    at::Device device,
    const std::string& kernel_func_name,
    Dtype dtype)
    : CodeGen(stmt, args, device, kernel_func_name),
      impl_(std::make_unique<LLVMCodeGenImpl>(stmt, args, device, dtype)) {}

static void* argToPtr(
    const CodeGen::BufferArg& bufferArg,
    const CodeGen::CallArg& callArg) {
  if (!bufferArg.isVar()) {
    return callArg.data();
  }

  switch (bufferArg.dtype().scalar_type()) {
#define TYPE_CASE(_1, Name) \
  case ScalarType::Name:    \
    return callArg.Name##Ptr();
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE

    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

void LLVMCodeGen::call(const std::vector<CallArg>& args) {
  const auto& buf_args = buffer_args();
  if (args.size() != buf_args.size()) {
    throw malformed_input("wrong number of args in call");
  }

  constexpr unsigned nargs = 8;
  c10::SmallVector<void*, nargs> argv;
  argv.resize(buf_args.size());
  for (size_t i = 0, e = buf_args.size(); i < e; i++) {
    auto const& bufferArg = buf_args[i];
    auto const& callArg = args[i];
    argv[i] = argToPtr(bufferArg, callArg);
  }
  value<float>(argv.data());
  USE_TRIGGER(llvm_codegen_executed);
}

at::Tensor LLVMCodeGen::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return at::native::empty_strided_cpu(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

void* LLVMCodeGen::getKernelAddress(LLVMCodeGenImpl* impl) {
  return (void*)impl->getKernelAddress();
}

std::string LLVMCodeGen::getCodeText(const std::string& attr /*=""*/) {
  if (attr == "asm") {
    return impl_->getASMCodeText();
  } else {
    return impl_->getLLVMCodeText();
  }
}

llvm::JITTargetAddress LLVMCodeGenImpl::getKernelAddress() const {
  return kernelAddress_;
}

LLVMCodeGenImpl::LLVMCodeGenImpl(
    Stmt* stmt,
    const std::vector<CodeGen::BufferArg>& args,
    at::Device device,
    Dtype dtype)
    : context_(std::make_unique<llvm::LLVMContext>()), irb_(getContext()) {
  // Manually map types to LLVM types.
  ByteTy_ = llvm::Type::getInt8Ty(getContext());
  CharTy_ = llvm::Type::getInt8Ty(getContext());
  ShortTy_ = llvm::Type::getInt16Ty(getContext());
  IntTy_ = llvm::Type::getInt32Ty(getContext());
  LongTy_ = llvm::Type::getInt64Ty(getContext());
  HalfTy_ = llvm::Type::getHalfTy(getContext());
  FloatTy_ = llvm::Type::getFloatTy(getContext());
  DoubleTy_ = llvm::Type::getDoubleTy(getContext());
  Int8PtrTy_ = llvm::Type::getInt8PtrTy(getContext());
  VoidTy_ = llvm::Type::getVoidTy(getContext());
  BoolTy_ = ByteTy_;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>();
  module_ = std::make_unique<llvm::Module>("pytorch", getContext());
  module_->setDataLayout(jit_->getDataLayout());
  module_->setTargetTriple(jit_->getTargetMachine().getTargetTriple().str());

  // We support float16 ops by casting expr inputs to float32
  // and then casting the result back to float16
  HalfRewriter hsFix;
  stmt = stmt->accept_mutator(&hsFix);

  // Emit prototype and bind argument Vars to parameter indices.
  llvm::Type* retTy = dtypeToLLVM(dtype);
  std::vector<llvm::Type*> params;
  for (size_t i = 0; i < args.size(); i++) {
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
  fn_->addAttribute(
      llvm::AttributeList::AttrIndex::FunctionIndex,
      llvm::Attribute::AlwaysInline);
  for (size_t i = 0; i < args.size(); i++) {
    if (!args[i].isVar()) {
      fn_->addParamAttr(i, llvm::Attribute::NoAlias);
    }
  }

  emitWrapper(params);
  emitKernel(stmt, params);

  jit_->addModule(std::move(module_), std::move(context_));
  auto sym = jit_->findSymbol("wrapper");
  kernelAddress_ = assertSuccess(sym.getAddress());

  USE_TRIGGER(llvm_codegen_created);
}

llvm::LLVMContext& LLVMCodeGenImpl::getContext() {
  return *context_;
}

llvm::Type* LLVMCodeGenImpl::dtypeToLLVM(Dtype dtype) {
  switch (dtype.scalar_type()) {
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return n##Ty_;       \
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

llvm::Type* LLVMCodeGenImpl::dtypeToLLVMPtr(Dtype dtype) {
  return dtypeToLLVM(dtype)->getPointerTo();
}

void LLVMCodeGenImpl::emitWrapper(const std::vector<llvm::Type*>& params) {
  auto voidPtrPtrTy = llvm::Type::getInt8PtrTy(getContext())->getPointerTo();
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(IntTy_, {voidPtrPtrTy}, false),
      llvm::Function::ExternalLinkage,
      "wrapper",
      module_.get());
  auto wrapBB = llvm::BasicBlock::Create(getContext(), "wrapBB", wrapper);
  irb_.SetInsertPoint(wrapBB);
  llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
  for (size_t i = 0; i < params.size(); i++) {
    auto argp = irb_.CreateGEP(
        wrapper->arg_begin(), llvm::ConstantInt::getSigned(IntTy_, i));
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

class LLVMIntrinsicsExpander : public GenericIntrinsicsExpander {
 private:
  const Expr* mutate(const Intrinsics* v) {
    if (v->op_type() == kTanh) {
      ScalarType stype = v->dtype().scalar_type();
      if (stype == ScalarType::Float) {
        return fast_tanh(v->param(0)->accept_mutator(this)).node();
      }
    } else if (v->op_type() == kSigmoid) {
      ScalarType stype = v->dtype().scalar_type();
      if (stype == ScalarType::Float) {
        return fast_sigmoid(v->param(0)->accept_mutator(this)).node();
      }
    }
    // TODO: fast exp
    // TODO: fast erf
    // TODO: fast sigmoid
    return GenericIntrinsicsExpander::mutate(v);
  }
};

void LLVMCodeGenImpl::emitKernel(
    Stmt* stmt,
    const std::vector<llvm::Type*>& params) {
  // Set insert point to the real function.
  bb_ = llvm::BasicBlock::Create(getContext(), "entry", fn_);
  irb_.SetInsertPoint(bb_);

  // Maybe expand some of the intrinsics.
  if (FLAGS_torch_jit_llvm_use_fast_intrinsics) {
    LLVMIntrinsicsExpander intrinsics_expander;
    stmt = stmt->accept_mutator(&intrinsics_expander);
  } else {
    GenericIntrinsicsExpander intrinsics_expander;
    stmt = stmt->accept_mutator(&intrinsics_expander);
  }

  // Compile the kernel.
  stmt->accept(this);

  // If the kernel is empty, set a default return value.
  if (value_ == nullptr) {
    value_ = llvm::ConstantInt::get(IntTy_, 0);
  }

  irb_.CreateRet(value_);

  if (llvm::verifyFunction(*fn_, &llvm::outs())) {
    throw std::runtime_error("Function verification failed");
  }

  // print graph debug info before optimization
  llvm::SmallVector<char, 0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  if (GRAPH_DEBUG_ENABLED) {
    module_->print(asmStream, nullptr);
  }
  GRAPH_DEBUG(
      "\nLLVM module before optimizations\n\n", asmStream.str().str(), "\n");

  optimize(*module_);

  // print graph debug info after optimization
  asmBuffer.set_size(0);
  module_->print(asmStream, nullptr);
  llvmCode = asmStream.str().str();
  asmBuffer.set_size(0);
  llvm::legacy::PassManager PM;
  jit_->getTargetMachine().addPassesToEmitFile(
      PM,
      asmStream,
      nullptr,
#if LLVM_VERSION_MAJOR >= 10
      llvm::CodeGenFileType::CGFT_AssemblyFile);
#else
      llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#endif
  PM.run(*module_);
  asmCode = asmStream.str().str();

  GRAPH_DEBUG(
      "\nLLVM module after optimizations\n\n", llvmCode, "\n", asmCode, "\n");
}

// TODO: The binary ops are copypasta.

void LLVMCodeGenImpl::visit(const Add* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFAdd(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateAdd(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Add", v);
  }
}

void LLVMCodeGenImpl::visit(const Sub* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFSub(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSub(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Sub", v);
  }
}

void LLVMCodeGenImpl::visit(const Mul* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFMul(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateMul(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Mul", v);
  }
}

void LLVMCodeGenImpl::visit(const Div* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFDiv(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSDiv(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Div", v);
  }
}

void LLVMCodeGenImpl::visit(const And* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateAnd(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in And", v);
  }
}

void LLVMCodeGenImpl::visit(const Or* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateOr(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Or", v);
  }
}

void LLVMCodeGenImpl::visit(const Xor* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateXor(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Xor", v);
  }
}

void LLVMCodeGenImpl::visit(const Lshift* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateShl(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Lshift", v);
  }
}

void LLVMCodeGenImpl::visit(const Rshift* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    if (v->lhs()->dtype().is_signed()) {
      value_ = irb_.CreateAShr(lhs, rhs);
    } else {
      value_ = irb_.CreateLShr(lhs, rhs);
    }
  } else {
    throw malformed_input("llvm_codgen: bad type in Rshift", v);
  }
}

void LLVMCodeGenImpl::visit(const Mod* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateSRem(lhs, rhs);
  } else {
    throw malformed_input("llvm_codgen: bad type in Mod", v);
  }
}

void LLVMCodeGenImpl::visit(const Max* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;

  if (v->dtype().is_integral()) {
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSGT(lhs, rhs)
                                       : irb_.CreateICmpUGT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),
      lhs,
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OGT, lhs, rhs), lhs, rhs));
}

void LLVMCodeGenImpl::visit(const Min* v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;
  if (v->dtype().is_integral()) {
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSLT(lhs, rhs)
                                       : irb_.CreateICmpULT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),
      lhs,
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OLT, lhs, rhs), lhs, rhs));
}

void LLVMCodeGenImpl::visit(const CompareSelect* v) {
  auto genUnbiased = [this, v]() -> llvm::Value* {
    v->lhs()->accept(this);
    auto lhs = this->value_;
    v->rhs()->accept(this);
    auto rhs = this->value_;
    v->ret_val1()->accept(this);
    auto retval1 = this->value_;
    v->ret_val2()->accept(this);
    auto retval2 = this->value_;

    auto type_used = v->lhs()->dtype().scalar_type();

    llvm::Value* cmp_;
    CompareSelectOperation cmp_op_ = v->compare_select_op();

    if (is_integral(type_used)) {
      cmp_ = irb_.CreateICmp(
          llvm_comparison_predicate(cmp_op_, type_used), lhs, rhs);
    } else if (is_floating_point(type_used)) {
      cmp_ = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op_), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    return irb_.CreateSelect(cmp_, retval1, retval2);
  };

  auto genBiased = [this, v]() -> llvm::Value* {
    v->lhs()->accept(this);
    auto lhs = this->value_;
    v->rhs()->accept(this);
    auto rhs = this->value_;

    auto cmp_type = v->lhs()->dtype().scalar_type();
    auto cmp_op = v->compare_select_op();
    llvm::Value* cmp;

    if (is_integral(cmp_type)) {
      cmp = irb_.CreateICmp(
          llvm_comparison_predicate(cmp_op, cmp_type), lhs, rhs);
    } else if (is_floating_point(cmp_type)) {
      cmp = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    auto lanes = v->lhs()->dtype().lanes();
    if (lanes > 1) {
      auto maskType = llvm::Type::getIntNTy(getContext(), lanes);
      auto zero = llvm::ConstantInt::get(maskType, 0);
      auto mask = irb_.CreateBitOrPointerCast(cmp, maskType);
      cmp = irb_.CreateICmpNE(mask, zero);
    }

    auto then_block = llvm::BasicBlock::Create(getContext(), "then", fn_);
    auto else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
    auto end_block = llvm::BasicBlock::Create(getContext(), "block", fn_);
    constexpr int32_t total_weight = 100000;
    auto true_weight = v->bias() == kLikely ? total_weight : 0;
    auto false_weight = total_weight - true_weight;
    irb_.CreateCondBr(
        cmp,
        then_block,
        else_block,
        llvm::MDBuilder(getContext())
            .createBranchWeights(true_weight, false_weight));

    irb_.SetInsertPoint(then_block);
    v->ret_val1()->accept(this);
    llvm::Value* then_val = value_;
    then_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    irb_.SetInsertPoint(else_block);
    v->ret_val2()->accept(this);
    llvm::Value* else_val = value_;
    else_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    irb_.SetInsertPoint(end_block);
    llvm::PHINode* phi = irb_.CreatePHI(then_val->getType(), 2);
    phi->addIncoming(then_val, then_block);
    phi->addIncoming(else_val, else_block);
    return phi;
  };

  value_ = v->bias() == kUnbiased ? genUnbiased() : genBiased();
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, llvm::Value*>::type
getFromType(llvm::Type* type, T value) {
  return llvm::ConstantInt::get(type, value, std::is_signed<T>::value);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, llvm::Value*>::type
getFromType(llvm::Type* type, T value) {
  return llvm::ConstantFP::get(type, value);
}

#define IMM_VISIT_DECLARE(Type, Name)                  \
  void LLVMCodeGenImpl::visit(const Name##Imm* v) {    \
    value_ = getFromType<Type>(Name##Ty_, v->value()); \
  }
AT_FORALL_SCALAR_TYPES(IMM_VISIT_DECLARE);
#undef IMM_VISIT_DECLARE

void LLVMCodeGenImpl::visit(const HalfImm* v) {
  value_ = llvm::ConstantFP::get(HalfTy_, v->value());
}

void LLVMCodeGenImpl::visit(const BoolImm* v) {
  value_ = llvm::ConstantInt::get(BoolTy_, v->value());
}

llvm::Type* llvmTypeToVec(llvm::Type* type, int lanes) {
  if (lanes > 1) {
    return llvm::VectorType::get(type, ElementCount(lanes));
  } else {
    return type;
  }
}

void LLVMCodeGenImpl::visit(const Cast* v) {
  v->src_value()->accept(this);

  llvm::Type* dstType =
      llvmTypeToVec(dtypeToLLVM(v->dtype()), v->dtype().lanes());
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  if (srcType == dstType) {
    // do nothing.
    return;
  }

  bool destUnsigned = v->dtype().scalar_type() == ScalarType::Byte ||
      v->dtype().scalar_type() == ScalarType::Bool;

  // Scalar casts
  if (srcType->isFPOrFPVectorTy()) {
    if (dstType->isFPOrFPVectorTy()) {
      // as with eager, convert from Double -> Half by Converting to Float then
      // Half. TODO: __truncdfhf2
      if (v->dtype().scalar_type() == ScalarType::Half &&
          v->src_value()->dtype().scalar_type() == ScalarType::Double) {
        value_ = irb_.CreateFPCast(
            value_, llvmTypeToVec(FloatTy_, v->dtype().lanes()));
      }
      value_ = irb_.CreateFPCast(value_, dstType);
    } else if (dstType->isIntOrIntVectorTy()) {
      // Strictly casting from Float -> i8 doesnt give correct results
      // set one bit true if the input float is not 0
      if (v->dtype().scalar_type() == ScalarType::Bool) {
        llvm::Value* zero =
            toVec(llvm::ConstantFP::get(srcType, 0.), v->dtype().lanes());
        value_ = irb_.CreateFCmp(llvm::FCmpInst::FCMP_UNO, value_, zero);
        value_ = irb_.CreateICmpEQ(
            value_, llvm::ConstantInt::get(value_->getType(), 0));
        value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
        return;
      }

      if (destUnsigned) {
        value_ = irb_.CreateFPToUI(value_, dstType);
      } else {
        value_ = irb_.CreateFPToSI(value_, dstType);
      }
    } else {
      throw unimplemented_lowering(v);
    }
    return;
  }
  if (!srcType->isIntOrIntVectorTy()) {
    throw unimplemented_lowering(v);
  }
  if (dstType->isFPOrFPVectorTy()) {
    if (destUnsigned) {
      value_ = irb_.CreateUIToFP(value_, dstType);
    } else {
      value_ = irb_.CreateSIToFP(value_, dstType);
    }
  } else if (dstType->isIntOrIntVectorTy()) {
    // Ensure bool true value is exactly one, since we convert to int
    // from bool by zero extending the int8
    if (v->dtype().scalar_type() == ScalarType::Bool) {
      llvm::Value* zero =
          toVec(llvm::ConstantInt::get(srcType, 0), v->dtype().lanes());
      value_ = irb_.CreateICmpNE(value_, zero);
    }
    value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
  } else {
    throw unimplemented_lowering(v);
  }
}

void LLVMCodeGenImpl::visit(const BitCast* v) {
  v->src_value()->accept(this);

  llvm::Type* dstType = dtypeToLLVM(v->dtype());
  if (v->dtype().lanes() > 1) {
    dstType = llvm::VectorType::get(dstType, ElementCount(v->dtype().lanes()));
  }
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  if (srcType == dstType) {
    // do nothing.
    return;
  }

  TORCH_CHECK(llvm::CastInst::isBitCastable(
      srcType->getScalarType(), dstType->getScalarType()));
  value_ = irb_.CreateBitOrPointerCast(value_, dstType);
}

void LLVMCodeGenImpl::visit(const Var* v) {
  value_ = varToValue(v);
}

llvm::Value* LLVMCodeGenImpl::varToValue(const Var* v) {
  // It is possible for v to be in both varToVal_ and varToArgs.
  // In that case, varToVal_ takes precedence.
  if (varToVal_.count(v)) {
    return varToVal_.at(v);
  } else if (varToArg_.count(v)) {
    auto idx = varToArg_.at(v);
    auto arg = fn_->arg_begin() + idx;
    return arg;
  }
  return nullptr;
}

void LLVMCodeGenImpl::replaceVarMapping(
    const std::vector<const Var*>& vars,
    const std::vector<llvm::Value*>& vals) {
  TORCH_CHECK(vars.size() == vals.size());
  int i = 0;
  for (int i = 0; i < vars.size(); i++) {
    const Var* var = vars[i];
    llvm::Value* val = vals[i];
    if (val) {
      varToVal_[var] = val;
    } else {
      varToVal_.erase(var);
    }
  }
}

void LLVMCodeGenImpl::visit(const Ramp* v) {
  v->base()->accept(this);
  auto base = this->value_;
  v->stride()->accept(this);
  auto stride = this->value_;
  int lanes = v->lanes();

  if (llvm::ConstantInt* const_stride =
          llvm::dyn_cast<llvm::ConstantInt>(stride)) {
    std::vector<llvm::Constant*> vals = {
        llvm::ConstantInt::get(base->getType(), 0)};
    for (int i = 1; i < lanes; ++i) {
      vals.push_back(llvm::ConstantExpr::getAdd(vals.back(), const_stride));
    }

    llvm::Value* offsets = llvm::ConstantVector::get(vals);
    llvm::Value* splat = irb_.CreateVectorSplat(lanes, base);
    value_ = irb_.CreateAdd(splat, offsets);
    return;
  }

  llvm::Type* vecType = nullptr;
  auto element_count = ElementCount(lanes);
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                    \
  case ScalarType::Name:                                       \
    vecType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw std::runtime_error("invalid dtype in Ramp");
  }

  value_ = llvm::UndefValue::get(vecType);
  for (int i = 0; i < lanes; ++i) {
    value_ = irb_.CreateInsertElement(value_, base, i);
    base = irb_.CreateAdd(base, stride);
  }
}

llvm::Value* LLVMCodeGenImpl::emitUnmaskedLoad(
    llvm::Value* base,
    llvm::Value* idx) {
  auto addr = irb_.CreateGEP(base, idx);
  return irb_.CreateLoad(addr);
}

llvm::Value* LLVMCodeGenImpl::emitMaskedLoad(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask) {
  // Create block structure for the masked load.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(IntTy_, 1));
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

void LLVMCodeGenImpl::visit(const Load* v) {
  if (v->dtype().lanes() == 1) {
    v->base_handle()->accept(this);
    auto base = this->value_;
    v->flat_index()->accept(this);
    auto idx = this->value_;

    auto* maskimm = dynamic_cast<const IntImm*>(v->mask());
    if (maskimm && maskimm->value() == 1) {
      value_ = emitUnmaskedLoad(base, idx);
    } else {
      v->mask()->accept(this);
      auto mask = this->value_;
      value_ = emitMaskedLoad(base, idx, mask);
    }
    return;
  }

  llvm::Type* loadType = nullptr;

  auto element_count = ElementCount(v->dtype().lanes());
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                     \
  case ScalarType::Name:                                        \
    loadType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw std::runtime_error("invalid dtype in Load");
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
  auto* idx_ramp = dynamic_cast<const Ramp*>(v->flat_index());
  if (unmasked_load && idx_ramp) {
    auto* stride_imm = dynamic_cast<const IntImm*>(idx_ramp->stride());
    if (stride_imm && stride_imm->value() == 1) {
      v->base_handle()->accept(this);
      auto base = this->value_;
      idx_ramp->base()->accept(this);
      auto first_idx = this->value_;

      auto addr = irb_.CreateGEP(base, first_idx);
      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(loadType, 0));
#if LLVM_VERSION_MAJOR >= 13
      value_ = irb_.CreateAlignedLoad(vaddr, llvm::MaybeAlign(4));
#else
      value_ = irb_.CreateAlignedLoad(vaddr, 4);
#endif
      return;
    }
  }

  // Fallback to a scalar implementation
  v->base_handle()->accept(this);
  auto base = this->value_;
  v->flat_index()->accept(this);
  auto idx = this->value_;
  v->mask()->accept(this);
  auto mask = this->value_;

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

// Pack the arguments into an aggregate struct for forwarding.
llvm::Value* LLVMCodeGenImpl::packFuncArgs(
    const std::vector<llvm::Value*>& func_args) {
  if (func_args.empty()) {
    llvm::PointerType* VoidPtrType = llvm::Type::getInt8PtrTy(getContext());
    llvm::Constant* NullPtr = llvm::ConstantPointerNull::get(VoidPtrType);
    return NullPtr;
  }
  std::vector<llvm::Type*> arg_types(func_args.size());
  for (int i = 0; i < func_args.size(); i++) {
    arg_types[i] = func_args[i]->getType();
  }
  llvm::StructType* packed_type = llvm::StructType::create(arg_types);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
  llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
  for (int i = 0; i < func_args.size(); i++) {
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    irb_.CreateStore(func_args[i], dst_ptr);
  }
  return packed;
}

// Unpack the aggregate struct into individual arguments.
std::vector<llvm::Value*> LLVMCodeGenImpl::unpackFuncArgs(
    llvm::Value* packed,
    int arg_count) {
  // TODO: extract arg_count from packed.
  std::vector<llvm::Value*> func_args(arg_count);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  for (int i = 0; i < arg_count; i++) {
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    func_args[i] = irb_.CreateLoad(dst_ptr);
  }
  return func_args;
}

// Lower the parallel for-loop.
// * Move the body into its own closure.
// * Identify var across the boundary into arguments and forward them.
// * Send the closure and range to the dispatcher for execution.
void LLVMCodeGenImpl::processParallelFor(const For* v) {
  // Create "start" and "stop" values.
  v->start()->accept(this);
  auto start = this->value_;
  v->stop()->accept(this);
  auto stop = this->value_;

  // The Vars that need to be forward in the body closure.
  std::vector<const Var*> body_arg_vars;
  // Corresponding Value* that was used in the old body for the caller.
  std::vector<llvm::Value*> body_caller_vals;
  // Corresponding Value* that will be used in the new body closure.
  std::vector<llvm::Value*> body_closure_args;

  // Identify the Var* used in the body, and generated outside.
  VarFinder var_finder;
  v->body()->accept(&var_finder);
  const auto& vars = var_finder.vars();
  for (auto& var : vars) {
    if (llvm::Value* value = varToValue(var)) {
      body_arg_vars.push_back(var);
      body_caller_vals.push_back(value);
    }
  }

  // Pack the arguments in an automatic variable for forwarding.
  llvm::Value* packed_caller_args = packFuncArgs(body_caller_vals);

  // Remember where we are before moving to the new function.
  llvm::BasicBlock* old_insert_block = irb_.GetInsertBlock();

  // Create the new body closure code.
  auto func_type =
      llvm::FunctionType::get(VoidTy_, {IntTy_, Int8PtrTy_}, false);
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::PrivateLinkage, "func", module_.get());
  auto func_body = llvm::BasicBlock::Create(getContext(), "func_body", func);
  irb_.SetInsertPoint(func_body);
  auto args = func->arg_begin();
  llvm::Value* index = args++;
  llvm::Value* packed_func_args_raw = args++;
  llvm::Value* packed_func_args = irb_.CreatePointerCast(
      packed_func_args_raw, packed_caller_args->getType());

  // Unpack the arguments from the opaque buffer.
  body_closure_args = unpackFuncArgs(packed_func_args, body_arg_vars.size());
  // Set the codegen to the new func.
  // TODO: this should be replaced by RAII wrappers.
  varToVal_[v->var()] = index;
  replaceVarMapping(body_arg_vars, body_closure_args);
  llvm::Function* old_fn = fn_;
  fn_ = func;
  if (v->body()) {
    v->body()->accept(this);
  }
  // Restore back to the previous fn_
  fn_ = old_fn;
  irb_.CreateRet(nullptr);
  replaceVarMapping(body_arg_vars, body_caller_vals);
  varToVal_.erase(v->var());

  // Points back to the original block and generate the callee code.
  irb_.SetInsertPoint(old_insert_block);
  llvm::Value* packed_caller_args_ptr =
      irb_.CreatePointerCast(packed_caller_args, Int8PtrTy_);
  llvm::Value* func_value = irb_.CreatePointerCast(func, Int8PtrTy_);
  llvm::FunctionType* dispatcher_fntype = llvm::FunctionType::get(
      VoidTy_, {Int8PtrTy_, IntTy_, IntTy_, Int8PtrTy_}, false);
  FunctionCallee dispatcher_callee =
      module_->getOrInsertFunction("DispatchParallel", dispatcher_fntype);
  llvm::Function* dispatcher =
      llvm::cast<llvm::Function>(dispatcher_callee.getCallee());
  irb_.CreateCall(
      dispatcher, {func_value, start, stop, packed_caller_args_ptr});
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

void LLVMCodeGenImpl::visit(const For* v) {
  if (v->is_parallel()) {
    processParallelFor(v);
    return;
  }

  // Create "start" and "stop" values.
  v->start()->accept(this);
  auto start = this->value_;
  v->stop()->accept(this);
  auto stop = this->value_;

  // Create block for loop condition test.
  auto preheader = irb_.GetInsertBlock();
  auto condBlock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  irb_.CreateBr(condBlock);
  irb_.SetInsertPoint(condBlock);

  // Set up phi node for index variable.
  auto idx = irb_.CreatePHI(IntTy_, 2);
  idx->addIncoming(start, preheader);
  if (!varToVal_.count(v->var())) {
    varToVal_.emplace(v->var(), idx);
  } else {
    throw std::runtime_error("var should not exist before");
  }

  // Create the body and exit blocks.
  auto body = llvm::BasicBlock::Create(getContext(), "body", fn_);
  auto exit = llvm::BasicBlock::Create(getContext(), "exit", fn_);

  // Create the stop condition.
  auto cond = irb_.CreateICmpSLT(idx, stop);
  irb_.CreateCondBr(cond, body, exit);

  // Codegen the body.
  irb_.SetInsertPoint(body);
  if (v->body()) {
    v->body()->accept(this);
  }
  // "Body" block may have changed if we generated nested control flow.
  body = irb_.GetInsertBlock();

  // Increment the index variable and branch back to loop test.
  auto inc = irb_.CreateAdd(idx, llvm::ConstantInt::getSigned(IntTy_, 1));
  irb_.CreateBr(condBlock);
  idx->addIncoming(inc, body);

  // Exit the loop.
  irb_.SetInsertPoint(exit);

  varToVal_.erase(v->var());
  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

void LLVMCodeGenImpl::visit(const Block* v) {
  const Block* last = scope_;
  scope_ = v;

  for (Stmt* s : *v) {
    s->accept(this);
  }

  scope_ = last;

  auto it = scopeToVar_.find(v);
  if (it != scopeToVar_.end()) {
    for (const Var* e : it->second) {
      if (varToVal_.erase(e) != 1) {
        throw std::runtime_error("erasing var that doesn't exist");
      }
    }
  }
}

void LLVMCodeGenImpl::emitUnmaskedStore(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* val) {
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
}

void LLVMCodeGenImpl::emitMaskedStore(
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask,
    llvm::Value* val) {
  // Create block structure for the masked store.
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(IntTy_, 1));
  irb_.CreateCondBr(cond, condblock, tailblock);

  // Do the store
  irb_.SetInsertPoint(condblock);
  auto addr = irb_.CreateGEP(base, idx);
  irb_.CreateStore(val, addr);
  irb_.CreateBr(tailblock);

  // Merge the masked and unmasked CFG edges
  irb_.SetInsertPoint(tailblock);
}

void LLVMCodeGenImpl::visit(const Store* v) {
  if (v->value()->dtype().lanes() == 1) {
    v->base_handle()->accept(this);
    auto base = this->value_;
    v->flat_index()->accept(this);
    auto idx = this->value_;
    v->value()->accept(this);
    auto val = this->value_;

    auto* maskimm = dynamic_cast<const IntImm*>(v->mask());
    if (maskimm && maskimm->value() == 1) {
      emitUnmaskedStore(base, idx, val);
    } else {
      v->mask()->accept(this);
      auto mask = this->value_;

      emitMaskedStore(base, idx, mask, val);
    }

    value_ = llvm::ConstantInt::get(IntTy_, 0);
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

  v->base_handle()->accept(this);
  auto base = this->value_;
  v->value()->accept(this);
  auto val = this->value_;

  // Handle the case where the store is contiguous and unmasked efficiently
  auto* idx_ramp = dynamic_cast<const Ramp*>(v->flat_index());
  if (unmasked_store && idx_ramp) {
    auto* stride_imm = dynamic_cast<const IntImm*>(idx_ramp->stride());
    if (stride_imm && stride_imm->value() == 1) {
      idx_ramp->base()->accept(this);
      auto first_idx = value_;

      auto addr = irb_.CreateGEP(base, first_idx);
      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(val->getType(), 0));

#if LLVM_VERSION_MAJOR >= 13
      irb_.CreateAlignedStore(val, vaddr, llvm::MaybeAlign(4));
#else
      irb_.CreateAlignedStore(val, vaddr, 4);
#endif
      value_ = llvm::ConstantInt::get(IntTy_, 0);
      return;
    }
  }

  v->flat_index()->accept(this);
  auto idx = this->value_;
  v->mask()->accept(this);
  auto mask = this->value_;

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

  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

void LLVMCodeGenImpl::visit(const Broadcast* v) {
  v->value()->accept(this);
  int lanes = v->lanes();
  value_ = irb_.CreateVectorSplat(lanes, value_);
}

void LLVMCodeGenImpl::visit(const IfThenElse* v) {
  v->condition()->accept(this);
  llvm::Value* condition = value_;
  llvm::Value* c = irb_.CreateICmpNE(
      condition, llvm::ConstantInt::get(condition->getType(), 0));

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

static void applyMathFunctionAttributes(llvm::Function* f) {
  f->addFnAttr(llvm::Attribute::ReadNone);
  f->addFnAttr(llvm::Attribute::NoUnwind);
  // TODO: Adding this attr should be correct, but as of LLVM 9.0.1 adding it
  // causes some math functions to incorrectly be turned into tail calls.
  // f->addFnAttr(llvm::Attribute::Speculatable);
#if LLVM_VERSION_MAJOR >= 9
  f->addFnAttr(llvm::Attribute::NoFree);
  f->addFnAttr(llvm::Attribute::WillReturn);
#endif
}

llvm::Value* LLVMCodeGenImpl::toVec(llvm::Value* v, int lanes) {
  if (lanes > 1) {
    return irb_.CreateVectorSplat(lanes, v);
  } else {
    return v;
  }
}

void LLVMCodeGenImpl::emitIsNan(const Intrinsics* v) {
  v->param(0)->accept(this);
  llvm::Type* dstType = dtypeToLLVM(v->dtype());
  if (!v->param(0)->dtype().is_floating_point()) {
    value_ = toVec(llvm::ConstantInt::get(dstType, 0), v->dtype().lanes());
  } else {
    TORCH_INTERNAL_ASSERT(v->dtype().scalar_type() == ScalarType::Int);
    auto is_nan = irb_.CreateFCmpUNO(
        value_, llvm::ConstantFP::get(value_->getType(), 0.));
    if (v->dtype().lanes() > 1) {
      dstType =
          llvm::VectorType::get(dstType, ElementCount(v->dtype().lanes()));
    }
    value_ = irb_.CreateIntCast(is_nan, dstType, /*isSigned*/ false);
  }
}

static bool wantSleef(const std::string& name) {
  // Using sleef on these ops is slower than libm.
  static std::unordered_set<std::string> noSleef = {
      "sqrt",
      "ceil",
      "trunc",
      "fabs",
      "floor",
      "sqrtf",
      "ceilf",
      "truncf",
      "fabsf",
      "floorf",
  };
  return noSleef.find(name) == noSleef.end();
}

LLVMCodeGenImpl::SimdCallee LLVMCodeGenImpl::getSimdFunction(
    const std::string& basename,
    llvm::Type* basetype,
    Arity arity,
    int lanes) {
  std::string name;
  llvm::Type* type;
  bool useSimd;

  // Determine whether to use vectorized intrinsic.
  auto const& featureString = jit_->getTargetMachine().getTargetFeatureString();
  bool hasAVX = featureString.find("+avx") != llvm::StringRef::npos;
  std::string typeSuffix = basetype == DoubleTy_ ? "d" : "";
  std::string sleefName =
      "Sleef_" + basename + typeSuffix + std::to_string(lanes);
  if (wantSleef(basename) && hasAVX && jit_->hasSymbol(sleefName)) {
    name = std::move(sleefName);
    type = llvm::VectorType::get(basetype, ElementCount(lanes));
    useSimd = true;
  } else {
    name = basename;
    type = basetype;
    useSimd = false;
  }

  // Get function to call from name and type.
  llvm::FunctionType* fntype;
  switch (arity) {
    case Unary:
      fntype = llvm::FunctionType::get(type, {type}, false);
      break;
    case Binary:
      fntype = llvm::FunctionType::get(type, {type, type}, false);
      break;
  }
  FunctionCallee callee = module_->getOrInsertFunction(name, fntype, {});
  applyMathFunctionAttributes(llvm::cast<llvm::Function>(callee.getCallee()));
  return SimdCallee{callee.getFunctionType(), callee.getCallee(), useSimd};
}

void LLVMCodeGenImpl::visit(const Intrinsics* v) {
  llvm::FunctionType* call_ty = nullptr;
  llvm::Value* call_fn = nullptr;
  bool call_simd_sleef = false;

  if (v->op_type() == kIsNan) {
    return emitIsNan(v);
  }

  if (v->dtype().scalar_type() == ScalarType::Float) {
    switch (v->op_type()) {
      case kRsqrt: {
        v->params().front()->accept(this);
        value_ = irb_.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, value_);
        llvm::Value* constant =
            toVec(llvm::ConstantFP::get(FloatTy_, 1.0), v->dtype().lanes());
        value_ = irb_.CreateFDiv(constant, value_);
        return;
      } break;

#define SIMD_UNARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                  \
    std::tie(call_ty, call_fn, call_simd_sleef) =               \
        getSimdFunction(name, type, Unary, v->dtype().lanes()); \
  } break;
        SIMD_UNARY_MATH_CASE(kLog10, "log10f", FloatTy_)
        SIMD_UNARY_MATH_CASE(kLog, "logf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kLog1p, "log1pf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kLog2, "log2f", FloatTy_)
        SIMD_UNARY_MATH_CASE(kExp, "expf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kCos, "cosf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kSin, "sinf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kSqrt, "sqrtf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kAbs, "fabsf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kFloor, "floorf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kCeil, "ceilf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kTrunc, "truncf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kRound, "roundf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kErf, "erff", FloatTy_)
        SIMD_UNARY_MATH_CASE(kErfc, "erfcf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kTan, "tanf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kAcos, "acosf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kAsin, "asinf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kAtan, "atanf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kCosh, "coshf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kSinh, "sinhf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kTanh, "tanhf", FloatTy_)
        SIMD_UNARY_MATH_CASE(kExpm1, "expm1f", FloatTy_)
        SIMD_UNARY_MATH_CASE(kLgamma, "lgammaf", FloatTy_)
#undef SIMD_UNARY_MATH_CASE

#define SIMD_BINARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                   \
    std::tie(call_ty, call_fn, call_simd_sleef) =                \
        getSimdFunction(name, type, Binary, v->dtype().lanes()); \
  } break;
        SIMD_BINARY_MATH_CASE(kAtan2, "atan2f", FloatTy_)
        SIMD_BINARY_MATH_CASE(kPow, "powf", FloatTy_)
        SIMD_BINARY_MATH_CASE(kFmod, "fmodf", FloatTy_)
#undef SIMD_BINARY_MATH_CASE

      case kRemainder: {
        FunctionCallee callee = module_->getOrInsertFunction(
            "remainderf",
            llvm::FunctionType::get(FloatTy_, {FloatTy_, FloatTy_}, false),
            {});
        call_ty = callee.getFunctionType();
        call_fn = callee.getCallee();
        applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn));
      } break;

      default: {
        throw unimplemented_lowering(v);
      } break;
    }

  } else if (v->dtype().scalar_type() == ScalarType::Double) {
    switch (v->op_type()) {
#define SIMD_UNARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                  \
    std::tie(call_ty, call_fn, call_simd_sleef) =               \
        getSimdFunction(name, type, Unary, v->dtype().lanes()); \
  } break;
      SIMD_UNARY_MATH_CASE(kLog10, "log10", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kLog, "log", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kLog1p, "log1p", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kLog2, "log2", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kExp, "exp", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kCos, "cos", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kSin, "sin", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kSqrt, "sqrt", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kAbs, "fabs", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kFloor, "floor", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kCeil, "ceil", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kTrunc, "trunc", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kRound, "round", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kErf, "erf", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kErfc, "erfc", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kTan, "tan", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kAcos, "acos", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kAsin, "asin", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kAtan, "atan", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kCosh, "cosh", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kSinh, "sinh", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kTanh, "tanh", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kExpm1, "expm1", DoubleTy_)
      SIMD_UNARY_MATH_CASE(kLgamma, "lgamma", DoubleTy_)
#undef SIMD_UNARY_MATH_CASE

      case kRsqrt: {
        v->params().front()->accept(this);
        value_ = irb_.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, value_);
        llvm::Value* constant = llvm::ConstantFP::get(DoubleTy_, 1.0);
        if (v->dtype().lanes() > 1) {
          constant = irb_.CreateVectorSplat(v->dtype().lanes(), constant);
        }
        value_ = irb_.CreateFDiv(constant, value_);
        return;
      } break;

#define SIMD_BINARY_MATH_CASE(enum, name, type)                  \
  case enum: {                                                   \
    std::tie(call_ty, call_fn, call_simd_sleef) =                \
        getSimdFunction(name, type, Binary, v->dtype().lanes()); \
  } break;
        SIMD_BINARY_MATH_CASE(kAtan2, "atan2", DoubleTy_)
        SIMD_BINARY_MATH_CASE(kPow, "pow", DoubleTy_)
        SIMD_BINARY_MATH_CASE(kFmod, "fmod", DoubleTy_)
#undef SIMD_BINARY_MATH_CASE

      case kRemainder: {
        FunctionCallee callee = module_->getOrInsertFunction(
            "remainder",
            llvm::FunctionType::get(DoubleTy_, {DoubleTy_, DoubleTy_}, false),
            {});
        call_ty = callee.getFunctionType();
        call_fn = callee.getCallee();
        applyMathFunctionAttributes(llvm::cast<llvm::Function>(call_fn));
      } break;

      default: {
        throw unimplemented_lowering(v);
      } break;
    }
  } else if (v->dtype().is_integral() && v->op_type() == kAbs) {
    // abs is only intrinsic defined for integer inputs in pytorch eager
    v->params().front()->accept(this);
    if (!v->dtype().is_signed()) {
      return;
    }
    // TODO: use llvm.abs intrinsic for LLVM 12
    auto zero = llvm::ConstantInt::get(value_->getType(), 0);
    auto neg_value = irb_.CreateSub(zero, value_);
    auto icmp = irb_.CreateICmpSGT(value_, zero);
    value_ = irb_.CreateSelect(icmp, value_, neg_value);
    return;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        v,
        "Unimplemented lowering:",
        v->op_type(),
        " for input of dtype",
        v->dtype().scalar_dtype());
  }

  std::vector<llvm::Value*> params;
  for (auto& p : v->params()) {
    p->accept(this);
    params.push_back(value_);
  }

  if (v->dtype().lanes() == 1 || call_simd_sleef == true) {
    value_ = irb_.CreateCall(call_ty, call_fn, params);
  } else {
    llvm::Type* vecType = params[0]->getType();
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

void LLVMCodeGenImpl::visit(const ExternalCall* v) {
  constexpr int max_buffers = 10;
  constexpr int max_dimensions = 40;

  auto& func_registry = getNNCFunctionRegistry();
  if (!func_registry.count(v->func_name())) {
    throw unimplemented_lowering(v);
  }

  // Prepare a vector of bufs that we need to pass to the external function.
  // This vector is the output buf followed by the buf_args.
  std::vector<const Buf*> bufs(v->buf_args());
  bufs.insert(bufs.begin(), v->buf());

  int64_t bufs_num = bufs.size();
  int64_t args_num = v->args().size();

  // Count the size of dims array - it consists of dimension of all bufs
  // concatenated together.
  int64_t dims_num = 0;
  for (const Buf* b : bufs) {
    dims_num += b->dims().size();
  }

  llvm::Value* buf_ptrs = irb_.CreateAlloca(
      Int8PtrTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
  llvm::Value* buf_ranks = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
  llvm::Value* buf_dims = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, dims_num));
  llvm::Value* buf_dtypes = irb_.CreateAlloca(
      ByteTy_, llvm::ConstantInt::getSigned(IntTy_, bufs_num));
  llvm::Value* extra_args = irb_.CreateAlloca(
      LongTy_, llvm::ConstantInt::getSigned(IntTy_, args_num));

  int i = 0;
  int dim_idx = 0;
  for (const Buf* b : bufs) {
    // Store value for buf pointer
    auto gep = irb_.CreateInBoundsGEP(
        buf_ptrs, {llvm::ConstantInt::getSigned(IntTy_, i)});
    b->base_handle()->accept(this);
    auto buf_ptr = this->value_;
    auto buf_void_ptr = irb_.CreatePointerCast(buf_ptr, Int8PtrTy_);
    irb_.CreateStore(buf_void_ptr, gep);

    // Store dtype of the buf
    gep = irb_.CreateInBoundsGEP(
        buf_dtypes, {llvm::ConstantInt::getSigned(IntTy_, i)});
    irb_.CreateStore(
        llvm::ConstantInt::getSigned(ByteTy_, (int8_t)b->dtype().scalar_type()),
        gep);

    // Store rank of the buf
    gep = irb_.CreateInBoundsGEP(
        buf_ranks, {llvm::ConstantInt::getSigned(IntTy_, i)});
    irb_.CreateStore(
        llvm::ConstantInt::getSigned(LongTy_, b->dims().size()), gep);

    // Store dims of the buf
    for (int dim = 0; dim < b->dims().size(); dim++) {
      gep = irb_.CreateInBoundsGEP(
          buf_dims, {llvm::ConstantInt::getSigned(IntTy_, dim_idx)});
      b->dims()[dim]->accept(this);
      auto dim_val = this->value_;
      irb_.CreateStore(irb_.CreateZExt(dim_val, LongTy_), gep);
      dim_idx++;
    }

    i++;
  }

  i = 0;
  for (const Expr* arg : v->args()) {
    auto gep = irb_.CreateInBoundsGEP(
        extra_args, {llvm::ConstantInt::getSigned(IntTy_, i)});
    arg->accept(this);
    irb_.CreateStore(irb_.CreateZExtOrBitCast(this->value_, LongTy_), gep);
    i++;
  }

  // Generate the call itself
  std::string fname = v->func_name();
  FunctionCallee callee = module_->getOrInsertFunction(
      fname,
      llvm::FunctionType::get(
          llvm::Type::getVoidTy(getContext()), // return type
          {LongTy_, // int64_t bufs_num
           Int8PtrTy_->getPointerTo(), // void** buf_data
           LongTy_->getPointerTo(), // int64_t* buf_ranks
           LongTy_->getPointerTo(), // int64_t* buf_dims
           ByteTy_->getPointerTo(), // int64_t* buf_dtypes
           LongTy_, // int64_t args_num
           LongTy_->getPointerTo()}, // int64_t* extra_args
          false)); // is var_arg

  auto call_ty = callee.getFunctionType();
  auto call_fn = callee.getCallee();
  llvm::cast<llvm::Function>(call_fn)->addFnAttr(llvm::Attribute::NoUnwind);

  irb_.CreateCall(
      call_ty,
      call_fn,
      {llvm::ConstantInt::getSigned(LongTy_, bufs_num),
       buf_ptrs,
       buf_ranks,
       buf_dims,
       buf_dtypes,
       llvm::ConstantInt::getSigned(LongTy_, args_num),
       extra_args});

  value_ = llvm::ConstantInt::get(IntTy_, 0);
}

void LLVMCodeGenImpl::visit(const Allocate* v) {
  llvm::Value* size =
      llvm::ConstantInt::getSigned(LongTy_, v->dtype().byte_size());
  for (const Expr* e : v->dims()) {
    e->accept(this);
    size = irb_.CreateMul(size, irb_.CreateZExt(value_, LongTy_));
  }

  value_ = llvm::ConstantInt::get(IntTy_, 0);

  if (llvm::ConstantInt* CI = llvm::dyn_cast<llvm::ConstantInt>(size)) {
    if (CI->getSExtValue() < 512) {
      llvm::Value* alloca = irb_.CreateAlloca(dtypeToLLVM(v->dtype()), size);
      varToVal_[v->buffer_var()] = alloca;
      return;
    }
  }

  llvm::Instruction* I = llvm::CallInst::CreateMalloc(
      irb_.GetInsertBlock(),
      LongTy_,
      dtypeToLLVM(v->dtype()),
      size,
      nullptr,
      nullptr);

  // Insert the bitcast into the block.
  irb_.SetInsertPoint(irb_.GetInsertBlock());
  llvm::Value* malloc = irb_.Insert(I);
  varToVal_[v->buffer_var()] = malloc;
}

void LLVMCodeGenImpl::visit(const Free* v) {
  value_ = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* ptr = varToVal_.at(v->buffer_var());
  if (!llvm::isa<llvm::AllocaInst>(ptr)) {
    irb_.Insert(llvm::CallInst::CreateFree(ptr, irb_.GetInsertBlock()));
  }
}

void LLVMCodeGenImpl::visit(const Let* v) {
  v->value()->accept(this);
  if (!varToVal_.count(v->var())) {
    varToVal_.emplace(v->var(), value_);
    scopeToVar_[scope_].push_back(v->var());
  } else {
    throw std::runtime_error("var should not exist before");
  }
}

void LLVMCodeGenImpl::visit(const Cond* v) {
  // Even if true_stmt and false_stmt are nullptr,
  // in case condition is a function call with side effect,
  // we still evaluate it.
  v->condition()->accept(this);

  if (!v->true_stmt() && !v->false_stmt()) {
    return;
  }
  assert(v->true_stmt());

  llvm::Value* condition = value_;
  llvm::Value* c = irb_.CreateICmpNE(
      condition, llvm::ConstantInt::get(condition->getType(), 0));
  llvm::BasicBlock* then_block =
      llvm::BasicBlock::Create(getContext(), "then", fn_);
  llvm::BasicBlock* else_block = nullptr;
  if (v->false_stmt()) {
    else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
  }
  llvm::BasicBlock* end_block =
      llvm::BasicBlock::Create(getContext(), "end", fn_);

  if (else_block) {
    irb_.CreateCondBr(c, then_block, else_block);
  } else {
    irb_.CreateCondBr(c, then_block, end_block);
  }

  irb_.SetInsertPoint(then_block);
  v->true_stmt()->accept(this);
  irb_.CreateBr(end_block);

  if (else_block) {
    irb_.SetInsertPoint(else_block);
    v->false_stmt()->accept(this);
    irb_.CreateBr(end_block);
  }

  irb_.SetInsertPoint(end_block);
}

void LLVMCodeGenImpl::optimize(llvm::Module& M) {
  llvm::legacy::FunctionPassManager FPM(&M);
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  auto& TM = jit_->getTargetMachine();
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
  PM.add(llvm::createDeadCodeEliminationPass());
  PM.add(llvm::createAlwaysInlinerLegacyPass());
  PM.run(M);
  for (auto& FF : M) {
    FPM.run(FF);
  }
  FPM.doFinalization();
  PM.run(M);
}

RegisterCodeGen<LLVMCodeGen> llvm_codegen_reg("llvm_codegen");

#endif // TORCH_ENABLE_LLVM
