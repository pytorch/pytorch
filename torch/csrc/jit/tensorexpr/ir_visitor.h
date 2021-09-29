#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRVisitor {
 public:
  virtual ~IRVisitor() = default;
  virtual void visit(AddPtr v);
  virtual void visit(SubPtr v);
  virtual void visit(MulPtr v);
  virtual void visit(DivPtr v);
  virtual void visit(ModPtr v);
  virtual void visit(MaxPtr v);
  virtual void visit(MinPtr v);
  virtual void visit(AndPtr v);
  virtual void visit(OrPtr v);
  virtual void visit(XorPtr v);
  virtual void visit(LshiftPtr v);
  virtual void visit(RshiftPtr v);
  virtual void visit(CompareSelectPtr v);

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(Name##ImmPtr v);

  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

  virtual void visit(CastPtr v);
  virtual void visit(BitCastPtr v);
  virtual void visit(VarPtr v);
  virtual void visit(BufPtr v);
  virtual void visit(RampPtr v);
  virtual void visit(LoadPtr v);
  virtual void visit(ForPtr v);
  virtual void visit(BlockPtr v);
  virtual void visit(StorePtr v);
  virtual void visit(BroadcastPtr v);
  virtual void visit(IfThenElsePtr v);
  virtual void visit(IntrinsicsPtr v);
  virtual void visit(AllocatePtr v);
  virtual void visit(FreePtr v);
  virtual void visit(LetPtr v);
  virtual void visit(CondPtr v);
  virtual void visit(TermPtr v);
  virtual void visit(PolynomialPtr v);
  virtual void visit(RoundOffPtr v);
  virtual void visit(MaxTermPtr v);
  virtual void visit(MinTermPtr v);
  virtual void visit(ReduceOpPtr v);
  virtual void visit(AtomicAddPtr v);
  virtual void visit(SyncThreadsPtr v);
  virtual void visit(ExternalCallPtr v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
