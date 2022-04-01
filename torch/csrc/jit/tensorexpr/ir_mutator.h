#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() = default;
  virtual ExprPtr mutate(AddPtr v);
  virtual ExprPtr mutate(SubPtr v);
  virtual ExprPtr mutate(MulPtr v);
  virtual ExprPtr mutate(DivPtr v);
  virtual ExprPtr mutate(ModPtr v);
  virtual ExprPtr mutate(MaxPtr v);
  virtual ExprPtr mutate(MinPtr v);
  virtual ExprPtr mutate(AndPtr v);
  virtual ExprPtr mutate(OrPtr v);
  virtual ExprPtr mutate(XorPtr v);
  virtual ExprPtr mutate(LshiftPtr v);
  virtual ExprPtr mutate(RshiftPtr v);
  virtual ExprPtr mutate(CompareSelectPtr v);
#define IMM_MUTATE_DECLARE(Type, Name) virtual ExprPtr mutate(Name##ImmPtr v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  virtual ExprPtr mutate(CastPtr v);
  virtual ExprPtr mutate(BitCastPtr v);
  virtual ExprPtr mutate(VarPtr v);
  virtual ExprPtr mutate(BufPtr v);
  virtual ExprPtr mutate(RampPtr v);
  virtual ExprPtr mutate(LoadPtr v);
  virtual ExprPtr mutate(BroadcastPtr v);
  virtual ExprPtr mutate(IfThenElsePtr v);
  virtual ExprPtr mutate(IntrinsicsPtr v);

  virtual ExprPtr mutate(TermPtr v);
  virtual ExprPtr mutate(PolynomialPtr v);
  virtual ExprPtr mutate(RoundOffPtr v);
  virtual ExprPtr mutate(MaxTermPtr v);
  virtual ExprPtr mutate(MinTermPtr v);

  virtual ExprPtr mutate(ReduceOpPtr v);

  virtual StmtPtr mutate(ForPtr v);
  virtual StmtPtr mutate(BlockPtr v);
  virtual StmtPtr mutate(StorePtr v);
  virtual StmtPtr mutate(AtomicAddPtr v);
  virtual StmtPtr mutate(SyncThreadsPtr v);
  virtual StmtPtr mutate(ExternalCallPtr v);

  virtual StmtPtr mutate(AllocatePtr v);
  virtual StmtPtr mutate(FreePtr v);
  virtual StmtPtr mutate(PlacementAllocatePtr v);
  virtual StmtPtr mutate(LetPtr v);
  virtual StmtPtr mutate(CondPtr v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
