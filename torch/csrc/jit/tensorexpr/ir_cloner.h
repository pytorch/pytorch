#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRCloner : public IRMutator {
 public:
  ~IRCloner() override = default;
  ExprPtr mutate(AddPtr v) override;
  ExprPtr mutate(SubPtr v) override;
  ExprPtr mutate(MulPtr v) override;
  ExprPtr mutate(DivPtr v) override;
  ExprPtr mutate(ModPtr v) override;
  ExprPtr mutate(MaxPtr v) override;
  ExprPtr mutate(MinPtr v) override;
  ExprPtr mutate(AndPtr v) override;
  ExprPtr mutate(OrPtr v) override;
  ExprPtr mutate(XorPtr v) override;
  ExprPtr mutate(LshiftPtr v) override;
  ExprPtr mutate(RshiftPtr v) override;
  ExprPtr mutate(CompareSelectPtr v) override;
#define IMM_MUTATE_DECLARE(Type, Name) ExprPtr mutate(Name##ImmPtr v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  ExprPtr mutate(CastPtr v) override;
  ExprPtr mutate(BitCastPtr v) override;
  ExprPtr mutate(VarPtr v) override;
  ExprPtr mutate(BufPtr v) override;
  ExprPtr mutate(RampPtr v) override;
  ExprPtr mutate(LoadPtr v) override;
  ExprPtr mutate(BroadcastPtr v) override;
  ExprPtr mutate(IfThenElsePtr v) override;
  ExprPtr mutate(IntrinsicsPtr v) override;

  ExprPtr mutate(TermPtr v) override;
  ExprPtr mutate(PolynomialPtr v) override;
  ExprPtr mutate(RoundOffPtr v) override;
  ExprPtr mutate(MaxTermPtr v) override;
  ExprPtr mutate(MinTermPtr v) override;

  ExprPtr mutate(ReduceOpPtr v) override;

  StmtPtr mutate(ForPtr v) override;
  StmtPtr mutate(BlockPtr v) override;
  StmtPtr mutate(StorePtr v) override;
  StmtPtr mutate(AtomicAddPtr v) override;
  StmtPtr mutate(SyncThreadsPtr v) override;
  StmtPtr mutate(ExternalCallPtr v) override;

  StmtPtr mutate(AllocatePtr v) override;
  StmtPtr mutate(FreePtr v) override;
  StmtPtr mutate(LetPtr v) override;
  StmtPtr mutate(CondPtr v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
