#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

namespace torch::jit::tensorexpr {

class TORCH_API IRCloner : public IRMutator {
 public:
  ~IRCloner() override = default;
  ExprPtr mutate(const AddPtr& v) override;
  ExprPtr mutate(const SubPtr& v) override;
  ExprPtr mutate(const MulPtr& v) override;
  ExprPtr mutate(const DivPtr& v) override;
  ExprPtr mutate(const ModPtr& v) override;
  ExprPtr mutate(const MaxPtr& v) override;
  ExprPtr mutate(const MinPtr& v) override;
  ExprPtr mutate(const AndPtr& v) override;
  ExprPtr mutate(const OrPtr& v) override;
  ExprPtr mutate(const XorPtr& v) override;
  ExprPtr mutate(const LshiftPtr& v) override;
  ExprPtr mutate(const RshiftPtr& v) override;
  ExprPtr mutate(const CompareSelectPtr& v) override;
#define IMM_MUTATE_DECLARE(Type, Name) \
  ExprPtr mutate(const Name##ImmPtr& v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE)
#undef IMM_MUTATE_DECLARE
  ExprPtr mutate(const CastPtr& v) override;
  ExprPtr mutate(const BitCastPtr& v) override;
  ExprPtr mutate(const VarPtr& v) override;
  ExprPtr mutate(const BufPtr& v) override;
  ExprPtr mutate(const RampPtr& v) override;
  ExprPtr mutate(const LoadPtr& v) override;
  ExprPtr mutate(const BroadcastPtr& v) override;
  ExprPtr mutate(const IfThenElsePtr& v) override;
  ExprPtr mutate(const IntrinsicsPtr& v) override;

  ExprPtr mutate(const TermPtr& v) override;
  ExprPtr mutate(const PolynomialPtr& v) override;
  ExprPtr mutate(const RoundOffPtr& v) override;
  ExprPtr mutate(const MaxTermPtr& v) override;
  ExprPtr mutate(const MinTermPtr& v) override;

  ExprPtr mutate(const ReduceOpPtr& v) override;

  StmtPtr mutate(const ForPtr& v) override;
  StmtPtr mutate(const BlockPtr& v) override;
  StmtPtr mutate(const StorePtr& v) override;
  StmtPtr mutate(const AtomicAddPtr& v) override;
  StmtPtr mutate(const SyncThreadsPtr& v) override;
  StmtPtr mutate(const ExternalCallPtr& v) override;
  StmtPtr mutate(const ExternalCallWithAllocPtr& v) override;

  StmtPtr mutate(const AllocatePtr& v) override;
  StmtPtr mutate(const FreePtr& v) override;
  StmtPtr mutate(const LetPtr& v) override;
  StmtPtr mutate(const CondPtr& v) override;
};

} // namespace torch::jit::tensorexpr
