#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

namespace torch::jit::tensorexpr {

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() = default;
  virtual ExprPtr mutate(const AddPtr& v);
  virtual ExprPtr mutate(const SubPtr& v);
  virtual ExprPtr mutate(const MulPtr& v);
  virtual ExprPtr mutate(const DivPtr& v);
  virtual ExprPtr mutate(const ModPtr& v);
  virtual ExprPtr mutate(const MaxPtr& v);
  virtual ExprPtr mutate(const MinPtr& v);
  virtual ExprPtr mutate(const AndPtr& v);
  virtual ExprPtr mutate(const OrPtr& v);
  virtual ExprPtr mutate(const XorPtr& v);
  virtual ExprPtr mutate(const LshiftPtr& v);
  virtual ExprPtr mutate(const RshiftPtr& v);
  virtual ExprPtr mutate(const CompareSelectPtr& v);
#define IMM_MUTATE_DECLARE(Type, Name) \
  virtual ExprPtr mutate(const Name##ImmPtr& v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE)
#undef IMM_MUTATE_DECLARE
  virtual ExprPtr mutate(const CastPtr& v);
  virtual ExprPtr mutate(const BitCastPtr& v);
  virtual ExprPtr mutate(const VarPtr& v);
  virtual ExprPtr mutate(const BufPtr& v);
  virtual ExprPtr mutate(const RampPtr& v);
  virtual ExprPtr mutate(const LoadPtr& v);
  virtual ExprPtr mutate(const BroadcastPtr& v);
  virtual ExprPtr mutate(const IfThenElsePtr& v);
  virtual ExprPtr mutate(const IntrinsicsPtr& v);

  virtual ExprPtr mutate(const TermPtr& v);
  virtual ExprPtr mutate(const PolynomialPtr& v);
  virtual ExprPtr mutate(const RoundOffPtr& v);
  virtual ExprPtr mutate(const MaxTermPtr& v);
  virtual ExprPtr mutate(const MinTermPtr& v);

  virtual ExprPtr mutate(const ReduceOpPtr& v);

  virtual StmtPtr mutate(const ForPtr& v);
  virtual StmtPtr mutate(const BlockPtr& v);
  virtual StmtPtr mutate(const StorePtr& v);
  virtual StmtPtr mutate(const AtomicAddPtr& v);
  virtual StmtPtr mutate(const SyncThreadsPtr& v);
  virtual StmtPtr mutate(const ExternalCallPtr& v);
  virtual StmtPtr mutate(const ExternalCallWithAllocPtr& v);

  virtual StmtPtr mutate(const AllocatePtr& v);
  virtual StmtPtr mutate(const FreePtr& v);
  virtual StmtPtr mutate(const FreeExtPtr& v);
  virtual StmtPtr mutate(const PlacementAllocatePtr& v);
  virtual StmtPtr mutate(const LetPtr& v);
  virtual StmtPtr mutate(const CondPtr& v);
};

} // namespace torch::jit::tensorexpr
