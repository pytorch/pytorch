#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

namespace torch::jit::tensorexpr {

class TORCH_API IRVisitor {
 public:
  virtual ~IRVisitor() = default;
  virtual void visit(const AddPtr& v);
  virtual void visit(const SubPtr& v);
  virtual void visit(const MulPtr& v);
  virtual void visit(const DivPtr& v);
  virtual void visit(const ModPtr& v);
  virtual void visit(const MaxPtr& v);
  virtual void visit(const MinPtr& v);
  virtual void visit(const AndPtr& v);
  virtual void visit(const OrPtr& v);
  virtual void visit(const XorPtr& v);
  virtual void visit(const LshiftPtr& v);
  virtual void visit(const RshiftPtr& v);
  virtual void visit(const CompareSelectPtr& v);

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(const Name##ImmPtr& v);

  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

  virtual void visit(const CastPtr& v);
  virtual void visit(const BitCastPtr& v);
  virtual void visit(const VarPtr& v);
  virtual void visit(const BufPtr& v);
  virtual void visit(const RampPtr& v);
  virtual void visit(const LoadPtr& v);
  virtual void visit(const ForPtr& v);
  virtual void visit(const BlockPtr& v);
  virtual void visit(const StorePtr& v);
  virtual void visit(const BroadcastPtr& v);
  virtual void visit(const IfThenElsePtr& v);
  virtual void visit(const IntrinsicsPtr& v);
  virtual void visit(const AllocatePtr& v);
  virtual void visit(const FreePtr& v);
  virtual void visit(const FreeExtPtr& v);
  virtual void visit(const PlacementAllocatePtr& v);
  virtual void visit(const LetPtr& v);
  virtual void visit(const CondPtr& v);
  virtual void visit(const TermPtr& v);
  virtual void visit(const PolynomialPtr& v);
  virtual void visit(const RoundOffPtr& v);
  virtual void visit(const MaxTermPtr& v);
  virtual void visit(const MinTermPtr& v);
  virtual void visit(const ReduceOpPtr& v);
  virtual void visit(const AtomicAddPtr& v);
  virtual void visit(const SyncThreadsPtr& v);
  virtual void visit(const ExternalCallPtr& v);
  virtual void visit(const ExternalCallWithAllocPtr& v);
};

} // namespace torch::jit::tensorexpr
