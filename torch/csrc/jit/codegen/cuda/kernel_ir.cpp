#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  std::string parallel_dim = stringifyThreadSize(p_type);
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  return ir_builder.create<NamedScalar>(parallel_dim, DataType::Int);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  std::string parallel_ind = stringifyThread(p_type);
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  return ir_builder.create<NamedScalar>(parallel_ind, DataType::Int);
}

c10::optional<ParallelType> NamedScalar::getParallelDim() const {
  if (stringifyThreadSize(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThreadSize(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThreadSize(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThreadSize(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThreadSize(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThreadSize(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

c10::optional<ParallelType> NamedScalar::getParallelIndex() const {
  if (stringifyThread(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThread(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThread(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThread(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThread(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThread(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

IterDomain::IterDomain(Passkey, Val* start, Val* extent)
    : Val(ValType::KirIterDomain, DataType::Int, true, true),
      start_(start),
      extent_(extent) {}

IterDomain::IterDomain(Passkey, const fuser::cuda::IterDomain* iter_domain)
    : Val(iter_domain),
      start_(GpuLower::lowerValue(iter_domain->start())),
      extent_(GpuLower::lowerValue(iter_domain->rawExtent())),
      parallel_type_(iter_domain->getParallelType()),
      iter_type_(iter_domain->getIterType()),
      is_rfactor_domain_(iter_domain->isRFactorProduct()) {}

Val* IterDomain::extent() const {
  TORCH_CHECK(isLoweredVal(extent_));
  if (isThread()) {
    if (extent_->getValType() == ValType::KirScalar) {
      if (extent_->as<kir::Int>()->isConst()) {
        return extent_;
      }
    }
    return NamedScalar::getParallelDim(getParallelType());
  }
  return extent_;
}

TensorDomain::TensorDomain(Passkey, std::vector<IterDomain*> domain)
    : Val(ValType::KirTensorDomain), root_domain_(std::move(domain)) {
  domain_ = root_domain_;
  resetDomains();
}

TensorDomain::TensorDomain(
    Passkey,
    const fuser::cuda::TensorDomain* tensor_domain)
    : Val(tensor_domain), contiguity_(tensor_domain->contiguity()) {
  const auto lowerIterDomains =
      [](const std::vector<fuser::cuda::IterDomain*>& domains) {
        std::vector<IterDomain*> lowered_domains;
        lowered_domains.reserve(domains.size());
        for (const auto iter_domain : domains) {
          lowered_domains.push_back(
              GpuLower::lowerValue(iter_domain)->as<IterDomain>());
        }
        return lowered_domains;
      };

  root_domain_ = lowerIterDomains(tensor_domain->getRootDomain());
  domain_ = lowerIterDomains(tensor_domain->domain());
  no_bcast_domain_ = lowerIterDomains(tensor_domain->noBroadcasts());
  no_reduction_domain_ = lowerIterDomains(tensor_domain->noReductions());
  rfactor_domain_ = lowerIterDomains(tensor_domain->getRFactorDomain());
}

bool TensorDomain::hasReduction() const {
  return no_reduction_domain_.size() != domain_.size();
}

bool TensorDomain::hasBlockReduction() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isReduction() && id->isThreadDim();
  });
}

bool TensorDomain::hasGridReduction() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isReduction() && id->isBlockDim();
  });
}

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isBroadcast() && id->isThreadDim();
  });
}

bool TensorDomain::hasBroadcast() const {
  return no_bcast_domain_.size() != domain_.size();
}

bool TensorDomain::hasRFactor() const {
  return !rfactor_domain_.empty();
}

IterDomain* TensorDomain::axis(int i) const {
  TORCH_INTERNAL_ASSERT(i >= 0 && i < int(domain_.size()));
  return domain_[i];
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> no_reduction_domains;
  for (auto id : td) {
    if (!id->isReduction()) {
      no_reduction_domains.push_back(id);
    }
  }
  return no_reduction_domains;
}

std::vector<IterDomain*> TensorDomain::noBroadcasts(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> no_broadcast_domains;
  for (auto id : td) {
    if (!id->isBroadcast()) {
      no_broadcast_domains.push_back(id);
    }
  }
  return no_broadcast_domains;
}

TensorView::TensorView(Passkey, const fuser::cuda::TensorView* tv)
    : Val(tv), fuser_tv_(tv) {
  domain_ = GpuLower::lowerValue(tv->domain())->as<TensorDomain>();
  memory_type_ = tv->getMemoryType();
}

UnaryOp::UnaryOp(Passkey, UnaryOpType type, Val* out, Val* in)
    : Expr(ExprType::KirUnaryOp), unary_op_type_{type}, out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

BinaryOp::BinaryOp(Passkey, BinaryOpType type, Val* out, Val* lhs, Val* rhs)
    : Expr(ExprType::KirBinaryOp),
      binary_op_type_{type},
      out_{out},
      lhs_{lhs},
      rhs_{rhs} {
  addOutput(out);
  addInput(lhs);
  addInput(rhs);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

TernaryOp::TernaryOp(
    Passkey,
    TernaryOpType type,
    Val* out,
    Val* in1,
    Val* in2,
    Val* in3)
    : Expr(ExprType::KirTernaryOp),
      ternary_op_type_{type},
      out_{out},
      in1_{in1},
      in2_{in2},
      in3_{in3} {
  addOutput(out);
  addInput(in1);
  addInput(in2);
  addInput(in3);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

ReductionOp::ReductionOp(
    Passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    Bool* pred)
    : Expr(ExprType::KirReductionOp),
      reduction_op_type_(reduction_op_type),
      init_(init),
      out_(out),
      in_(in),
      pred_(pred) {
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

std::vector<IterDomain*> ReductionOp::getReductionDomains() const {
  // out is a TensorIndex after lowering
  const auto out_val = out()->as<kir::TensorIndex>()->view();

  auto vec_domain = out_val->as<TensorView>()->domain()->domain();

  vec_domain.erase(
      std::remove_if(
          vec_domain.begin(),
          vec_domain.end(),
          [](IterDomain* id) { return !id->isReduction(); }),
      vec_domain.end());
  return vec_domain;
}

std::unordered_map<ParallelType, IterDomain*, TypeHash> ReductionOp::
    getParallelReductionDomains() const {
  std::unordered_map<ParallelType, IterDomain*, TypeHash> parallel_domains;
  for (auto d : getReductionDomains()) {
    if (d->isThread()) {
      parallel_domains.insert(std::make_pair(d->getParallelType(), d));
    }
  }
  return parallel_domains;
}

BroadcastOp::BroadcastOp(Passkey, Val* out, Val* in)
    : Expr(ExprType::KirBroadcastOp), out_(out), in_(in) {
  TORCH_CHECK(in->getValType().value() == ValType::TensorIndex);
  TORCH_CHECK(out->getValType().value() == ValType::TensorIndex);
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

TensorIndex::TensorIndex(
    Passkey,
    const fuser::cuda::TensorView* view,
    std::vector<Val*> indices)
    : Val(ValType::TensorIndex, view->getDataType().value(), true, true),
      view_(GpuLower::lowerValue(view)->as<TensorView>()),
      indices_(indices) {
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          indices.begin(),
          indices.end(),
          [](Val* v) {
            return (v->getValType() == ValType::KirScalar ||
                    v->getValType() == ValType::KirNamedScalar) &&
                v->getDataType() == DataType::Int;
          }),
      "Cannot index with a value other than an int.");
}

Sync::Sync(Passkey, bool war_sync) : Expr(ExprType::Sync), war_sync_(war_sync) {
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

void Scope::insert_before(Expr* ref, Expr* expr) {
  auto it = exprs_.begin();
  while (it != exprs_.end()) {
    if ((*it)->sameAs(ref))
      break;
    it++;
  }
  if (it != exprs_.end())
    exprs_.insert(it, expr);
}

void Scope::insert_after(Expr* ref, Expr* expr) {
  auto it = exprs_.begin();
  while (it != exprs_.end()) {
    if (*it == ref)
      break;
    it++;
  }
  if (it != exprs_.end())
    exprs_.insert(++it, expr);
}

void Scope::erase(Expr* ref) {
  auto it = exprs_.begin();
  while (it != exprs_.end()) {
    if (*it == ref)
      break;
    it++;
  }
  if (it != exprs_.end())
    exprs_.erase(it);
}

bool Scope::contains(Expr* expr) const {
  for (auto e : exprs_)
    if (e == expr)
      return true;
  return false;
}

void Scope::clear() {
  exprs_ = std::vector<Expr*>();
}

ForLoop::ForLoop(
    Passkey,
    Val* index,
    IterDomain* iter_domain,
    Expr* parent_scope)
    : Expr(ExprType::ForLoop),
      index_{index},
      iter_domain_{iter_domain},
      parent_scope_{parent_scope} {
  TORCH_INTERNAL_ASSERT(index->isAnInt());
  TORCH_INTERNAL_ASSERT(isLoweredScalar(index));
  addInput(index);
  addInput(iter_domain);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

void ForLoop::setParentScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      !scope_utils::exprInScope(parentScope(), this),
      "Cannot change parent scope if not already removed from previous parent.");
  parent_scope_ = scope;
}

IfThenElse::IfThenElse(Passkey, Bool* cond, Expr* parent_scope)
    : Expr(ExprType::IfThenElse), cond_{cond}, parent_scope_(parent_scope) {
  addInput(cond);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

void IfThenElse::setParentScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      !scope_utils::exprInScope(parentScope(), this),
      "Cannot change parent scope if not already removed from previous parent.");
  parent_scope_ = scope;
}

Val* TensorIndex::index(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to get an index of a 0-dim TensorIndex");
  if (i < 0)
    i += nDims();
  assert(i >= 0 && i < nDims());
  return indices_[i];
}

Allocate::Allocate(
    Passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init)
    : Expr(ExprType::Allocate),
      buffer_(buffer),
      memory_type_(memory_type),
      size_(size),
      zero_init_(zero_init) {
  if (size_ != nullptr) {
    TORCH_INTERNAL_ASSERT(
        size_->isOneInt() ||
            buffer_->getValType().value() == ValType::KirTensorView,
        "Cannot allocate a non-TensorView buffer with a size != 1, received buffer: ",
        buffer_);
  } else {
    TORCH_INTERNAL_ASSERT(
        buffer_->getValType().value() == ValType::KirTensorView);
    TORCH_INTERNAL_ASSERT(
        buffer_->as<TensorView>()->memoryType() == memory_type_);
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto domain = buffer_->as<TensorView>()->domain();
    size_ = domain->nDims() == 0 ? ir_builder.create<Int>(1)
                                 : domain->axis(0)->extent();
    for (size_t i = 1; i < domain->nDims(); i++) {
      size_ = ir_builder.mulExpr(size_, domain->axis(i)->extent());
    }
  }

  if (memory_type_ == MemoryType::Local) {
    if (!size_->isConstScalar()) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Allocations must be based on constant integers for the memory type ",
          memory_type_,
          " but tried to alloc ",
          buffer_,
          " with symbolic size.");
    }
  }

  addInput(size_);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

GridReduction::GridReduction(Passkey, ReductionOp* reduction_op)
    : Expr(ExprType::GridReduction), reduction_op_(reduction_op) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

GridReduction::GridReduction(
    Passkey,
    ReductionOp* reduction_op,
    Allocate* reduction_buffer,
    Allocate* sync_buffer,
    Bool* pred)
    : Expr(ExprType::GridReduction),
      reduction_op_(reduction_op),
      reduction_buffer_(reduction_buffer),
      sync_buffer_(sync_buffer),
      pred_(pred) {}

std::string GridReduction::getPredicateFlagName(const TensorView* val) {
  std::stringstream ss;
  ss << "T" << val->name() << "_pred";
  return ss.str();
}

// TODO(kir): remove this
std::string GridReduction::getPredicateFlagName(
    const fuser::cuda::TensorView* val) {
  std::stringstream ss;
  ss << "T" << val->name() << "_pred";
  return ss.str();
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
