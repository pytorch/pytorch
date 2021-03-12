#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

void Node::print() const {
  std::cout << "\n";
  IrPrinter(std::cout).printNode(this);
  std::cout << "\n";
}

Val::Val(Passkey passkey, DataType dtype) : Node(passkey), dtype_(dtype) {
  // NOLINTNEXTLINE: https://bugs.llvm.org/show_bug.cgi?id=48534
  id_ = passkey.kernel->newValueId(passkey);
}

Expr* Expr::parentScope() const {
  if (scope()) {
    return scope()->owner();
  } else {
    return nullptr;
  }
}

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

IterDomain::IterDomain(Passkey passkey, Val* start, Val* extent)
    : Val(passkey, DataType::Int), start_(start), extent_(extent) {}

IterDomain::IterDomain(
    Passkey passkey,
    const fuser::cuda::IterDomain* iter_domain)
    : Val(passkey, iter_domain->getDataType().value()),
      start_(GpuLower::current()->lowerValue(iter_domain->start())),
      extent_(GpuLower::current()->lowerValue(iter_domain->rawExtent())),
      parallel_type_(iter_domain->getParallelType()),
      iter_type_(iter_domain->getIterType()),
      is_rfactor_domain_(iter_domain->isRFactorProduct()),
      is_simple_(iter_domain->definition() == nullptr) {
  // preserve the fusion node's name
  setName(iter_domain->name());
}

Val* IterDomain::extent() const {
  TORCH_INTERNAL_ASSERT(extent_ != nullptr);
  if (isThread()) {
    if (extent_->isScalar() && extent_->isConst()) {
      return extent_;
    }
    return NamedScalar::getParallelDim(parallelType());
  }
  return extent_;
}

TensorDomain::TensorDomain(Passkey passkey, std::vector<IterDomain*> domain)
    : Val(passkey, DataType::Null), root_domain_(std::move(domain)) {
  domain_ = root_domain_;
  resetDomains();
}

TensorDomain::TensorDomain(
    Passkey passkey,
    const fuser::cuda::TensorDomain* tensor_domain)
    : Val(passkey, DataType::Null), contiguity_(tensor_domain->contiguity()) {
  // preserve the fusion node's name
  setName(tensor_domain->name());

  const auto lowerIterDomains =
      [](const std::vector<fuser::cuda::IterDomain*>& domains) {
        std::vector<IterDomain*> lowered_domains;
        lowered_domains.reserve(domains.size());
        for (const auto iter_domain : domains) {
          lowered_domains.push_back(
              GpuLower::current()->lowerValue(iter_domain)->as<IterDomain>());
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

TensorView::TensorView(Passkey passkey, const fuser::cuda::TensorView* tv)
    : Val(passkey, tv->getDataType().value()), fuser_tv_(tv) {
  setName(tv->name());
  domain_ = GpuLower::current()->lowerValue(tv->domain())->as<TensorDomain>();
  memory_type_ = tv->getMemoryType();
}

TensorView::TensorView(
    Passkey passkey,
    DataType dtype,
    TensorDomain* domain,
    MemoryType memory_type)
    : Val(passkey, dtype), domain_(domain), memory_type_(memory_type) {}

UnaryOp::UnaryOp(Passkey passkey, UnaryOpType operation, Val* out, Val* in)
    : Expr(passkey), operation_(operation), out_(out), in_(in) {
  addOutput(out);
  addInput(in);
}

BinaryOp::BinaryOp(
    Passkey passkey,
    BinaryOpType operation,
    Val* out,
    Val* lhs,
    Val* rhs)
    : Expr(passkey), operation_(operation), out_(out), lhs_(lhs), rhs_(rhs) {
  addOutput(out);
  addInput(lhs);
  addInput(rhs);
}

TernaryOp::TernaryOp(
    Passkey passkey,
    TernaryOpType operation,
    Val* out,
    Val* in1,
    Val* in2,
    Val* in3)
    : Expr(passkey),
      operation_(operation),
      out_(out),
      in1_(in1),
      in2_(in2),
      in3_(in3) {
  addOutput(out);
  addInput(in1);
  addInput(in2);
  addInput(in3);
}

ReductionOp::ReductionOp(
    Passkey passkey,
    BinaryOpType operation,
    Val* init,
    Val* out,
    Val* in)
    : Expr(passkey), operation_(operation), init_(init), out_(out), in_(in) {
  addOutput(out);
  addInput(in);
}

WelfordOp::WelfordOp(
    Passkey passkey,
    Val* out_var,
    Val* out_avg,
    Val* out_N,
    Val* init_var,
    Val* init_avg,
    Val* init_N,
    Val* in_var,
    Val* in_avg,
    Val* in_N)
    : Expr(passkey),
      out_var_(out_var),
      out_avg_(out_avg),
      out_N_(out_N),
      init_var_(init_var),
      init_avg_(init_avg),
      init_N_(init_N),
      in_var_(in_var),
      in_avg_(in_avg),
      in_N_(in_N) {
  addOutput(out_avg);
  addOutput(out_var);
  addOutput(out_N);

  if (!in_N->isOneInt()) {
    addInput(in_var);
  }
  addInput(in_avg);
  addInput(in_N);
}

std::vector<IterDomain*> WelfordOp::getReductionDomains() const {
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

std::unordered_map<ParallelType, IterDomain*, TypeHash> WelfordOp::
    getParallelReductionDomains() const {
  std::unordered_map<ParallelType, IterDomain*, TypeHash> parallel_domains;
  for (auto d : getReductionDomains()) {
    if (d->isThread()) {
      parallel_domains.insert(std::make_pair(d->parallelType(), d));
    }
  }
  return parallel_domains;
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
      parallel_domains.insert(std::make_pair(d->parallelType(), d));
    }
  }
  return parallel_domains;
}

BroadcastOp::BroadcastOp(Passkey passkey, Val* out, Val* in)
    : Expr(passkey), out_(out), in_(in) {
  TORCH_CHECK(in->isA<TensorIndex>() || in->isA<TensorView>());
  TORCH_CHECK(out->isA<TensorIndex>() || out->isA<TensorView>());
  addOutput(out);
  addInput(in);
}

TensorIndex::TensorIndex(
    Passkey passkey,
    const fuser::cuda::TensorView* view,
    std::vector<Val*> indices)
    : Val(passkey, view->getDataType().value()),
      view_(GpuLower::current()->lowerValue(view)->as<TensorView>()),
      indices_(indices) {
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          indices.begin(),
          indices.end(),
          [](Val* v) { return v->dtype() == DataType::Int; }),
      "Cannot index with a value other than an int.");
}

Sync::Sync(Passkey passkey, bool war_sync)
    : Expr(passkey), war_sync_(war_sync) {}

void Scope::insert(std::vector<Expr*>::const_iterator pos, Expr* expr) {
  exprs_.insert(pos, expr);
  expr->setScope(this);
}

void Scope::insert_before(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " before the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it, expr);
}

void Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it + 1, expr);
}

void Scope::insert(size_t pos, Expr* expr) {
  const auto it = exprs_.begin() + pos;
  insert(it, expr);
}

void Scope::erase(std::vector<Expr*>::const_iterator pos) {
  // Remove the scope of the expr if this is the scope
  auto expr = *pos;
  TORCH_INTERNAL_ASSERT(
      expr->scope() == this,
      "Inconsistent scoping of expression detected: ",
      kir::toString(expr));
  expr->setScope(nullptr);
  exprs_.erase(pos);
}

void Scope::erase(Expr* ref) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  if (it != exprs_.end()) {
    erase(it);
  }
}

void Scope::erase(size_t pos) {
  TORCH_INTERNAL_ASSERT(pos < size());
  erase(exprs_.begin() + pos);
}

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

ForLoop::ForLoop(
    Passkey passkey,
    Val* index,
    IterDomain* iter_domain,
    bool unroll)
    : Expr(passkey),
      index_{index},
      iter_domain_{iter_domain},
      body_(this),
      unroll_(unroll) {
  TORCH_INTERNAL_ASSERT(index->dtype() == DataType::Int);
  addInput(index);
  addInput(iter_domain);
}

IfThenElse::IfThenElse(Passkey passkey, Bool* cond)
    : Expr(passkey), cond_{cond}, then_body_(this), else_body_(this) {
  addInput(cond);
}

Val* TensorIndex::index(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to get an index of a 0-dim TensorIndex");
  if (i < 0)
    i += nDims();
  TORCH_INTERNAL_ASSERT(i >= 0 && i < int(nDims()));
  return indices_[i];
}

Allocate::Allocate(
    Passkey passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init)
    : Expr(passkey),
      buffer_(buffer),
      memory_type_(memory_type),
      size_(size),
      zero_init_(zero_init) {
  if (size_ != nullptr) {
    TORCH_INTERNAL_ASSERT(size_->isOneInt() || buffer_->isA<TensorView>());
  } else {
    TORCH_INTERNAL_ASSERT(buffer_->isA<TensorView>());
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
  addInput(size_);
}

GridReduction::GridReduction(Passkey passkey, ReductionOp* reduction_op)
    : Expr(passkey), reduction_op_(reduction_op) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

GridReduction::GridReduction(
    Passkey passkey,
    ReductionOp* reduction_op,
    Allocate* reduction_buffer,
    Allocate* sync_buffer)
    : Expr(passkey),
      reduction_op_(reduction_op),
      reduction_buffer_(reduction_buffer),
      sync_buffer_(sync_buffer) {}

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

GridWelford::GridWelford(
    Passkey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer)
    : Expr(passkey),
      welford_op_(welford_op),
      var_buffer_(var_buffer),
      avg_buffer_(avg_buffer),
      n_buffer_(n_buffer),
      sync_buffer_(sync_buffer) {}

std::string GridWelford::getPredicateFlagName(const TensorView* val) {
  std::stringstream ss;
  ss << "T" << val->name() << "_pred";
  return ss.str();
}

// TODO(kir): remove this
std::string GridWelford::getPredicateFlagName(
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
