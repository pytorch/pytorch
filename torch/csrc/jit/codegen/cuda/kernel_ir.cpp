
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

// TODO: remove
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>

namespace torch {
namespace jit {
namespace fuser {
namespace kir {

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  std::string parallel_dim = stringifyThreadSize(p_type);
  return new NamedScalar(parallel_dim, DataType::Int);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  std::string parallel_ind = stringifyThread(p_type);
  return new NamedScalar(parallel_ind, DataType::Int);
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

UnaryOp::UnaryOp(UnaryOpType type, Val* out, Val* in)
    : Expr(ExprType::KirUnaryOp), unary_op_type_{type}, out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

UnaryOp::UnaryOp(const UnaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      unary_op_type_(src->unary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

BinaryOp::BinaryOp(BinaryOpType type, Val* out, Val* lhs, Val* rhs)
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

BinaryOp::BinaryOp(const BinaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      binary_op_type_(src->binary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      lhs_(ir_cloner->clone(src->lhs_)),
      rhs_(ir_cloner->clone(src->rhs_)) {}

TernaryOp::TernaryOp(const TernaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      ternary_op_type_(src->ternary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in1_(ir_cloner->clone(src->in1_)),
      in2_(ir_cloner->clone(src->in2_)),
      in3_(ir_cloner->clone(src->in3_)) {}

ReductionOp::ReductionOp(
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in)
    : Expr(ExprType::KirReductionOp),
      reduction_op_type_(reduction_op_type),
      init_(init),
      out_(out),
      in_(in) {
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

ReductionOp::ReductionOp(const ReductionOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_type_(src->reduction_op_type_),
      init_(ir_cloner->clone(src->init_)),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

std::vector<IterDomain*> ReductionOp::getReductionDomains() const {
  const Val* out_val = out();
  TORCH_INTERNAL_ASSERT(
      out_val->getValType() == ValType::TensorView ||
          out_val->getValType() == ValType::TensorIndex,
      "Output of reduction must be TensorView or TensorIndex");

  // out is a TensorIndex after lowering
  if (out_val->getValType() == ValType::TensorIndex) {
    out_val = out_val->as<kir::TensorIndex>()->view();
  }

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

BroadcastOp::BroadcastOp(Val* out, Val* in)
    : Expr(ExprType::KirBroadcastOp), out_(out), in_(in) {
  TORCH_CHECK(in->getValType().value() == ValType::TensorIndex);
  TORCH_CHECK(out->getValType().value() == ValType::TensorIndex);
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
}

BroadcastOp::BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

TensorIndex::TensorIndex(const TensorView* view, std::vector<Val*> indices)
    : Val(ValType::TensorIndex, view->getDataType().value()),
      view_(view),
      indices_(indices) {
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          indices.begin(),
          indices.end(),
          [](Val* v) {
            return (v->getValType() == ValType::Scalar ||
                    v->getValType() == ValType::NamedScalar) &&
                v->getDataType() == DataType::Int;
          }),
      "Cannot index with a value other than an int.");
}

TensorIndex::TensorIndex(const TensorIndex* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      view_(ir_cloner->clone(src->view_)),
      indices_(ir_cloner->clone(src->indices_)) {}

Scope::Scope(const Scope* src, IrCloner* ir_cloner)
    : exprs_(ir_cloner->clone(src->exprs_)) {}

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
    Val* index,
    IterDomain* iter_domain,
    const std::vector<Expr*>& body,
    Expr* parent_scope)
    : Expr(ExprType::ForLoop),
      index_{index},
      iter_domain_{iter_domain},
      parent_scope_{parent_scope} {
  TORCH_INTERNAL_ASSERT(
      index->isAnInt(),
      "Cannot create a for loop with an index that is not an int.");
  addInput(index);
  addInput(iter_domain);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);
  for (Expr* expr : body) {
    body_.push_back(expr);
  }
}

ForLoop::ForLoop(const ForLoop* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      index_(ir_cloner->clone(src->index_)),
      iter_domain_(ir_cloner->clone(src->iter_domain_)),
      body_(&src->body_, ir_cloner),
      parent_scope_(ir_cloner->clone(src->parent_scope_)) {}

IfThenElse::IfThenElse(
    Bool* cond,
    const std::vector<Expr*>& if_body,
    const std::vector<Expr*>& else_body,
    Expr* parent_scope)
    : Expr(ExprType::IfThenElse), cond_{cond}, parent_scope_(parent_scope) {
  addInput(cond);
  name_ = FusionGuard::getCurFusion()->registerLoweredExpr(this);

  for (auto* expr : if_body)
    body_.push_back(expr);
  for (auto* expr : else_body)
    else_body_.push_back(expr);
}

IfThenElse::IfThenElse(const IfThenElse* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      cond_(src->cond_),
      body_(&src->body_, ir_cloner),
      else_body_(&src->else_body_, ir_cloner),
      parent_scope_(ir_cloner->clone(src->parent_scope_)) {}

Val* TensorIndex::index(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to get an index of a 0-dim TensorIndex");
  if (i < 0)
    i += nDims();
  assert(i >= 0 && i < nDims());
  return indices_[i];
}

Allocate::Allocate(Val* buffer, MemoryType memory_type, Val* size)
    : Expr(ExprType::Allocate),
      buffer_(buffer),
      memory_type_(memory_type),
      size_(size) {
  if (size_ != nullptr) {
    TORCH_INTERNAL_ASSERT(
        size_->isOneInt() ||
            buffer_->getValType().value() == ValType::TensorView,
        "Cannot allocate a non-TensorView buffer with a size != 1, received buffer: ",
        buffer_);
  } else {
    if (buffer_->getValType().value() == ValType::TensorView) {
      auto tv = buffer_->as<TensorView>();
      size_ = tv->nDims() == 0 ? new Int(1) : tv->axis(0)->extent();
      for (size_t i = 1; i < tv->nDims(); i++) {
        auto result = new Int();
        new BinaryOp(BinaryOpType::Mul, result, size_, tv->axis(i)->extent());
        size_ = result;
      }
    }
  }

  if ((memory_type_ == MemoryType::Local ||
       memory_type_ == MemoryType::Shared)) {
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

Allocate::Allocate(const Allocate* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      buffer_(ir_cloner->clone(src->buffer_)),
      memory_type_(src->memory_type_),
      size_(ir_cloner->clone(src->size_)) {}

Sync::Sync() : Expr(ExprType::Sync) {
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Sync::Sync(const Sync* src, IrCloner* ir_cloner) : Expr(src, ir_cloner) {}

GridReduction::GridReduction(ReductionOp* reduction_op)
    : Expr(ExprType::GridReduction), reduction_op_(reduction_op) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

GridReduction::GridReduction(
    ReductionOp* reduction_op,
    kir::Allocate* reduction_buffer,
    kir::Allocate* sync_buffer)
    : Expr(ExprType::GridReduction),
      reduction_op_(reduction_op),
      reduction_buffer_(reduction_buffer),
      sync_buffer_(sync_buffer) {}

GridReduction::GridReduction(const GridReduction* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_(ir_cloner->clone(src->reduction_op_)),
      reduction_buffer_(ir_cloner->clone(src->reduction_buffer_)),
      sync_buffer_(ir_cloner->clone(src->sync_buffer_)) {}

std::string GridReduction::getPredicateFlagName(const TensorView* val) {
  std::stringstream ss;
  ss << "T" << val->name() << "pred";
  return ss.str();
}

Val* andExpr(Val* v1, Val* v2) {
  auto result = new Bool();
  new BinaryOp(BinaryOpType::And, result, v1, v2);
  return result;
}

Val* eqExpr(Val* v1, Val* v2) {
  auto result = new Bool();
  new BinaryOp(BinaryOpType::Eq, result, v1, v2);
  return result;
}

Val* ltExpr(Val* v1, Val* v2) {
  auto result = new Bool();
  new BinaryOp(BinaryOpType::LT, result, v1, v2);
  return result;
}

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
