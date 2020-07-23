
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>

namespace torch {
namespace jit {
namespace fuser {
namespace kir {

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
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
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
  name_ = FusionGuard::getCurFusion()->registerExpr(this);

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
    }
  }
  addInput(size_);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Allocate::Allocate(const Allocate* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      buffer_(ir_cloner->clone(src->buffer_)),
      memory_type_(src->memory_type_),
      size_(ir_cloner->clone(src->size_)) {}

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

std::string getPredicateFlagName(const TensorView* val) {
  std::stringstream ss;
  ss << "T" << val->name() << "_pred";
  return ss.str();
}

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
