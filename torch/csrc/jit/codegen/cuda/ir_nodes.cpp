
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {

bool Float::sameAs(const Float* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

bool Int::sameAs(const Int* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

UnaryOp::UnaryOp(UnaryOpType _type, Val* _out, Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool UnaryOp::sameAs(const UnaryOp* const other) const {
  if (this->type() != other->type())
    return false;
  return static_cast<const Expr*>(this)->sameAs(other);
}

BinaryOp::BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs)
    : Expr(ExprType::BinaryOp),
      binary_op_type_{_type},
      out_{_out},
      lhs_{_lhs},
      rhs_{_rhs} {
  addOutput(_out);
  addInput(_lhs);
  addInput(_rhs);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool BinaryOp::sameAs(const BinaryOp* other) const {
  if (getBinaryOpType() != other->getBinaryOpType())
    return false;
  if (!(lhs()->sameAs(other->lhs()) && rhs()->sameAs(other->rhs())))
    return false;
  return true;
}

IterDomain::IterDomain(
    Val* _size,
    ParallelType _parallel_method,
    bool _reduction_domain)
    : Val(ValType::IterDomain, DataType::Int),
      size_(_size),
      parallel_method_(_parallel_method),
      is_reduction_domain_(_reduction_domain) {
  TORCH_INTERNAL_ASSERT(
      _size->isAnInt(),
      "Cannot create an iter domain over a size that is not an int.");
}

bool IterDomain::sameAs(const IterDomain* const other) const {
  bool is_same = isReduction() == other->isReduction() &&
      parallel_method() == other->parallel_method();

  if (size()->getValType() == ValType::NamedScalar &&
      other->size()->getValType() == ValType::NamedScalar) {
    is_same = is_same &&
        (static_cast<NamedScalar*>(size())->name().compare(
             static_cast<NamedScalar*>(other->size())->name()) == 0);
  } else {
    is_same = is_same && size()->sameAs(other->size());
  }
  return is_same;
}

Val* IterDomain::size() const {
  if (isThread()) {
    if (size_->getValType() == ValType::Scalar)
      if (static_cast<Int*>(size_)->isConst())
        return size_;

    std::string parallel_dim = stringifyThreadSize(parallel_method_);
    return new NamedScalar(parallel_dim, DataType::Int);
  }
  return size_;
}

bool TensorDomain::sameAs(const TensorDomain* const other) const {
  if (size() != other->size())
    return false;

  for (decltype(size()) i = 0; i < size(); i++)
    if (!(axis(i)->sameAs(other->axis(i))))
      return false;

  return true;
}

TensorDomain* TensorDomain::noReductions() const {
  std::vector<IterDomain*> noReductionDomain;
  for (IterDomain* id : domain_)
    if (!id->isReduction())
      noReductionDomain.push_back(id);
  return new TensorDomain(noReductionDomain);
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  if (i < 0)
    i += size();
  TORCH_CHECK(
      i >= 0 && i < size(), "Tried to access axis ", i, " in domain ", this);
  return domain_[i];
}

Split::Split(TensorDomain* _out, TensorDomain* _in, int _axis, Int* _factor)
    : Expr(ExprType::Split),
      out_{_out},
      in_{_in},
      axis_{_axis},
      factor_{_factor} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool Split::sameAs(const Split* const other) const {
  return (
      out()->sameAs(other->out()) && in()->sameAs(other->in()) &&
      axis() == other->axis() && factor()->sameAs(other->factor()));
}

Merge::Merge(TensorDomain* _out, TensorDomain* _in, int _axis)
    : Expr(ExprType::Merge), out_{_out}, in_{_in}, axis_{_axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool Merge::sameAs(const Merge* const other) const {
  return (
      out()->sameAs(other->out()) && in()->sameAs(other->in()) &&
      axis() == other->axis());
}

Reorder::Reorder(
    TensorDomain* _out,
    TensorDomain* _in,
    std::vector<int> _pos2axis)
    : Expr(ExprType::Reorder),
      out_{_out},
      in_{_in},
      pos2axis_{std::move(_pos2axis)} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool Reorder::sameAs(const Reorder* const other) const {
  // Implicitly in and out matching means pos2axis matches
  return (out()->sameAs(other->out()) && in()->sameAs(other->in()));
}

ForLoop::ForLoop(
    Val* _index,
    IterDomain* _range,
    const std::vector<Expr*>& _body,
    Expr* _parent_scope)
    : Expr(ExprType::ForLoop),
      index_{_index},
      range_{_range},
      parent_scope_{_parent_scope} {
  TORCH_INTERNAL_ASSERT(
      _index->isAnInt(),
      "Cannot create a for loop with an index that is not an int.");
  addInput(_index);
  addInput(_range);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  for (Expr* expr : _body)
    body().push_back(expr);
}

bool ForLoop::sameAs(const ForLoop* other) const {
  if (this->range() != other->range())
    return false;
  if (!(constBody().sameAs(other->constBody())))
    return false;
  return other == this;
}

IfThenElse::IfThenElse(
    Int* _cond,
    const std::vector<Expr*>& _if_body,
    const std::vector<Expr*>& _else_body,
    Expr* _parent_scope)
    : Expr(ExprType::IfThenElse), cond_{_cond}, parent_scope_(_parent_scope) {
  addInput(_cond);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);

  for (auto* expr : _if_body)
    body_.push_back(expr);
  for (auto* expr : _else_body)
    else_body_.push_back(expr);
}

bool IfThenElse::sameAs(const IfThenElse* other) const {
  if (!(this->cond()->sameAs(other->cond()) &&
        this->constBody().sameAs(other->constBody()) &&
        this->constElseBody().sameAs(other->constElseBody())))
    return false;
  return true;
}

bool TensorIndex::sameAs(const TensorIndex* const other) const {
  if (size() != other->size())
    return false;

  if (!view()->sameAs(other->view()))
    return false;

  for (decltype(size()) i = 0; i < size(); i++)
    if (!(index(i)->sameAs(other->index(i))))
      return false;

  return true;
}

Val* TensorIndex::index(int i) const {
  if (i < 0)
    i += size();
  assert(i >= 0 && i < size());
  return indices_[i];
}

Allocate::Allocate(TensorView* _tv, Val* _size)
    : Expr(ExprType::Allocate), buffer_(_tv), extent_{_size} {
  if (!_size->isAnInt() || !_size->isConstScalar()) {
    std::stringstream flat_size;
    IRPrinter irp(flat_size);
    irp.print_inline(_size);
    TORCH_INTERNAL_ASSERT(
        false,
        "Allocations must be based on constant integers but tried to alloc ",
        _tv,
        " with size ",
        flat_size.str(),
        ".");
  }
  addInput(_size);
  addInput(_tv);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

DataType Allocate::buf_type() const {
  return buffer_->getDataType().value();
}

bool Allocate::sameAs(const Allocate* other) const {
  if (!this->buffer_->sameAs(other->buffer()))
    return false;
  if (!this->extent()->sameAs(other->extent()))
    return false;
  if (this->type() != other->type())
    return false;

  return true;
}

} // namespace fuser
} // namespace jit
} // namespace torch
