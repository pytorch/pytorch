
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

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
    Int* _size,
    ParallelType _parallel_method,
    bool _reduction_domain)
    : Val(ValType::IterDomain, DataType::Int),
      size_(_size),
      parallel_method_(_parallel_method),
      is_reduction_domain_(_reduction_domain) {}

IterDomain::IterDomain(
    Val* int_size,
    ParallelType _parallel_method,
    bool _reduction_domain)
    : Val(ValType::IterDomain, DataType::Int),
      size_(static_cast<Int*>(int_size)),
      parallel_method_(_parallel_method),
      is_reduction_domain_(_reduction_domain) {
  assert(int_size->isVal());
  assert(int_size->getDataType() == DataType::Int);
}

bool IterDomain::sameAs(const IterDomain* const other) const {
  return (
      isReduction() == other->isReduction() &&
      parallel_method() == other->parallel_method() &&
      size()->sameAs(other->size()));
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
  for (IterDomain* id : domain)
    if (!id->isReduction())
      noReductionDomain.push_back(id);
  return new TensorDomain(noReductionDomain);
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  if (i < 0)
    i += size();
  assert(i >= 0 && i < size());
  return domain[i];
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

} // namespace fuser
} // namespace jit
} // namespace torch
