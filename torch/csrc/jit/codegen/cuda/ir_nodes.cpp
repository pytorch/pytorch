
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {

namespace {
struct ScalarCheck : OptInDispatch {
  Val* v1_;
  Val* v2_;
  bool same = false;

  void handle(Float* f) override {
    same = static_cast<Float*>(v1_)->sameAs(static_cast<Float*>(v2_));
  }

  void handle(Int* i) override {
    same = static_cast<Int*>(v1_)->sameAs(static_cast<Int*>(v2_));
  }

  void handle(NamedScalar* ns) override {
    same =
        static_cast<NamedScalar*>(v1_)->sameAs(static_cast<NamedScalar*>(v2_));
  }

  ScalarCheck(Val* _v1, Val* _v2) : v1_(_v1), v2_(_v2) {
    OptInDispatch::handle(v1_);
  }

 public:
  static bool sameAs(Val* v1, Val* v2) {
    if (v1 == v2)
      return true;

    if (v1->getValType() != v2->getValType())
      return false;

    if (v1->getDataType() != v2->getDataType())
      return false;

    ScalarCheck sc(v1, v2);
    return sc.same;
  }
};
} // namespace

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
    Val* _start,
    Val* _extent,
    ParallelType _parallel_method,
    bool _reduction_domain)
    : Val(ValType::IterDomain, DataType::Int),
      start_(_start),
      extent_(_extent),
      parallel_method_(_parallel_method),
      is_reduction_domain_(_reduction_domain) {
  TORCH_INTERNAL_ASSERT(
      _extent->isAnInt(),
      "Cannot create an iter domain over an extent that is not an int but recieved ",
      _extent,
      " .");
  TORCH_INTERNAL_ASSERT(
      _start->isAnInt(),
      "Cannot create an iter domain with a start that is not an int but recieved ",
      _extent,
      " .");
}

bool IterDomain::sameAs(const IterDomain* const other) const {
  bool is_same = isReduction() == other->isReduction() &&
      parallel_method() == other->parallel_method();
  is_same = is_same && ScalarCheck::sameAs(extent(), other->extent());
  is_same = is_same && ScalarCheck::sameAs(start(), other->start());

  return is_same;
}

Val* IterDomain::extent() const {
  if (isThread()) {
    if (extent_->getValType() == ValType::Scalar)
      if (static_cast<Int*>(extent_)->isConst())
        return extent_;

    std::string parallel_dim = stringifyThreadSize(parallel_method_);
    return new NamedScalar(parallel_dim, DataType::Int);
  }
  return extent_;
}

bool TensorDomain::sameAs(const TensorDomain* const other) const {
  if (nDims() != other->nDims())
    return false;

  for (decltype(nDims()) i = 0; i < nDims(); i++)
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
    i += nDims();
  TORCH_CHECK(
      i >= 0 && i < nDims(), "Tried to access axis ", i, " in domain ", this);
  return domain_[i];
}

// Split "axis" into 2 axes where the inner axes is size of "factor"
// and outer axis is size axis.extent() / factor
TensorDomain* TensorDomain::split(int axis_, int factor) {
  if (axis_ < 0)
    axis_ += nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0 && axis_ < nDims(),
      "Tried to split on axis outside TensorDomain's range.");

  IterDomain* id = axis(axis_);

  TORCH_CHECK(
      id->start()->isZeroInt(),
      "Splitting IterDomains with starting values that aren't 0, is not supported at this time.");

  if (id->parallel_method() != ParallelType::Serial)
    TORCH_CHECK(
        false,
        "Splitting an axis of non-Serial iteration is not supported at this time."
        " Parallelization strategy must be set after calling split.");

  std::vector<IterDomain*> new_domain;

  Int* fact = new Int(factor);
  Int* one = new Int(1);

  for (decltype(nDims()) i = 0; i < nDims(); i++) {
    if (i != axis_)
      new_domain.push_back(axis(i));
    else {
      // outer loop size
      Val* vo = ceilDiv(id->extent(), fact);
      Int* so = static_cast<Int*>(vo);

      // outer loop IterDomain
      IterDomain* ido = new IterDomain(
          new Int(0), so, id->parallel_method(), id->isReduction());
      new_domain.push_back(ido);

      // inner loop IterDomain
      IterDomain* idi = new IterDomain(
          new Int(0), fact, id->parallel_method(), id->isReduction());
      new_domain.push_back(idi);
    }
  }
  TensorDomain* split_td = new TensorDomain(new_domain);
  Split* split_node =
      new Split(split_td, this, axis_, fact); // For record keeping
  return split_td;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorDomain* TensorDomain::merge(int axis_) {
  if (axis_ < 0)
    axis_ += nDims();

  TORCH_CHECK(
      axis_ >= 0 && axis_ + 1 < nDims(),
      "Trying to merge axis_ outside of TensorView's range.");

  IterDomain* first = axis(axis_);
  IterDomain* second = axis(axis_ + 1);

  TORCH_CHECK(
      first->start()->isZeroInt() && second->start()->isZeroInt(),
      "Merging IterDomains with starting values that aren't 0, is not supported at this time.");
  TORCH_CHECK(
      first->isReduction() == second->isReduction(),
      "Merging domains requires that they're either both a reduction axis_, or both an iteration axis_.");
  TORCH_CHECK(
      first->parallel_method() == second->parallel_method(),
      "Axes must have matching parallel types.");

  Val* merged_id_size = mul(first->extent(), second->extent());
  IterDomain* merged_id = new IterDomain(
      new Int(0),
      static_cast<Int*>(merged_id_size),
      first->parallel_method(),
      first->isReduction());

  std::vector<IterDomain*> new_domain;
  for (decltype(nDims()) i = 0; i < nDims(); i++) {
    if (i < axis_ || i > axis_ + 1)
      new_domain.push_back(axis(i));
    else if (i == axis_) {
      new_domain.push_back(merged_id);
    }
  }
  TensorDomain* merged_td = new TensorDomain(new_domain);
  Merge* merge_node = new Merge(merged_td, this, axis_); // For record keeping
  return merged_td;
}

// Reorder axes according to map[old_pos] = new_pos
TensorDomain* TensorDomain::reorder(
    const std::unordered_map<int, int>& axis2pos_) {
  // START VALIDATION CHECKS
  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> axis2pos;
  auto ndims = nDims();
  std::transform(
      axis2pos_.begin(),
      axis2pos_.end(),
      std::inserter(axis2pos, axis2pos.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid
  bool out_of_range = std::any_of(
      axis2pos.begin(),
      axis2pos.end(),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return entry.first < 0 || entry.first >= ndims || entry.second < 0 ||
            entry.second >= ndims;
      });

  TORCH_CHECK(
      !out_of_range,
      "TensorView reorder axes are outside the number of dimensions in the TensorView.")

  // Going to use sets, to see if any duplicate values are in the map.

  std::set<int> old_pos_set;
  std::transform(
      axis2pos.begin(),
      axis2pos.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      axis2pos.begin(),
      axis2pos.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == axis2pos.size() &&
          new_pos_set.size() == axis2pos.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // END VALIDATION CHECKS

  // Map to save, from previous order, to new order.
  std::vector<int> pos2axis(ndims, -1);

  // Go through each old and new position, make sure they're within 0-ndims
  for (std::pair<int, int> elem : axis2pos) {
    int old_pos = elem.first;
    int new_pos = elem.second;

    assert(old_pos >= 0 && old_pos < ndims && new_pos >= 0 && new_pos < ndims);

    if (pos2axis[new_pos] != -1)
      TORCH_CHECK(false, "Reorder found duplicate destination positions.");

    pos2axis[new_pos] = old_pos;
  }

  std::set<int> old_positions(pos2axis.begin(), pos2axis.end());
  old_positions.erase(-1);

  if (old_positions.size() != axis2pos.size())
    TORCH_INTERNAL_ASSERT(
        false, "Reorder found duplicate destination positions.");

  std::set<int> all_positions;
  for (decltype(ndims) i{0}; i < ndims; i++)
    all_positions.insert(i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // pos2axis[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  std::transform(
      pos2axis.begin(), pos2axis.end(), pos2axis.begin(), [&it](int i) -> int {
        return i == -1 ? *it++ : i;
      });

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      pos2axis.begin(),
      pos2axis.end(),
      std::back_inserter(reordered_domain),
      [this](int i) -> IterDomain* { return this->axis(i); });

  TensorDomain* reordered_td = new TensorDomain(reordered_domain);
  Reorder* merge_node = new Reorder(reordered_td, this, pos2axis);
  return reordered_td;
}

TensorDomain* TensorDomain::rootDomain() {
  return TransformIter::getRoot(this);
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
    IterDomain* _iter_domain,
    const std::vector<Expr*>& _body,
    Expr* _parent_scope)
    : Expr(ExprType::ForLoop),
      index_{_index},
      iter_domain_{_iter_domain},
      parent_scope_{_parent_scope} {
  TORCH_INTERNAL_ASSERT(
      _index->isAnInt(),
      "Cannot create a for loop with an index that is not an int.");
  addInput(_index);
  addInput(_iter_domain);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  for (Expr* expr : _body)
    body().push_back(expr);
}

bool ForLoop::sameAs(const ForLoop* other) const {
  if (this->iter_domain() != other->iter_domain())
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
  if (nDims() != other->nDims())
    return false;

  if (!view()->sameAs(other->view()))
    return false;

  for (decltype(nDims()) i = 0; i < nDims(); i++)
    if (!(index(i)->sameAs(other->index(i))))
      return false;

  return true;
}

Val* TensorIndex::index(int i) const {
  if (i < 0)
    i += nDims();
  assert(i >= 0 && i < nDims());
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
