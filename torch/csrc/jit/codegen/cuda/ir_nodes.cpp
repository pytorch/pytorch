#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

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

  void handle(Bool* b) override {
    same = static_cast<Bool*>(v1_)->sameAs(static_cast<Bool*>(v2_));
  }

  void handle(Float* f) override {
    same = static_cast<Float*>(v1_)->sameAs(static_cast<Float*>(v2_));
  }

  void handle(Half* h) override {
    same = static_cast<Half*>(v1_)->sameAs(static_cast<Half*>(v2_));
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

bool Bool::sameAs(const Bool* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

bool Float::sameAs(const Float* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

bool Half::sameAs(const Half* const other) const {
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

TernaryOp::TernaryOp(
    TernaryOpType _type,
    Val* _out,
    Val* _in1,
    Val* _in2,
    Val* _in3)
    : Expr(ExprType::TernaryOp),
      ternary_op_type_{_type},
      out_{_out},
      in1_{_in1},
      in2_{_in2},
      in3_{_in3} {
  addOutput(_out);
  addInput(_in1);
  addInput(_in2);
  addInput(_in3);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool TernaryOp::sameAs(const TernaryOp* other) const {
  if (getTernaryOpType() != other->getTernaryOpType())
    return false;
  if (!(in1()->sameAs(other->in1()) && in2()->sameAs(other->in2()) &&
        in3()->sameAs(other->in3())))
    return false;
  return true;
}

BroadcastOp::BroadcastOp(Val* _out, Val* _in)
    : Expr(ExprType::BroadcastOp), out_(_out), in_(_in) {
  auto out_type = _out->getValType().value();
  auto in_type = _in->getValType().value();
  if (out_type == ValType::TensorView) {
    TORCH_INTERNAL_ASSERT(
        in_type == ValType::TensorView,
        "Cannot braodcast a non-tensor object.");

    int ndims = 0;
    for (auto dom : static_cast<TensorView*>(out())->domain()->domain())
      if (!dom->isBroadcast())
        ndims++;

    TORCH_INTERNAL_ASSERT(
        ndims == (int)static_cast<TensorView*>(in_)->nDims(),
        "Invalid broadcast op. Non-broadcasted dims don't match from input to output.");
  } else {
    TORCH_INTERNAL_ASSERT(
        in_type == ValType::TensorIndex && out_type == ValType::TensorIndex,
        "Invalid types provided for broadcast op, ",
        in_type,
        " ",
        out_type,
        ".");
  }

  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool BroadcastOp::sameAs(const BroadcastOp* const other) const {
  return other->in() == in() && other->out() == out();
}

ReductionOp::ReductionOp(
    BinaryOpType _reduction_op_type,
    Val* _init,
    Val* _out,
    Val* _in)
    : Expr(ExprType::ReductionOp),
      reduction_op_type_(_reduction_op_type),
      init_(_init),
      out_(_out),
      in_(_in) {
  TORCH_INTERNAL_ASSERT(
      _init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't a constant.");
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool ReductionOp::sameAs(const ReductionOp* other) const {
  return (
      this->in()->sameAs(other->in()) &&
      this->getReductionOpType() == other->getReductionOpType() &&
      this->init()->sameAs(other->init()));
}

IterDomain::IterDomain(
    Val* _start,
    Val* _extent,
    ParallelType _parallel_method,
    bool _reduction_domain,
    bool _rfactor_domain,
    bool _broadcast_domain)
    : Val(ValType::IterDomain, DataType::Int, false),
      start_(_start),
      extent_(_extent),
      parallel_method_(_parallel_method),
      is_reduction_domain_(_reduction_domain),
      is_rfactor_domain_(_rfactor_domain),
      is_broadcast_domain_(_broadcast_domain) {
  TORCH_CHECK(
      !(is_reduction_domain_ && is_broadcast_domain_),
      "IterDomain cannot be both a broadcast and reduction domain.");
  TORCH_CHECK(
      !(is_rfactor_domain_ && is_broadcast_domain_),
      "IterDomain cannot be both a broadcast and rfactor domain.");

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
  this->name_ = fusion_->registerVal(this);
}

bool IterDomain::sameAs(const IterDomain* const other) const {
  bool is_same = isReduction() == other->isReduction() &&
      parallel_method() == other->parallel_method();
  is_same = is_same && ScalarCheck::sameAs(extent(), other->extent());
  is_same = is_same && ScalarCheck::sameAs(start(), other->start());

  return is_same;
}

IterDomain* IterDomain::merge(IterDomain* outer, IterDomain* inner) {
  TORCH_CHECK(
      outer->start()->isZeroInt() && inner->start()->isZeroInt(),
      "Merging IterDomains with starting values that aren't 0 is not supported at this time.");
  TORCH_CHECK(
      outer->isReduction() == inner->isReduction(),
      "Merging IterDomains requires that their iteration types match.");
  TORCH_CHECK(
      outer->parallel_method() == inner->parallel_method(),
      "Merging IterDomains requires that their parallel types match.");

  Val* merged_id_size = mul(outer->extent(), inner->extent());
  IterDomain* merged_id = new IterDomain(
      new Int(0),
      static_cast<Int*>(merged_id_size),
      outer->parallel_method(),
      outer->isReduction(),
      outer->isRFactorProduct() || inner->isRFactorProduct(),
      outer->isBroadcast() && inner->isBroadcast());

  new Merge(merged_id, outer, inner);

  return merged_id;
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    unsigned int factor) {
  TORCH_CHECK(
      in->start()->isZeroInt(),
      "Splitting IterDomains with starting values that aren't 0 is not supported at this time.");

  if (in->parallel_method() != ParallelType::Serial)
    TORCH_CHECK(
        false,
        "Splitting an axis of non-Serial iteration is not supported at this time."
        " Parallelization strategy must be set after calling split.");

  Int* fact = new Int(factor);
  // outer loop size
  Val* vo = ceilDiv(in->extent(), fact);

  // outer loop IterDomain
  IterDomain* ido = new IterDomain(
      new Int(0),
      static_cast<Int*>(vo),
      in->parallel_method(),
      in->isReduction(),
      in->isRFactorProduct(),
      in->isBroadcast());

  // inner loop IterDomain
  IterDomain* idi = new IterDomain(
      new Int(0),
      fact,
      in->parallel_method(),
      in->isReduction(),
      in->isRFactorProduct(),
      in->isBroadcast());
  new Split(ido, idi, in, fact);
  return {ido, idi};
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

TensorDomain::TensorDomain(std::vector<IterDomain*> _domain)
    : Val(ValType::TensorDomain), root_domain_(std::move(_domain)) {
  domain_ = std::vector<IterDomain*>(root_domain_.begin(), root_domain_.end());
  resetDomains();
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> _root_domain,
    std::vector<IterDomain*> _domain)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(_root_domain)),
      domain_(std::move(_domain)) {
  std::vector<Val*> domain_vals(domain_.begin(), domain_.end());
  auto inps = IterVisitor::getInputsTo(domain_vals);

  // Validate that the root domain consists of all inputs to _domain
  // Uncertain if this will hold for RFactor

  std::unordered_set<Val*> root_vals(root_domain_.begin(), root_domain_.end());
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of domain, but it is not found in the root domain.");
  });

  resetDomains();

  this->name_ = fusion_->registerVal(this);
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> _root_domain,
    std::vector<IterDomain*> _rfactor_domain,
    std::vector<IterDomain*> _domain)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(_root_domain)),
      domain_(std::move(_domain)),
      rfactor_domain_(std::move(_rfactor_domain)) {
  auto inps = IterVisitor::getInputsTo(
      std::vector<Val*>(domain_.begin(), domain_.end()));

  // Validate that the root domain consists of all inputs to _domain
  // Uncertain if this will hold for RFactor

  std::unordered_set<Val*> root_vals(root_domain_.begin(), root_domain_.end());
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of domain, but it is not found in the root domain.");
  });

  inps = IterVisitor::getInputsTo(
      std::vector<Val*>(rfactor_domain_.begin(), rfactor_domain_.end()));
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of the rfactor domain, but it is not found in the root domain.");
  });

  resetDomains();
  this->name_ = fusion_->registerVal(this);
}

bool TensorDomain::sameAs(const TensorDomain* const other) const {
  if (nDims() != other->nDims())
    return false;
  if (rootDomain().size() != other->rootDomain().size())
    return false;
  if (rfactorDomain().size() != other->rfactorDomain().size())
    return false;

  for (size_t i = 0; i < nDims(); i++)
    if (!(axis(i)->sameAs(other->axis(i))))
      return false;

  for (size_t i = 0; i < rootDomain().size(); i++)
    if (!(rootDomain()[i]->sameAs(other->rootDomain()[i])))
      return false;

  for (size_t i = 0; i < rfactorDomain().size(); i++)
    if (!(rfactorDomain()[i]->sameAs(other->rfactorDomain()[i])))
      return false;

  return true;
}

bool TensorDomain::sameAs(
    const std::vector<IterDomain*>& lhs,
    const std::vector<IterDomain*>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  size_t i = 0;
  for (auto td_lhs : lhs) {
    if (!td_lhs->sameAs(rhs[i++]))
      return false;
  }
  return true;
}

bool TensorDomain::hasReduction() const {
  return no_reduction_domain_.size() != domain_.size();
}

bool TensorDomain::hasBroadcast() const {
  return no_bcast_domain_.size() != domain_.size();
}

bool TensorDomain::hasRFactor() const {
  return !rfactor_domain_.empty();
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  if (i < 0)
    i += nDims();
  TORCH_CHECK(
      i >= 0 && (unsigned int)i < nDims(),
      "Tried to access axis ",
      i,
      " in domain ",
      this);
  return domain_[i];
}

size_t TensorDomain::posOf(IterDomain* id) const {
  size_t i = 0;
  while (i < domain_.size()) {
    if (domain_[i] == id)
      return i;
    i++;
  }
  TORCH_CHECK(false, "Provided id is not part of this domain.");
}

// Split "axis" into 2 axes where the inner axes is size of "factor"
// and outer axis is size axis.extent() / factor
void TensorDomain::split(int axis_, unsigned int factor) {
  if (axis_ < 0)
    axis_ += nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0 && (unsigned int)axis_ < nDims(),
      "Tried to split on axis outside TensorDomain's range.");

  IterDomain* id = axis(axis_);
  auto split_ids = IterDomain::split(id, factor);
  domain_.erase(domain_.begin() + axis_);
  domain_.insert(domain_.begin() + axis_, split_ids.second);
  domain_.insert(domain_.begin() + axis_, split_ids.first);
  resetDomains();
}

// Merge "axis" and "axis+1" into 1 dimension
void TensorDomain::merge(int axis_o, int axis_i) {
  if (axis_o < 0)
    axis_o += nDims();

  if (axis_i < 0)
    axis_i += nDims();

  TORCH_CHECK(
      axis_o >= 0 && (unsigned int)axis_o < nDims() && axis_i >= 0 &&
          (unsigned int)axis_i < nDims(),
      "Invalid merge detected, either one or both axes are outside of TensorView's range.");

  TORCH_CHECK(
      axis_o != axis_i,
      "Invalid merge detected, axes provided are the same axis.");

  if (axis_o > axis_i) {
    auto tmp = axis_i;
    axis_i = axis_o;
    axis_o = tmp;
  }

  IterDomain* first = axis(axis_o);
  IterDomain* second = axis(axis_i);

  IterDomain* merged_id = IterDomain::merge(first, second);

  domain_.erase(domain_.begin() + axis_i);
  domain_.erase(domain_.begin() + axis_o);
  domain_.insert(domain_.begin() + axis_o, merged_id);
  resetDomains();
}

// Reorder axes according to map[old_pos] = new_pos
void TensorDomain::reorder(const std::unordered_map<int, int>& old2new_) {
  domain_ = orderedAs(domain_, old2new_);
  resetDomains();
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int, int>& old2new_) {
  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> old2new;
  auto ndims = dom.size();
  std::transform(
      old2new_.begin(),
      old2new_.end(),
      std::inserter(old2new, old2new.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid

  TORCH_CHECK(
      std::none_of(
          old2new.begin(),
          old2new.end(),
          [ndims](std::unordered_map<int, int>::value_type entry) {
            return entry.first < 0 || (unsigned int)entry.first >= ndims ||
                entry.second < 0 || (unsigned int)entry.second >= ndims;
          }),
      "Reorder axes are not within the number of dimensions of the provided domain.");

  // Going to use sets, to see if any duplicate values are in the map.

  std::set<int> old_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.second;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == old2new.size() &&
          new_pos_set.size() == old2new.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // END VALIDATION CHECKS

  std::vector<int> new2old(ndims, -1);

  // Go through each old and new position, make sure they're within [0, ndims)
  for (std::pair<int, int> elem : old2new) {
    int old_pos = elem.first;
    int new_pos = elem.second;
    new2old[new_pos] = old_pos;
  }

  // old_positions that already have a new position
  std::set<int> old_positions(new2old.begin(), new2old.end());
  old_positions.erase(-1);

  // All available new positions
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
  // new2old[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  std::transform(
      new2old.begin(), new2old.end(), new2old.begin(), [&it](int i) -> int {
        return i == -1 ? *it++ : i;
      });

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  size_t size_out = 0;
  for (auto id : td)
    if (!id->isReduction())
      size_out++;
  std::vector<IterDomain*> noReductionDomain(size_out);

  int it = 0;
  for (auto id : td)
    if (!id->isReduction())
      noReductionDomain[it++] = id;

  return noReductionDomain;
}

std::vector<IterDomain*> TensorDomain::noBroadcasts(
    const std::vector<IterDomain*>& td) {
  size_t size_out = 0;
  for (auto id : td)
    if (!id->isBroadcast())
      size_out++;
  std::vector<IterDomain*> noBroadcastDomain(size_out);

  int it = 0;
  for (auto id : td)
    if (!id->isBroadcast())
      noBroadcastDomain[it++] = id;

  return noBroadcastDomain;
}

bool TensorDomain::hasBroadcast(const std::vector<IterDomain*>& td) {
  for (auto id : td)
    if (id->isBroadcast())
      return true;
  return false;
}
bool TensorDomain::hasReduction(const std::vector<IterDomain*>& td) {
  for (auto id : td)
    if (id->isReduction())
      return true;
  return false;
}

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int>& axes_) {
  std::vector<int> axes(axes_.size());

  auto ndims = nDims();
  std::transform(axes_.begin(), axes_.end(), axes.begin(), [ndims](int i) {
    return i < 0 ? i + ndims : i;
  });

  TORCH_CHECK(
      std::none_of(
          axes.begin(),
          axes.end(),
          [ndims](int i) { return i < 0 || (unsigned int)i >= ndims; }),
      "RFactor axes less than 0 or >= ndims.");

  // We might be able to lift this constraint in some instances, but needs more
  // investigation.
  TORCH_CHECK(
      !hasRFactor(), "Cannot call rfactor on the same tensor domain twice.");

  std::unordered_set<int> axes_set(axes.begin(), axes.end());

  bool rfactor_found = false;
  bool reduction_found = false;
  for (decltype(nDims()) i{0}; i < nDims(); i++) {
    if (axis(i)->isReduction()) {
      if (axes_set.find(i) != axes_set.end()) {
        rfactor_found = true;
      } else {
        reduction_found = true;
      }
    }
  }

  TORCH_CHECK(
      rfactor_found && reduction_found,
      "Invalid rfactor found, rfactor must be provided at least one reduction axis, but not all reduction axes.");

  return std::pair<TensorDomain*, TensorDomain*>{
      TransformRFactor::runReplay(this, axes),
      TransformRFactor::runReplay2(this, axes)};
}

Split::Split(
    IterDomain* _outer,
    IterDomain* _inner,
    IterDomain* _in,
    Int* _factor)
    : Expr(ExprType::Split),
      outer_{_outer},
      inner_{_inner},
      in_{_in},
      factor_{_factor} {
  addOutput(_outer);
  addOutput(_inner);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool Split::sameAs(const Split* const other) const {
  return (
      outer()->sameAs(other->outer()) && inner()->sameAs(other->inner()) &&
      in()->sameAs(other->in()) && factor()->sameAs(other->factor()));
}

Merge::Merge(IterDomain* _out, IterDomain* _outer, IterDomain* _inner)
    : Expr(ExprType::Merge), out_{_out}, outer_{_outer}, inner_{_inner} {
  addOutput(_out);
  addInput(_outer);
  addInput(_inner);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

bool Merge::sameAs(const Merge* const other) const {
  return (
      out()->sameAs(other->out()) && outer()->sameAs(other->outer()) &&
      inner()->sameAs(other->inner()));
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
    Bool* _cond,
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

Allocate::Allocate(Val* _val, Val* _size)
    : Expr(ExprType::Allocate), buffer_(_val), extent_{_size} {
  if (!_size->isAnInt() || !_size->isConstScalar()) {
    std::stringstream flat_size;
    IRPrinter irp(flat_size);
    irp.print_inline(_size);
    TORCH_INTERNAL_ASSERT(
        false,
        "Allocations must be based on constant integers but tried to alloc ",
        _val,
        " with size ",
        flat_size.str(),
        ".");
  }
  addInput(_size);
  addInput(_val);
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
