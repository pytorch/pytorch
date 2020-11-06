#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ScalarCheck : OptInConstDispatch {
 public:
  static bool sameAs(const Val* v1, const Val* v2) {
    if (v1 == v2)
      return true;

    if (v1->getValType() != v2->getValType())
      return false;

    if (v1->getDataType() != v2->getDataType())
      return false;

    ScalarCheck sc(v1, v2);
    return sc.same_;
  }

 private:
  void handle(const Bool* b) override {
    same_ = v1_->as<Bool>()->sameAs(v2_->as<Bool>());
  }

  void handle(const Float* f) override {
    same_ = v1_->as<Float>()->sameAs(v2_->as<Float>());
  }

  void handle(const Half* h) override {
    same_ = v1_->as<Half>()->sameAs(v2_->as<Half>());
  }

  void handle(const Int* i) override {
    same_ = v1_->as<Int>()->sameAs(v2_->as<Int>());
  }

  void handle(const NamedScalar* ns) override {
    same_ = v1_->as<NamedScalar>()->sameAs(v2_->as<NamedScalar>());
  }

  ScalarCheck(const Val* _v1, const Val* _v2) : v1_(_v1), v2_(_v2) {
    OptInConstDispatch::handle(v1_);
  }

 private:
  const Val* v1_ = nullptr;
  const Val* v2_ = nullptr;
  bool same_ = false;
};

} // namespace

bool areEqualScalars(Val* v1, Val* v2) {
  return ScalarCheck::sameAs(v1, v2);
}

Bool::Bool(const Bool* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Bool::sameAs(const Bool* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Float::Float(const Float* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Float::sameAs(const Float* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Half::Half(const Half* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Half::sameAs(const Half* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Int::Int(const Int* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Int::sameAs(const Int* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

UnaryOp::UnaryOp(UnaryOpType type, Val* out, Val* in)
    : Expr(ExprType::UnaryOp), unary_op_type_{type}, out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

UnaryOp::UnaryOp(const UnaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      unary_op_type_(src->unary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool UnaryOp::sameAs(const UnaryOp* const other) const {
  if (type() != other->type())
    return false;
  return as<Expr>()->sameAs(other);
}

BinaryOp::BinaryOp(BinaryOpType type, Val* out, Val* lhs, Val* rhs)
    : Expr(ExprType::BinaryOp),
      binary_op_type_{type},
      out_{out},
      lhs_{lhs},
      rhs_{rhs} {
  addOutput(out);
  addInput(lhs);
  addInput(rhs);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(const BinaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      binary_op_type_(src->binary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      lhs_(ir_cloner->clone(src->lhs_)),
      rhs_(ir_cloner->clone(src->rhs_)) {}

bool BinaryOp::sameAs(const BinaryOp* other) const {
  if (getBinaryOpType() != other->getBinaryOpType())
    return false;
  if (!(lhs()->sameAs(other->lhs()) && rhs()->sameAs(other->rhs())))
    return false;
  return true;
}

TernaryOp::TernaryOp(TernaryOpType type, Val* out, Val* in1, Val* in2, Val* in3)
    : Expr(ExprType::TernaryOp),
      ternary_op_type_{type},
      out_{out},
      in1_{in1},
      in2_{in2},
      in3_{in3} {
  addOutput(out);
  addInput(in1);
  addInput(in2);
  addInput(in3);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

TernaryOp::TernaryOp(const TernaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      ternary_op_type_(src->ternary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in1_(ir_cloner->clone(src->in1_)),
      in2_(ir_cloner->clone(src->in2_)),
      in3_(ir_cloner->clone(src->in3_)) {}

bool TernaryOp::sameAs(const TernaryOp* other) const {
  if (getTernaryOpType() != other->getTernaryOpType())
    return false;
  if (!(in1()->sameAs(other->in1()) && in2()->sameAs(other->in2()) &&
        in3()->sameAs(other->in3())))
    return false;
  return true;
}

BroadcastOp::BroadcastOp(Val* out, Val* in, std::vector<bool> is_broadcast_dims)
    : Expr(ExprType::BroadcastOp),
      out_(out),
      in_(in),
      is_broadcast_dims_(std::move(is_broadcast_dims)) {
  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot braodcast a non-tensor object.");

  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);

  // This is a generic check that root dims of a consumer and producer match.
  // Maybe we shouldn't relegate it to this constructor.
  const auto c_tv = out_->as<TensorView>();
  const auto p_tv = in_->as<TensorView>();

  const auto& c_root = c_tv->getRootDomain();
  const auto& p_root = p_tv->getMaybeRFactorDomain();

  const auto root_p2c =
      PairwiseRootDomainMap(p_tv, c_tv)
          .mapProducerToConsumer(p_tv->domain(), c_tv->domain());

  for (auto id : p_root) {
    if (root_p2c.find(id) == root_p2c.end()) {
      TORCH_INTERNAL_ASSERT(
          id->isReduction(),
          "Invalid broadcast op: ",
          id,
          ". Non-reduction input dim does't match to output.");
    }
  }

  std::unordered_set<IterDomain*> c_mapped;
  for (auto pair_entry : root_p2c) {
    c_mapped.insert(pair_entry.second);
  }

  for (size_t i = 0; i < c_root.size(); ++i) {
    const auto c_id = c_root[i];
    if (c_mapped.find(c_id) != c_mapped.end()) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        c_id->isBroadcast() && is_broadcast_dims_[i],
        "Invalid broadcast op: ",
        c_id,
        ". Non-broadcasted output dim isn't matched from input.");
  }
}

BroadcastOp::BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      is_broadcast_dims_(src->is_broadcast_dims_) {}

bool BroadcastOp::sameAs(const BroadcastOp* const other) const {
  return other->in() == in() && other->out() == out();
}

ReductionOp::ReductionOp(
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in)
    : Expr(ExprType::ReductionOp),
      reduction_op_type_(reduction_op_type),
      init_(init),
      out_(out),
      in_(in) {
  TORCH_CHECK(out->getValType().value() == ValType::TensorView);

  TORCH_INTERNAL_ASSERT(
      in->getValType() == ValType::TensorView &&
          out->getValType() == ValType::TensorView,
      "Reduction operation was created that does not have tensor inputs and outputs.");

  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(in->as<TensorView>()->getMaybeRFactorDomain())
              .size() == out->as<TensorView>()->getRootDomain().size(),
      "Reduction operation created with mismatched domains.");

  TORCH_INTERNAL_ASSERT(
      init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't a constant.");

  addOutput(out);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

ReductionOp::ReductionOp(const ReductionOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_type_(src->reduction_op_type_),
      init_(ir_cloner->clone(src->init_)),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool ReductionOp::sameAs(const ReductionOp* other) const {
  return (
      in()->sameAs(other->in()) &&
      getReductionOpType() == other->getReductionOpType() &&
      init()->sameAs(other->init()));
}

IterDomain::IterDomain(
    Val* start,
    Val* extent,
    ParallelType parallel_type,
    IterType iter_type,
    bool is_rfactor_domain)
    : Val(ValType::IterDomain, DataType::Int, false),
      start_(start),
      extent_(extent),
      parallel_type_(parallel_type),
      iter_type_(iter_type),
      is_rfactor_domain_(is_rfactor_domain) {
  TORCH_CHECK(
      !(isRFactorProduct() && isBroadcast()),
      "IterDomain cannot be both a broadcast and rfactor domain.");

  TORCH_INTERNAL_ASSERT(
      extent->isAnInt(),
      "Cannot create an iter domain over an extent that is not an int but received ",
      extent,
      " .");

  TORCH_INTERNAL_ASSERT(
      start->isAnInt(),
      "Cannot create an iter domain with a start that is not an int but received ",
      extent,
      " .");

  // Check that all for-loops iterate from zero to some positive integer
  // lower_insert_syncs uses this assumption for correctness.
  TORCH_INTERNAL_ASSERT(
      start->isZeroInt(),
      "Cannot create an iter domain with a start that is non-zero but received ",
      extent,
      " .");

  TORCH_INTERNAL_ASSERT(
      !extent->isZeroInt(),
      "Cannot create an iter domain with a extent that is zero but received ",
      extent,
      " .");

  name_ = fusion_->registerVal(this);
}

IterDomain::IterDomain(const IterDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      extent_(ir_cloner->clone(src->extent_)),
      parallel_type_(src->parallel_type_),
      iter_type_(src->iter_type_),
      is_rfactor_domain_(src->is_rfactor_domain_) {}

bool IterDomain::sameAs(const IterDomain* const other) const {
  if (other == this)
    return true;

  bool is_same = isReduction() == other->isReduction() &&
      getParallelType() == other->getParallelType();
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
      outer->getParallelType() == inner->getParallelType(),
      "Merging IterDomains requires that their parallel types match.");

  Val* merged_id_size = mul(outer->extent(), inner->extent());

  IterType itype = outer->getIterType();

  if (outer->isBroadcast() && inner->isBroadcast()) {
    if (outer->getIterType() == IterType::BroadcastWithStride ||
        inner->getIterType() == IterType::BroadcastWithStride) {
      itype = IterType::BroadcastWithStride;
    } else {
      itype = IterType::BroadcastWithoutStride;
    }
  } else if (outer->isBroadcast() || inner->isBroadcast()) {
    itype = IterType::Iteration;
  }

  IterDomain* merged_id = new IterDomain(
      new Int(0),
      merged_id_size->as<Int>(),
      outer->getParallelType(),
      itype,
      outer->isRFactorProduct() || inner->isRFactorProduct());

  new Merge(merged_id, outer, inner);

  return merged_id;
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor) {
  TORCH_CHECK(
      in->start()->isZeroInt(),
      "Splitting IterDomains with starting values that aren't 0 is not supported at this time.");

  if (in->getParallelType() != ParallelType::Serial)
    TORCH_CHECK(
        false,
        "Splitting an axis of non-Serial iteration is not supported at this time."
        " Parallelization strategy must be set after calling split.");

  TORCH_CHECK(factor->isAnInt(), "Cannot split by non-integer value ", factor);

  if (factor->getValType() == ValType::Scalar) {
    TORCH_CHECK(
        factor->isConstScalar() ||
            FusionGuard::getCurFusion()->hasInput(factor),
        factor,
        " is not a constant nor an input. It must be one or the other to be used in a split.",
        " If you want a symbolic split based on a thread dimension please use IterDomain::split(IterDomain*, ParallelType);");
  } else if (factor->getValType() == ValType::NamedScalar) {
    TORCH_CHECK(
        factor->as<NamedScalar>()->getParallelDim() != c10::nullopt,
        "Splitting a dimension by a named scalar is only supported on block or grid dimensions but received ",
        factor);
  }

  // outer loop size
  Val* vo = ceilDiv(in->extent(), factor);

  // outer loop IterDomain
  IterDomain* ido = new IterDomain(
      new Int(0),
      vo->as<Int>(),
      in->getParallelType(),
      in->getIterType(),
      in->isRFactorProduct());

  // inner loop IterDomain
  IterDomain* idi = new IterDomain(
      new Int(0),
      factor,
      in->getParallelType(),
      in->getIterType(),
      in->isRFactorProduct());

  new Split(ido, idi, in, factor);
  return {ido, idi};
}

// TODO(kir): review if this is still needed in the Fusion IR
Val* IterDomain::extent() const {
  if (isThread()) {
    if (extent_->getValType() == ValType::Scalar)
      if (extent_->as<Int>()->isConst())
        return extent_;

    return NamedScalar::getParallelDim(getParallelType());
  }
  return extent_;
}

namespace {

class RejectMultipleGridReductions : public IterVisitor {
 public:
  static void analyze(Fusion* fusion) {
    RejectMultipleGridReductions multi_grid;
    multi_grid.traverse(fusion, true);
  }

 private:
  void handle(ReductionOp* rop) override {
    TensorView* out = dynamic_cast<TensorView*>(rop->out());
    // Filter out non-related ReductionOp
    if (out == nullptr) {
      return;
    }
    if (!out->domain()->hasGridReduction()) {
      return;
    }
    // rop is a grid reduction. It's an error if we have multiple grid
    // reductions.
    TORCH_CHECK(
        grid_reduction_op_ == nullptr,
        "Multiple grid reductions in a fusion is not supported:\n",
        grid_reduction_op_,
        rop);
    grid_reduction_op_ = rop;
  }

 private:
  ReductionOp* grid_reduction_op_ = nullptr;
};

} // namespace

void IterDomain::parallelize(ParallelType t) {
  parallel_type_ = t;

  TORCH_CHECK(t != ParallelType::Vectorize, "Vectorization not yet supported.");

  if (t == ParallelType::Unroll) {
    TORCH_CHECK(
        start()->isZeroInt() && extent()->isConstScalar(),
        "Unrolling only supported with start = 0 and extent as a const int, but got ",
        "a start of ",
        start(),
        " and extent ",
        extent(),
        " .");
  }

  if (isReduction() && isParallelTypeBlockDim(t)) {
    RejectMultipleGridReductions::analyze(fusion_);
  }
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> domain,
    std::vector<bool> contiguity)
    : Val(ValType::TensorDomain),
      root_domain_(std::move(domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  domain_ = root_domain_;
  resetDomains();
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> domain,
    std::vector<bool> contiguity)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(root_domain)),
      domain_(std::move(domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  std::vector<Val*> domain_vals(domain_.begin(), domain_.end());
  auto inps = IterVisitor::getInputsTo(domain_vals);

  // Validate that the root domain consists of all inputs to domain
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

  name_ = fusion_->registerVal(this);
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> rfactor_domain,
    std::vector<IterDomain*> domain,
    std::vector<bool> contiguity)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(root_domain)),
      domain_(std::move(domain)),
      rfactor_domain_(std::move(rfactor_domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  auto inps = IterVisitor::getInputsTo(
      std::vector<Val*>(domain_.begin(), domain_.end()));

  // Validate that the root domain consists of all inputs to domain
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
  name_ = fusion_->registerVal(this);
}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      domain_(ir_cloner->clone(src->domain_)),
      no_bcast_domain_(ir_cloner->clone(src->no_bcast_domain_)),
      no_reduction_domain_(ir_cloner->clone(src->no_reduction_domain_)),
      rfactor_domain_(ir_cloner->clone(src->rfactor_domain_)),
      contiguity_(src->contiguity()),
      has_reduction_(src->has_reduction_) {}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Should not be necessary to
  // check no_bcast_domain_ and no_reduction_domain_ as they are just
  // derived from domain_.
  return root_domain_ == other.root_domain_ && domain_ == other.domain_ &&
      rfactor_domain_ == other.rfactor_domain_ &&
      contiguity_ == other.contiguity_;
}

bool TensorDomain::sameAs(const TensorDomain* const other) const {
  if (nDims() != other->nDims())
    return false;
  if (getRootDomain().size() != other->getRootDomain().size())
    return false;
  if (getRFactorDomain().size() != other->getRFactorDomain().size())
    return false;

  for (size_t i = 0; i < nDims(); i++)
    if (!(axis(i)->sameAs(other->axis(i))))
      return false;

  for (size_t i = 0; i < getRootDomain().size(); i++)
    if (!(getRootDomain()[i]->sameAs(other->getRootDomain()[i])))
      return false;

  for (size_t i = 0; i < getRFactorDomain().size(); i++)
    if (!(getRFactorDomain()[i]->sameAs(other->getRFactorDomain()[i])))
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
  return has_reduction_;
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

c10::optional<unsigned int> TensorDomain::getReductionAxis() const {
  auto it = std::find_if(domain_.begin(), domain_.end(), [](const auto& id) {
    return id->isReduction();
  });
  if (it == domain_.end()) {
    return c10::optional<unsigned int>();
  } else {
    return c10::optional<unsigned int>(std::distance(domain_.begin(), it));
  }
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to access an axis in a 0-dim domain");
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
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to find an axis in a 0-dim domain");
  size_t i = 0;
  while (i < domain_.size()) {
    if (domain_[i] == id)
      return i;
    i++;
  }
  TORCH_CHECK(false, "Provided id is not part of this domain.");
}

void TensorDomain::split(int axis_, Val* factor) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim domain");
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
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim domain");
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
  TORCH_INTERNAL_ASSERT(
      !(nDims() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim domain");
  domain_ = orderedAs(domain_, old2new_);
  resetDomains();
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !(dom.size() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim domain");

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

bool TensorDomain::hasNontrivialReduction(const std::vector<IterDomain*>& td) {
  for (auto id : td) {
    if (id->isReduction() && !id->isTrivialReduction()) {
      return true;
    }
  }
  return false;
}

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int>& axes_) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim domain");

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

namespace {

//! Concretize broadcast axes, i.e. identifying a non-broadcast
//! IterDomain that the broadcast IterDomain can map to.
//!
//! This traversal processes root domains only, concretization works by
//! inspecting pointwise ops, e.g. : T2 [i0,i1] = T1[i0,B0] + T0[i0,i1]
//! will concretize axis B0 to i1
//!
class ConcretizeDomain : private BackwardVisitor {
 public:
  //! Traverses the graph backward from outputs
  //! to identify all concretizing opportunities
  //!
  explicit ConcretizeDomain(Fusion* fusion) {
    traverseFrom(fusion, fusion->outputs(), false);
  }

  //! API call to run the concretize pass and return the
  //! axis that bcast_dom concretizes to
  //!
  static const IterDomain* getConcreteDomain(IterDomain* bcast_dom) {
    ConcretizeDomain cd(bcast_dom->fusion());

    // Remove this assertion once we support broadcast on output
    TORCH_INTERNAL_ASSERT(cd.canConcretize(bcast_dom));
    return cd.concretized(bcast_dom);
  }

  // Returns true if either id is not a broadcast or
  // the traversal has found a concretized axis for id
  bool canConcretize(IterDomain* id) const {
    return !id->isBroadcast() || bcast_domain_map_.count(id);
  }

  // Returns the concretized id recorded from traversal
  IterDomain* concretized(IterDomain* id) const {
    TORCH_INTERNAL_ASSERT(canConcretize(id));
    if (!id->isBroadcast()) {
      return id;
    }
    return bcast_domain_map_.at(id);
  }

 private:
  // Utility to inspect a pointwise operator and
  // record concretize opportunities
  void concretizePwOp(Expr* e);

  // Utility to record new concretize opportunity
  void concretizeTo(IterDomain* id, IterDomain* To) {
    TORCH_INTERNAL_ASSERT(id->isBroadcast() && !To->isBroadcast());
    bcast_domain_map_[id] = concretized(To);
  }

  using BackwardVisitor::handle;

  void handle(ReductionOp* rop) override {
    concretizePwOp(rop);
  }

  void handle(UnaryOp* uop) override {
    concretizePwOp(uop);
  }

  void handle(BinaryOp* bop) override {
    concretizePwOp(bop);
  }

  void handle(TernaryOp* top) override {
    concretizePwOp(top);
  };

 private:
  using MapType = std::unordered_map<IterDomain*, IterDomain*>;
  MapType bcast_domain_map_;
};

void ConcretizeDomain::concretizePwOp(Expr* e) {
  if (e->output(0)->getValType() != ValType::TensorView) {
    return;
  }

  TORCH_INTERNAL_ASSERT(e->outputs().size() == 1);
  TensorView* tv = e->output(0)->as<TensorView>();

  std::vector<IterDomain*> io = tv->getRootDomain();

  for (auto* i : ir_utils::filterByType<TensorView>(e->inputs())) {
    std::vector<IterDomain*> ii =
        TensorDomain::noReductions(i->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(ii.size() == io.size());

    for (size_t it = 0; it < ii.size(); it++) {
      if (!canConcretize(io[it]))
        continue;

      if (!canConcretize(ii[it]))
        concretizeTo(ii[it], concretized(io[it]));
    }
  }
}

} // namespace

// API call to return the concretized axis of a broadcast axis
const IterDomain* IterDomain::concretizeDomain(IterDomain* bcast_dom) {
  return ConcretizeDomain::getConcreteDomain(bcast_dom);
}

Split::Split(IterDomain* outer, IterDomain* inner, IterDomain* in, Val* factor)
    : Expr(ExprType::Split),
      outer_{outer},
      inner_{inner},
      in_{in},
      factor_{factor} {
  TORCH_INTERNAL_ASSERT(
      factor_->isAnInt(),
      "Attempted to create a Split node with a non-integer factor.");
  addOutput(outer);
  addOutput(inner);
  addInput(in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Split::Split(const Split* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)),
      in_(ir_cloner->clone(src->in_)),
      factor_(ir_cloner->clone(src->factor_)) {}

bool Split::sameAs(const Split* const other) const {
  return (
      outer()->sameAs(other->outer()) && inner()->sameAs(other->inner()) &&
      in()->sameAs(other->in()) && factor()->sameAs(other->factor()));
}

Merge::Merge(IterDomain* out, IterDomain* outer, IterDomain* inner)
    : Expr(ExprType::Merge), out_{out}, outer_{outer}, inner_{inner} {
  addOutput(out);
  addInput(outer);
  addInput(inner);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Merge::Merge(const Merge* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)) {}

bool Merge::sameAs(const Merge* const other) const {
  return (
      out()->sameAs(other->out()) && outer()->sameAs(other->outer()) &&
      inner()->sameAs(other->inner()));
}

NamedScalar::NamedScalar(const NamedScalar* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), name_(src->name_) {}

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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
