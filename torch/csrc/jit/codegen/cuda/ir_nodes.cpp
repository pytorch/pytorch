#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>

#include <c10/util/irange.h>

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
  void handle(const Bool* b) final {
    same_ = v1_->as<Bool>()->sameAs(v2_->as<Bool>());
  }

  void handle(const Double* d) final {
    same_ = v1_->as<Double>()->sameAs(v2_->as<Double>());
  }

  void handle(const Int* i) final {
    same_ = v1_->as<Int>()->sameAs(v2_->as<Int>());
  }

  void handle(const NamedScalar* ns) final {
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

Bool::Bool(IrBuilderPasskey passkey)
    : Val(passkey, ValType::Scalar, DataType::Bool),
      maybe_value_{c10::nullopt} {}

Bool::Bool(IrBuilderPasskey passkey, bool value)
    : Val(passkey, ValType::Scalar, DataType::Bool), maybe_value_{value} {}

Bool::Bool(IrBuilderPasskey passkey, c10::optional<bool> value)
    : Val(passkey, ValType::Scalar, DataType::Bool), maybe_value_{value} {}

Bool::Bool(const Bool* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Bool::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Bool>()) {
    return false;
  }
  const auto other_bool = other->as<Bool>();
  if (isConst() && other_bool->isConst()) {
    return *value() == *(other_bool->value());
  }
  return false;
}

Double::Double(IrBuilderPasskey passkey)
    : Val(passkey, ValType::Scalar, DataType::Double),
      maybe_value_{c10::nullopt} {}

Double::Double(IrBuilderPasskey passkey, ScalarType value)
    : Val(passkey, ValType::Scalar, DataType::Double), maybe_value_{value} {}

Double::Double(IrBuilderPasskey passkey, c10::optional<ScalarType> value)
    : Val(passkey, ValType::Scalar, DataType::Double), maybe_value_{value} {}

Double::Double(const Double* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Double::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Double>()) {
    return false;
  }
  const auto other_double = other->as<Double>();
  if (isConst() && other_double->isConst())
    return *value() == *(other_double->value());
  return false;
}

Int::Int(IrBuilderPasskey passkey)
    : Val(passkey, ValType::Scalar, DataType::Int),
      maybe_value_{c10::nullopt} {}

Int::Int(IrBuilderPasskey passkey, ScalarType value)
    : Val(passkey, ValType::Scalar, DataType::Int), maybe_value_{value} {}

Int::Int(IrBuilderPasskey passkey, c10::optional<ScalarType> value)
    : Val(passkey, ValType::Scalar, DataType::Int), maybe_value_{value} {}

Int::Int(const Int* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Int::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Int>()) {
    return false;
  }
  const auto other_int = other->as<Int>();
  if (isConst() && other_int->isConst()) {
    return *value() == *(other_int->value());
  }
  return false;
}

ComplexDouble::ComplexDouble(IrBuilderPasskey passkey)
    : Val(passkey, ValType::Scalar, DataType::ComplexDouble),
      maybe_value_{c10::nullopt} {}

ComplexDouble::ComplexDouble(IrBuilderPasskey passkey, ScalarType value)
    : Val(passkey, ValType::Scalar, DataType::ComplexDouble),
      maybe_value_{value} {}

ComplexDouble::ComplexDouble(
    IrBuilderPasskey passkey,
    c10::optional<ScalarType> value)
    : Val(passkey, ValType::Scalar, DataType::ComplexDouble),
      maybe_value_{value} {}

ComplexDouble::ComplexDouble(const ComplexDouble* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool ComplexDouble::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<ComplexDouble>()) {
    return false;
  }
  const auto other_complex = other->as<ComplexDouble>();
  if (isConst() && other_complex->isConst())
    return *value() == *(other_complex->value());
  return false;
}

ARangeOp::ARangeOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* start,
    Val* end,
    Val* step,
    Val* linear_index)
    : Expr(passkey, ExprType::ARangeOp),
      start_(start),
      end_(end),
      step_(step),
      linear_index_(linear_index) {
  addInput(start);
  addInput(end);
  addInput(step);
  addOutput(out);
}

ARangeOp::ARangeOp(const ARangeOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      end_(ir_cloner->clone(src->end_)),
      step_(ir_cloner->clone(src->step_)),
      linear_index_(ir_cloner->clone(src->linear_index_)) {}

bool ARangeOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<ARangeOp>()) {
    return false;
  }
  const auto other_op = other->as<ARangeOp>();
  if (!start_->sameAs(other_op->start_)) {
    return false;
  }
  if (!end_->sameAs(other_op->end_)) {
    return false;
  }
  if (!step_->sameAs(other_op->step_)) {
    return false;
  }
  if ((linear_index_ == nullptr) != (other_op->linear_index_ == nullptr)) {
    return false;
  }
  if ((linear_index_ != nullptr) &&
      !linear_index_->sameAs(other_op->linear_index_)) {
    return false;
  }
  return Expr::sameAs(other);
}

UnaryOp::UnaryOp(
    IrBuilderPasskey passkey,
    UnaryOpType type,
    Val* out,
    Val* in,
    int rng_offset)
    : Expr(passkey, ExprType::UnaryOp),
      unary_op_type_{type},
      out_{out},
      in_{in} {
  addOutput(out);
  addInput(in);
}

UnaryOp::UnaryOp(const UnaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      unary_op_type_(src->unary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool UnaryOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<UnaryOp>()) {
    return false;
  }
  const auto other_op = other->as<UnaryOp>();
  if (getUnaryOpType() != other_op->getUnaryOpType())
    return false;
  return Expr::sameAs(other);
}

BinaryOp::BinaryOp(
    IrBuilderPasskey passkey,
    BinaryOpType type,
    Val* out,
    Val* lhs,
    Val* rhs)
    : Expr(passkey, ExprType::BinaryOp),
      binary_op_type_{type},
      out_{out},
      lhs_{lhs},
      rhs_{rhs} {
  addOutput(out);
  addInput(lhs);
  addInput(rhs);
}

BinaryOp::BinaryOp(const BinaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      binary_op_type_(src->binary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      lhs_(ir_cloner->clone(src->lhs_)),
      rhs_(ir_cloner->clone(src->rhs_)) {}

bool BinaryOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<BinaryOp>()) {
    return false;
  }
  const auto other_op = other->as<BinaryOp>();
  if (getBinaryOpType() != other_op->getBinaryOpType())
    return false;
  return Expr::sameAs(other);
}

TernaryOp::TernaryOp(
    IrBuilderPasskey passkey,
    TernaryOpType type,
    Val* out,
    Val* in1,
    Val* in2,
    Val* in3)
    : Expr(passkey, ExprType::TernaryOp),
      ternary_op_type_{type},
      out_{out},
      in1_{in1},
      in2_{in2},
      in3_{in3} {
  addOutput(out);
  addInput(in1);
  addInput(in2);
  addInput(in3);
}

TernaryOp::TernaryOp(const TernaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      ternary_op_type_(src->ternary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in1_(ir_cloner->clone(src->in1_)),
      in2_(ir_cloner->clone(src->in2_)),
      in3_(ir_cloner->clone(src->in3_)) {}

bool TernaryOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<TernaryOp>()) {
    return false;
  }
  const auto other_op = other->as<TernaryOp>();
  if (getTernaryOpType() != other_op->getTernaryOpType())
    return false;
  return Expr::sameAs(other);
}

RNGOp::RNGOp(
    IrBuilderPasskey passkey,
    RNGOpType type,
    Val* out,
    int rng_offset,
    Val* philox_index)
    : Expr(passkey, ExprType::RNGOp),
      rng_op_type_(type),
      rng_offset_(rng_offset),
      philox_index_(philox_index) {
  if (out->isA<TensorView>()) {
    for (auto id : out->as<TensorView>()->getRootDomain()) {
      addInput(id->extent());
    }
  }
  addOutput(out);
}

RNGOp::RNGOp(const RNGOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      rng_op_type_(src->rng_op_type_),
      rng_offset_(src->rng_offset_),
      philox_index_(ir_cloner->clone(src->philox_index_)) {}

bool RNGOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<RNGOp>()) {
    return false;
  }
  const auto other_op = other->as<RNGOp>();
  if (getRNGOpType() != other_op->getRNGOpType()) {
    return false;
  }
  if (getRNGOffset() != other_op->getRNGOffset()) {
    return false;
  }
  if ((philox_index_ == nullptr) != (other_op->philox_index_ == nullptr)) {
    return false;
  }
  if ((philox_index_ != nullptr) &&
      !philox_index_->sameAs(other_op->philox_index_)) {
    return false;
  }
  return Expr::sameAs(other);
}

BroadcastOp::BroadcastOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<bool> is_broadcast_dims)
    : Expr(passkey, ExprType::BroadcastOp),
      out_(out),
      in_(in),
      is_broadcast_dims_(std::move(is_broadcast_dims)) {
  // clang-tidy complains about out_ that it may be null.
  TORCH_INTERNAL_ASSERT(out_ != nullptr);
  TORCH_INTERNAL_ASSERT(in_ != nullptr);

  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      (out_type == ValType::TensorView && in_type == ValType::TensorView) ||
          (out_type == ValType::TensorIndex && in_type == ValType::TensorIndex),
      "Cannot braodcast a non-tensor object.");

  addOutput(out);
  addInput(in);

  if (!out->isA<TensorView>() || !in->isA<TensorView>()) {
    return;
  }

  passkey.ir_container_->registerExpr(exprPasskey(), this);

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
          id->isReduction() || id->isStride(),
          "Invalid broadcast op: ",
          id,
          ". Non-reduction input dim does't match to output.");
    }
  }

  std::unordered_set<IterDomain*> c_mapped;
  for (auto pair_entry : root_p2c) {
    c_mapped.insert(pair_entry.second);
  }

  for (const auto i : c10::irange(c_root.size())) {
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

bool BroadcastOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<BroadcastOp>()) {
    return false;
  }
  const auto other_op = other->as<BroadcastOp>();
  if (getBroadcastDimFlags() != other_op->getBroadcastDimFlags()) {
    return false;
  }
  return Expr::sameAs(other);
}

ReductionOp::ReductionOp(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    bool is_allreduce,
    ExprType expr_type)
    : Expr(passkey, expr_type),
      reduction_op_type_(reduction_op_type),
      init_(init),
      out_(out),
      in_(in),
      is_allreduce_(is_allreduce) {
  TORCH_CHECK(
      out->getValType().value() == ValType::TensorView ||
      out->getValType().value() == ValType::TensorIndex);

  TORCH_INTERNAL_ASSERT(
      (in->getValType() == ValType::TensorView &&
       out->getValType() == ValType::TensorView) ||
          (in->getValType() == ValType::TensorIndex &&
           out->getValType() == ValType::TensorIndex),
      "Reduction operation was created that does not have tensor inputs and outputs.");

  if (in->isA<TensorView>()) {
    TORCH_INTERNAL_ASSERT(
        TensorDomain::noReductions(
            in->as<TensorView>()->getMaybeRFactorDomain())
                .size() == out->as<TensorView>()->getRootDomain().size(),
        "Reduction operation created with mismatched domains.");
  }
  TORCH_INTERNAL_ASSERT(
      init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't a constant.");

  addOutput(out);
  addInput(in);
}

GroupedReductionOp::GroupedReductionOp(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    bool is_fused,
    ExprType expr_type)
    : Expr(passkey, expr_type),
      reduction_op_types_(std::move(reduction_op_types)),
      init_vals_(std::move(init_vals)),
      is_allreduce_(is_fused) {
  for (auto out : outputs) {
    addOutput(out);
  }

  for (auto in : inputs) {
    addInput(in);
  }
}

GroupedReductionOp::GroupedReductionOp(
    const GroupedReductionOp* src,
    IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_types_(src->reduction_op_types_),
      init_vals_(ir_cloner->clone(src->init_vals_)),
      is_allreduce_(src->is_allreduce_) {}

int GroupedReductionOp::getExprIndexOfOutput(Val* output_val) const {
  auto it = std::find(outputs().begin(), outputs().end(), output_val);
  if (it != outputs().end()) {
    return std::distance(outputs().begin(), it);
  }

  TORCH_INTERNAL_ASSERT(
      false, "Not an output, ", output_val->toString(), ", of ", toString());
}

bool GroupedReductionOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }

  auto grouped_rop = dynamic_cast<const GroupedReductionOp*>(other);
  if (grouped_rop == nullptr) {
    return false;
  }

  if (!Expr::sameAs(other) ||
      getReductionOpTypes() != grouped_rop->getReductionOpTypes()) {
    return false;
  }

  for (const auto i : c10::irange(numExprs())) {
    if (!initVal(i)->sameAs(grouped_rop->initVal(i))) {
      return false;
    }
  }

  return true;
}

WelfordOp::WelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    bool is_fused)
    : Expr(passkey, ExprType::WelfordOp),
      output_(output),
      input_(input),
      init_(init),
      is_allreduce_(is_fused) {
  // Previously, nullptr was accepted and implicitly replaced by
  // default values. Looks like we always pass some non-null values,
  // so removed the implicit default behavior for code simplicity.
  TORCH_INTERNAL_ASSERT(output.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(output.var() != nullptr);
  TORCH_INTERNAL_ASSERT(output.N() != nullptr);
  TORCH_INTERNAL_ASSERT(init.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(init.var() != nullptr);
  TORCH_INTERNAL_ASSERT(init.N() != nullptr);
  TORCH_INTERNAL_ASSERT(input.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(input.var() != nullptr);
  TORCH_INTERNAL_ASSERT(input.N() != nullptr);

  // Check output type
  TORCH_INTERNAL_ASSERT(
      output.avg()->getValType().value() == ValType::TensorView ||
      output.avg()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(
      output.var()->getValType().value() == ValType::TensorView ||
      output.var()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(
      output.N()->getValType().value() == ValType::TensorView ||
      output.N()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(isIntegralType(output.N()->dtype()));

  // check initial value
  TORCH_INTERNAL_ASSERT(init.N()->getValType().value() == ValType::Scalar);
  TORCH_INTERNAL_ASSERT(isIntegralType(init.N()->dtype()));
  if (!init.N()->isZeroInt()) {
    // when initial count is zero, no initial variance or average is needed
    // initial value with a count of 1 is un-common enough that I'll push
    // the responsibility of creating all-zero var tensors to the user
    TORCH_INTERNAL_ASSERT(
        init_.avg()->getValType().value() == ValType::TensorView ||
        init_.avg()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        init_.var()->getValType().value() == ValType::TensorView ||
            init_.var()->getValType().value() == ValType::TensorIndex,
        "Invalid initial var: ",
        init_.var()->toString());
  }

  // check input
  TORCH_INTERNAL_ASSERT(
      input_.avg()->getValType().value() == ValType::TensorView ||
          input_.avg()->getValType().value() == ValType::TensorIndex,
      input_.avg()->getValType().value());
  TORCH_INTERNAL_ASSERT(
      input_.N()->getValType().value() == ValType::Scalar ||
      input_.N()->getValType().value() == ValType::TensorView ||
      input_.N()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(isIntegralType(input_.N()->dtype()));
  if (!input_.N()->isOneInt()) {
    // when input is only one value, only the value is required through avg
    // input the var part is implicitly 0 and codegen will handle that.
    TORCH_INTERNAL_ASSERT(
        input_.var()->getValType().value() == ValType::TensorView ||
        input_.var()->getValType().value() == ValType::TensorIndex);
  } else {
    TORCH_INTERNAL_ASSERT(
        input_.var() == nullptr || input_.var()->isZeroInt(),
        "Invalid var input, which must be either nullptr or scalar zero when the N input is one.");
  }

  addOutput(output_.avg());
  addOutput(output_.var());
  addOutput(output_.N());

  addInput(input_.avg());
  addInput(input_.var());
  addInput(input_.N());
}

c10::optional<WelfordTriplet::ValName> WelfordTriplet::getNameOf(
    Val* val) const {
  auto it = std::find(begin(), end(), val);
  if (it != end()) {
    return indexToValName(std::distance(begin(), it));
  }

  return c10::optional<WelfordTriplet::ValName>();
}

bool WelfordTriplet::sameAs(const WelfordTriplet& other) const {
  return this == &other ||
      (avg()->sameAs(other.avg()) && var()->sameAs(other.var()) &&
       N()->sameAs(other.N()));
}

WelfordTriplet WelfordTriplet::clone(IrCloner* ir_cloner) const {
  return transform([&](const Val* val) { return ir_cloner->clone<Val>(val); });
}

std::vector<WelfordTriplet> WelfordTriplet::clone(
    const std::vector<WelfordTriplet>& src,
    IrCloner* ir_cloner) {
  std::vector<WelfordTriplet> cloned;
  for (const auto& triplet : src) {
    cloned.emplace_back(triplet.clone(ir_cloner));
  }
  return cloned;
}

WelfordOp::WelfordOp(
    IrBuilderPasskey passkey,
    Val* out_avg,
    Val* out_var,
    Val* out_N,
    Val* in_avg,
    Val* in_var,
    Val* in_N,
    Val* init_avg,
    Val* init_var,
    Val* init_N,
    bool is_fused)
    : WelfordOp(
          passkey,
          WelfordTriplet(out_avg, out_var, out_N),
          WelfordTriplet(in_avg, in_var, in_N),
          WelfordTriplet(init_avg, init_var, init_N),
          is_fused) {}

WelfordOp::WelfordOp(const WelfordOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      output_(src->output_.clone(ir_cloner)),
      input_(src->input_.clone(ir_cloner)),
      init_(src->init_.clone(ir_cloner)),
      is_allreduce_(src->is_allreduce_) {}

Val* WelfordOp::getInitValOfOutput(Val* output_val) const {
  auto val_name = output().getNameOf(output_val);

  TORCH_INTERNAL_ASSERT(
      val_name.has_value(),
      "Not an output val ",
      output_val->toString(),
      " of ",
      toString());

  return init().get(*val_name);
}

bool WelfordOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (auto other_wop = dynamic_cast<const WelfordOp*>(other)) {
    return input_.sameAs(other_wop->input_) && init_.sameAs(other_wop->init_);
  }
  return false;
}

std::vector<Val*> WelfordOp::getInitVals() const {
  std::vector<Val*> init_vals({init_.avg(), init_.var(), init_.N()});
  return init_vals;
}

GroupedWelfordOp::GroupedWelfordOp(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    bool is_allreduce,
    ExprType expr_type)
    : Expr(passkey, expr_type),
      output_vals_(std::move(output_vals)),
      input_vals_(std::move(input_vals)),
      init_vals_(std::move(init_vals)),
      is_allreduce_(is_allreduce) {
  const auto num_grouped_ops = output_vals_.size();

  TORCH_INTERNAL_ASSERT(
      input_vals_.size() == num_grouped_ops,
      "Invalid number of input arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      input_vals_.size());
  TORCH_INTERNAL_ASSERT(
      init_vals_.size() == num_grouped_ops,
      "Invalid number of N arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      init_vals_.size());

  for (const auto i : c10::irange(num_grouped_ops)) {
    // Check output type
    TORCH_INTERNAL_ASSERT(
        output_vals_[i].avg()->getValType().value() == ValType::TensorView ||
        output_vals_[i].avg()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        output_vals_[i].var()->getValType().value() == ValType::TensorView ||
        output_vals_[i].var()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        output_vals_[i].N()->getValType().value() == ValType::TensorView ||
        output_vals_[i].N()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(isIntegralType(output_vals_[i].N()->dtype()));

    // check initial value
    auto init_avg = init_vals_[i].avg();
    auto init_var = init_vals_[i].var();
    auto init_N = init_vals_[i].N();
    TORCH_INTERNAL_ASSERT(
        init_avg != nullptr && init_var != nullptr && init_N != nullptr,
        "nullptr init vals are not allowed");
    TORCH_INTERNAL_ASSERT(init_N->getValType().value() == ValType::Scalar);
    TORCH_INTERNAL_ASSERT(isIntegralType(init_N->dtype()));
    TORCH_INTERNAL_ASSERT(
        init_avg->getValType().value() == ValType::TensorView ||
            init_avg->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_avg->getValType().value() == ValType::Scalar),
        "Initial avg must be a tensor or, can be a scalar if initial N is zero.",
        " Initial avg: ",
        init_avg->toString(),
        ". Initial N: ",
        init_N->toString());
    TORCH_INTERNAL_ASSERT(
        init_var->getValType().value() == ValType::TensorView ||
            init_var->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_var->getValType().value() == ValType::Scalar),
        "Initial var must be a tensor or, can be a scalar if initial N is zero: ",
        init_var->toString());

    // check input
    auto in_avg = input_vals_[i].avg();
    auto in_var = input_vals_[i].var();
    auto in_N = input_vals_[i].N();
    TORCH_INTERNAL_ASSERT(
        in_avg != nullptr && in_var != nullptr && in_N != nullptr,
        "nullptr input vals are not allowed");
    TORCH_INTERNAL_ASSERT(
        in_N->getValType().value() == ValType::Scalar ||
        in_N->getValType().value() == ValType::TensorView ||
        in_N->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(isIntegralType(in_N->dtype()));
    TORCH_INTERNAL_ASSERT(
        in_avg->getValType().value() == ValType::TensorView ||
            in_avg->getValType().value() == ValType::TensorIndex,
        "Invalid input avg argument type: ",
        in_avg->getValType().value());

    if (in_N->isOneInt()) {
      // when input is only one value, only the value is required through avg
      // input the var part must be implicitly 0
      TORCH_INTERNAL_ASSERT(
          in_var->isZeroInt(),
          "Invalid var input, which must be scalar zero when the N input is one: ",
          in_var->toString());
    } else {
      TORCH_INTERNAL_ASSERT(
          in_var->getValType().value() == ValType::TensorView ||
              in_var->getValType().value() == ValType::TensorIndex,
          in_var->getValType().value(),
          ", ",
          in_N->toString());
    }
  }

  for (const auto i : c10::irange(num_grouped_ops)) {
    addOutput(output_vals_[i].avg());
    addOutput(output_vals_[i].var());
    addOutput(output_vals_[i].N());
    addInput(input_vals_[i].avg());
    addInput(input_vals_[i].var());
    addInput(input_vals_[i].N());
  }
}

GroupedWelfordOp::GroupedWelfordOp(
    const GroupedWelfordOp* src,
    IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      output_vals_(WelfordTriplet::clone(src->output_vals_, ir_cloner)),
      input_vals_(WelfordTriplet::clone(src->input_vals_, ir_cloner)),
      init_vals_(WelfordTriplet::clone(src->init_vals_, ir_cloner)),
      is_allreduce_(src->is_allreduce_) {}

bool GroupedWelfordOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }

  auto grouped_op = dynamic_cast<const GroupedWelfordOp*>(other);
  if (grouped_op == nullptr) {
    return false;
  }

  if (!Expr::sameAs(other)) {
    return false;
  }

  for (const auto i : c10::irange(numExprs())) {
    if (!initAvg(i)->sameAs(grouped_op->initAvg(i)) ||
        !initVar(i)->sameAs(grouped_op->initVar(i)) ||
        !initN(i)->sameAs(grouped_op->initN(i))) {
      return false;
    }
  }

  return true;
}

int GroupedWelfordOp::getExprIndexOfOutput(Val* output_val) const {
  for (const auto expr_idx : c10::irange(numExprs())) {
    if (outputVals().at(expr_idx).getNameOf(output_val).has_value()) {
      return expr_idx;
    }
  }

  TORCH_INTERNAL_ASSERT(
      false, "Not an output, ", output_val->toString(), ", of ", toString());
}

Val* GroupedWelfordOp::getInitValOfOutput(Val* output_val) const {
  auto expr_index = getExprIndexOfOutput(output_val);

  auto val_name = outputVals().at(expr_index).getNameOf(output_val).value();

  return initVals().at(expr_index).get(val_name);
}

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init)
    : Expr(passkey, ExprType::MmaOp),
      out_(out),
      in_a_(in_a),
      in_b_(in_b),
      init_(init) {
  // Check output type
  TORCH_INTERNAL_ASSERT(
      out->getValType().value() == ValType::TensorView ||
      out->getValType().value() == ValType::TensorIndex);

  TORCH_INTERNAL_ASSERT(
      in_a->getValType().value() == ValType::TensorView ||
          in_a->getValType().value() == ValType::TensorIndex,
      in_a->getValType().value());

  TORCH_INTERNAL_ASSERT(
      in_b->getValType().value() == ValType::TensorView ||
          in_b->getValType().value() == ValType::TensorIndex,
      in_b->getValType().value());

  addOutput(out);
  addInput(in_a);
  addInput(in_b);
}

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init,
    OptionsInMma options)
    : MmaOp(passkey, out, in_a, in_b, init) {
  options_ = options;
}

MmaOp::MmaOp(const MmaOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_a_(ir_cloner->clone(src->in_a_)),
      in_b_(ir_cloner->clone(src->in_b_)),
      init_(ir_cloner->clone(src->init_)),
      options_(src->options_) {}

bool MmaOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (auto other_mma = dynamic_cast<const MmaOp*>(other)) {
    return out_->sameAs(other_mma->out_) && in_a_->sameAs(other_mma->in_a_) &&
        in_b_->sameAs(other_mma->in_b_) && init_->sameAs(other_mma->init_) &&
        options_ == other_mma->options_;
  }
  return false;
}

ReductionOp::ReductionOp(const ReductionOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_type_(src->reduction_op_type_),
      init_(ir_cloner->clone(src->init_)),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      is_allreduce_(src->is_allreduce_) {}

bool ReductionOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<ReductionOp>()) {
    return false;
  }
  const auto other_op = other->as<ReductionOp>();
  // Note that init is not part of input vals, so it must be checked separately.
  return (
      Expr::sameAs(other) &&
      getReductionOpType() == other_op->getReductionOpType() &&
      init()->sameAs(other_op->init()));
}

TransposeOp::TransposeOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* in,
    std::vector<int64_t> new2old)
    : Expr(passkey, ExprType::TransposeOp),
      out_(out),
      in_(in),
      new2old_(std::move(new2old)) {
  // Sanity check of the input parameters. Maybe not necessary as they
  // should be checked at function transpose.

  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(in->getMaybeRFactorDomain()).size() ==
      out->getMaybeRFactorDomain().size());

  TORCH_INTERNAL_ASSERT(new2old_.size() == out->getMaybeRFactorDomain().size());

  // Make sure the entries of new2old are unique and range from 0 to
  // N-1, where N == new2old.size().
  std::set<int64_t> old_positions(new2old_.begin(), new2old_.end());
  TORCH_INTERNAL_ASSERT(old_positions.size() == new2old_.size());
  // old_positions is sorted, so the first entry must be 0.
  TORCH_INTERNAL_ASSERT(
      *(old_positions.begin()) == 0,
      "Invalid new2old vector detected: ",
      new2old_);
  // The last entry must be N-1, since old_positions is sorted, starts
  // with 0, and its length is N.
  TORCH_INTERNAL_ASSERT(
      *(old_positions.rbegin()) == (int)(new2old_.size() - 1),
      "Invalid new2old vector detected: ",
      new2old_);

  addOutput(out);
  addInput(in);
}

TransposeOp::TransposeOp(const TransposeOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      new2old_(src->new2old_) {}

std::vector<int64_t> TransposeOp::old2new() const {
  std::vector<int64_t> old2new(new2old_.size());
  for (auto new_axis : c10::irange(new2old_.size())) {
    auto old_axis = new2old_.at(new_axis);
    old2new[old_axis] = new_axis;
  }
  return old2new;
}

ExpandOp::ExpandOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* in,
    std::vector<Val*> _expanded_extents)
    : Expr(passkey, ExprType::ExpandOp),
      out_(out),
      in_(in),
      expanded_extents_(std::move(_expanded_extents)) {
  addOutput(out);
  addInput(in);
  for (auto expanded_extent : expanded_extents_) {
    TORCH_INTERNAL_ASSERT(expanded_extent != nullptr);
    TORCH_INTERNAL_ASSERT(
        expanded_extent->dtype() == DataType::Int,
        "Expanded extents must be of Int type.");
    addInput(expanded_extent);
  }
}

ExpandOp::ExpandOp(const ExpandOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {
  expanded_extents_.reserve(src->expanded_extents_.size());
  for (const auto expanded_extent : src->expanded_extents_) {
    expanded_extents_.push_back(ir_cloner->clone(expanded_extent));
  }
}

ShiftOp::ShiftOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<int> offsets,
    std::vector<int> pad_width)
    : Expr(passkey, ExprType::ShiftOp),
      out_(out),
      in_(in),
      offsets_(std::move(offsets)),
      pad_width_(std::move(pad_width)) {
  // clang-tidy complains about out_ that it may be null.
  TORCH_INTERNAL_ASSERT(out_ != nullptr);
  TORCH_INTERNAL_ASSERT(in_ != nullptr);

  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot shift a non-tensor object.");

  TORCH_INTERNAL_ASSERT(
      offsets_.size() ==
          TensorDomain::noReductions(in_->as<TensorView>()->getRootDomain())
              .size(),
      "Invalid offset vector: ",
      offsets_);

  TORCH_INTERNAL_ASSERT(
      pad_width_.size() ==
          TensorDomain::noReductions(in_->as<TensorView>()->getRootDomain())
              .size(),
      "Invalid padding width vector: ",
      pad_width_);

  addOutput(out);
  addInput(in);
}

ShiftOp::ShiftOp(const ShiftOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      offsets_(src->offsets_),
      pad_width_(src->pad_width_) {}

bool ShiftOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<ShiftOp>()) {
    return false;
  }
  const auto other_op = other->as<ShiftOp>();
  if (offsets() != other_op->offsets()) {
    return false;
  }
  return Expr::sameAs(other);
}

GatherOp::GatherOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<int> window_shape,
    std::vector<std::vector<int>> pad_width)
    : Expr(passkey, ExprType::GatherOp),
      out_(out),
      in_(in),
      window_shape_(std::move(window_shape)),
      pad_width_(std::move(pad_width)) {
  // clang-tidy complains about out_ that it may be null.
  TORCH_INTERNAL_ASSERT(out_ != nullptr);
  TORCH_INTERNAL_ASSERT(in_ != nullptr);

  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot shift a non-tensor object.");

  const auto ndims =
      TensorDomain::noReductions(in_->as<TensorView>()->getRootDomain()).size();

  TORCH_INTERNAL_ASSERT(
      window_shape_.size() == ndims,
      "Invalid window_shape vector: ",
      window_shape_);
  TORCH_INTERNAL_ASSERT(
      pad_width_.size() == ndims, "Invalid pad_width vector: ", pad_width_);

  for (const auto& pad : pad_width_) {
    TORCH_INTERNAL_ASSERT(
        pad.size() == 2, "Padding size for each axis must have two Int vals.");
  }

  addOutput(out);
  addInput(in);
}

GatherOp::GatherOp(const GatherOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      window_shape_(src->window_shape_),
      pad_width_(src->pad_width_) {}

bool GatherOp::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<GatherOp>()) {
    return false;
  }
  const auto other_op = other->as<GatherOp>();
  if (windowShape() != other_op->windowShape() ||
      padWidth() != other_op->padWidth()) {
    return false;
  }
  return Expr::sameAs(other);
}

int GatherOp::gatherAxis(int axis) const {
  if (axis < 0) {
    axis += out()->as<TensorView>()->nDims();
  }
  TORCH_INTERNAL_ASSERT(
      axis >= 0 && axis < (int)windowShape().size(), "Invalid axis: ", axis);
  return int(windowShape().size()) + axis;
}

ViewAsScalar::ViewAsScalar(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    IterDomain* vector_id,
    Val* index)
    : Expr(passkey, ExprType::ViewAsScalar),
      out_(out),
      in_(in),
      vector_id_(vector_id),
      index_(index) {
  addOutput(out);
  addInput(in);
}

ViewAsScalar::ViewAsScalar(const ViewAsScalar* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)),
      vector_id_(ir_cloner->clone(src->vector_id_)),
      index_(ir_cloner->clone(src->index_)) {}

ViewOp::ViewOp(IrBuilderPasskey passkey, TensorView* out, TensorView* in)
    : Expr(passkey, ExprType::ViewOp), out_(out), in_(in) {
  addOutput(out);
  addInput(in);
}

ViewOp::ViewOp(const ViewOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

LoadStoreOp::LoadStoreOp(
    IrBuilderPasskey passkey,
    LoadStoreOpType op_type,
    Val* out,
    Val* in)
    : Expr(passkey, ExprType::LoadStoreOp),
      load_store_type_(op_type),
      out_(out),
      in_(in) {
  addOutput(out);
  addInput(in);
}

LoadStoreOp::LoadStoreOp(const LoadStoreOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      load_store_type_(src->load_store_type_),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

IterDomainBuilder::IterDomainBuilder(Val* _start, Val* _extent)
    : start_(_start), extent_(_extent) {
  TORCH_INTERNAL_ASSERT(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
}

IterDomainBuilder::IterDomainBuilder(const IterDomain* id)
    : start_(id->start()),
      extent_(id->extent()),
      expanded_extent_(
          id->hasExpandedExtent() ? id->expandedExtent() : nullptr),
      stop_offset_(id->stopOffset()),
      parallel_type_(id->getParallelType()),
      iter_type_(id->getIterType()),
      is_rfactor_domain_(id->isRFactorProduct()),
      is_padded_dimension_(id->hasPaddingToMultipleOfWarp()),
      padded_to_size_(id->getMaybeSizeAfterPadding()),
      is_mma_swizzled_(id->isMmaSwizzled()) {}

IterDomainBuilder& IterDomainBuilder::resetSchedulingParams() {
  parallel_type_ = ParallelType::Serial;
  is_rfactor_domain_ = false;
  is_padded_dimension_ = false;
  padded_to_size_ = c10::nullopt;
  is_mma_swizzled_ = false;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::resetRfactor() {
  return is_rfactor_domain(false);
}

IterDomainBuilder& IterDomainBuilder::start(Val* _start) {
  start_ = _start;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::extent(Val* _extent) {
  extent_ = _extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::expanded_extent(Val* _expanded_extent) {
  expanded_extent_ = _expanded_extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::stop_offset(Val* _stop_offset) {
  stop_offset_ = _stop_offset;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::parallel_type(
    ParallelType _parallel_type) {
  parallel_type_ = _parallel_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::iter_type(IterType _iter_type) {
  iter_type_ = _iter_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_rfactor_domain(
    bool _is_rfactor_domain) {
  is_rfactor_domain_ = _is_rfactor_domain;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_padded_dimension(
    bool _is_padded_dimension) {
  is_padded_dimension_ = _is_padded_dimension;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::padded_to_size(
    c10::optional<int64_t> _padded_to_size) {
  padded_to_size_ = _padded_to_size;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_mma_swizzled(bool _is_mma_swizzled) {
  is_mma_swizzled_ = _is_mma_swizzled;
  return *this;
}

IterDomain* IterDomainBuilder::build() const {
  TORCH_INTERNAL_ASSERT(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
  return IrBuilder::create<IterDomain>(start_->container(), *this);
}

IterDomain::IterDomain(
    IrBuilderPasskey passkey,
    Val* start,
    Val* extent,
    Val* expanded_extent,
    Val* stop_offset,
    ParallelType parallel_type,
    IterType iter_type,
    bool is_rfactor_domain,
    bool is_padded_dimension,
    c10::optional<int64_t> padded_to_size,
    bool is_mma_swizzled)
    : Val(passkey, ValType::IterDomain, DataType::Int),
      start_(start),
      extent_(extent),
      expanded_extent_(expanded_extent),
      stop_offset_(
          stop_offset == nullptr ? passkey.ir_container_->zeroVal()
                                 : stop_offset),
      parallel_type_(parallel_type),
      iter_type_(iter_type),
      is_rfactor_domain_(is_rfactor_domain),
      is_padded_dimension_(is_padded_dimension),
      padded_to_size_(padded_to_size),
      is_mma_swizzled_(is_mma_swizzled) {
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
      start,
      " .");
}

IterDomain::IterDomain(IrBuilderPasskey passkey, const IterDomainBuilder& args)

    : IterDomain(
          passkey,
          args.start_,
          args.extent_,
          args.expanded_extent_,
          args.stop_offset_,
          args.parallel_type_,
          args.iter_type_,
          args.is_rfactor_domain_,
          args.is_padded_dimension_,
          args.padded_to_size_,
          args.is_mma_swizzled_) {}

IterDomain::IterDomain(const IterDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      extent_(ir_cloner->clone(src->extent_)),
      expanded_extent_(
          src->hasExpandedExtent() ? ir_cloner->clone(src->expandedExtent())
                                   : nullptr),
      stop_offset_(ir_cloner->clone(src->stop_offset_)),
      parallel_type_(src->parallel_type_),
      iter_type_(src->iter_type_),
      is_rfactor_domain_(src->is_rfactor_domain_),
      is_padded_dimension_(src->is_padded_dimension_),
      padded_to_size_(src->padded_to_size_),
      is_mma_swizzled_(src->is_mma_swizzled_) {}

bool IterDomain::sameAs(const Statement* other) const {
  if (other == this) {
    return true;
  }

  if (!other->isA<IterDomain>()) {
    return false;
  }

  const IterDomain* other_id = other->as<IterDomain>();

  bool is_same = isReduction() == other_id->isReduction() &&
      getParallelType() == other_id->getParallelType() &&
      isVectorComponent() == other_id->isVectorComponent();
  is_same = is_same && ScalarCheck::sameAs(extent(), other_id->extent());
  is_same = is_same && ScalarCheck::sameAs(start(), other_id->start());
  is_same =
      is_same && ScalarCheck::sameAs(stopOffset(), other_id->stopOffset());
  is_same = is_same && (hasExpandedExtent() == other_id->hasExpandedExtent());
  if (is_same && hasExpandedExtent()) {
    is_same = ScalarCheck::sameAs(expandedExtent(), other_id->expandedExtent());
  }

  return is_same;
}

// Returns a new IterDomain matching properties of this except for
// is_rfactor_domain_
IterDomain* IterDomain::cloneWithoutRFactor() const {
  auto cloned = IterDomainBuilder(this).resetRfactor().build();

  return cloned;
}

std::vector<IterDomain*> IterDomain::clone(
    const std::vector<IterDomain*>& domains) {
  std::vector<IterDomain*> cloned_domains;
  std::transform(
      domains.begin(),
      domains.end(),
      std::back_inserter(cloned_domains),
      [](auto id) { return id->cloneWithoutRFactor(); });
  return cloned_domains;
}

// Merging does not propagate the start and stop values of the input
// domains to the merged output domain. The actual range of the
// domains is enforced by predicates. Note that since only root
// domains have valid start and stop, it's not possible to contiguous
// predication.
IterDomain* IterDomain::merge(IterDomain* outer, IterDomain* inner) {
  TORCH_CHECK(
      !outer->extent()->isZeroInt() && !inner->extent()->isZeroInt(),
      "Merging IterDomains with ending values that are 0 is not supported at this time.");
  TORCH_CHECK(
      outer->isReduction() == inner->isReduction() ||
          (!outer->isReduction() && inner->isTrivialReduction()) ||
          (outer->isTrivialReduction() && !inner->isReduction()),
      "Merging IterDomains requires that their iteration types match.");
  TORCH_CHECK(
      (outer->isGather() && inner->isGather()) ||
          (!outer->isGather() && !inner->isGather()),
      "Merging gather and non-gather domains is not supported.");

  TORCH_CHECK(
      !outer->isStride() && !inner->isStride(),
      "No support for merging stride domains");

  Val* merged_id_size = mul(outer->extent(), inner->extent());

  IterType itype = outer->getIterType();

  if (outer->isBroadcast() && inner->isBroadcast()) {
    itype = IterType::Broadcast;
  }

  if ((outer->isBroadcast() || inner->isBroadcast()) &&
      (outer->getIterType() == IterType::Iteration ||
       inner->getIterType() == IterType::Iteration)) {
    itype = IterType::Iteration;
  }

  // Merging trivial reduction with iter domain, that's fine, just make it an
  // iter domain.
  if ((outer->isTrivialReduction() || inner->isTrivialReduction()) &&
      (outer->getIterType() == IterType::Iteration ||
       inner->getIterType() == IterType::Iteration)) {
    itype = IterType::Iteration;
  }

  // Merging trivial reduction with broadcasting, that's fine, just make it a
  // broadcasting.
  if ((outer->isTrivialReduction() || inner->isTrivialReduction()) &&
      (outer->isBroadcast() || inner->isBroadcast())) {
    itype = IterType::Broadcast;
  }

  Val* expanded_extent = nullptr;
  if (outer->hasExpandedExtent() || inner->hasExpandedExtent()) {
    if (outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      expanded_extent = mul(outer->expandedExtent(), inner->expandedExtent());
    } else if (outer->hasExpandedExtent() && !inner->hasExpandedExtent()) {
      if (inner->isBroadcast()) {
        expanded_extent = outer->expandedExtent();
      } else {
        expanded_extent = mul(outer->expandedExtent(), inner->extent());
      }
    } else if (outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      if (outer->isBroadcast()) {
        expanded_extent = inner->expandedExtent();
      } else {
        expanded_extent = mul(outer->extent(), inner->expandedExtent());
      }
    }
  }

  IterDomain* merged_id =
      IterDomainBuilder(
          outer->container()->zeroVal(), merged_id_size->as<Int>())
          .parallel_type(outer->getParallelType())
          .expanded_extent(expanded_extent)
          .iter_type(itype)
          .build();

  IrBuilder::create<Merge>(outer->container(), merged_id, outer, inner);

  return merged_id;
}

// Both outer and inner domains do not inherit start and stop
// values as they can't be split. The access range is enforced by
// predicates.
std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    Val* start_offset,
    Val* stop_offset) {
  TORCH_CHECK(
      !in->extent()->isZeroInt(),
      "Splitting IterDomains with ending values that are 0 is not supported at this time.");

  TORCH_CHECK(factor->isAnInt(), "Cannot split by non-integer value ", factor);

  if (factor->getValType() == ValType::Scalar) {
    TORCH_CHECK(
        factor->isConstScalar() ||
            (FusionGuard::getCurFusion() == factor->fusion() &&
             factor->isFusionInput()),
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
  Val* remainder =
      ceilDiv(Split::extent(in->extent(), start_offset, stop_offset), factor);
  Val* expanded_remainder = nullptr;
  if (in->hasExpandedExtent()) {
    expanded_remainder = ceilDiv(
        Split::extent(in->expandedExtent(), start_offset, stop_offset), factor);
  }

  if ((start_offset != nullptr && !start_offset->isZeroInt()) ||
      (stop_offset != nullptr && !stop_offset->isZeroInt())) {
    TORCH_INTERNAL_ASSERT(
        in->definition() == nullptr,
        "Partial split is only allowed with root domains");
  }
  // outer loop IterDomain
  IterDomain* ido =
      IterDomainBuilder(
          in->container()->zeroVal(),
          inner_split ? remainder->as<Int>() : factor)
          .expanded_extent(
              in->hasExpandedExtent() && inner_split ? expanded_remainder
                                                     : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(in->getIterType())
          .build();

  // inner loop IterDomain
  IterDomain* idi =
      IterDomainBuilder(
          in->container()->zeroVal(),
          inner_split ? factor : remainder->as<Int>())
          .expanded_extent(
              in->hasExpandedExtent() && !inner_split ? expanded_remainder
                                                      : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(in->getIterType())
          .build();

  IrBuilder::create<Split>(
      in->container(),
      ido,
      idi,
      in,
      factor,
      inner_split,
      start_offset,
      stop_offset);
  return {ido, idi};
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  auto start_offset = trim_out_of_bounds ? in->start() : nullptr;
  auto stop_offset = trim_out_of_bounds ? in->stopOffset() : nullptr;
  return IterDomain::split(in, factor, inner_split, start_offset, stop_offset);
}

std::pair<IterDomain*, IterDomain*> IterDomain::stridedSplit(int factor) {
  // Use partial split so that only valid values are retained
  auto split_out = IterDomain::split(
      this, IrBuilder::create<Int>(container(), factor), true, true);

  split_out.second->iter_type_ = IterType::Stride;
  split_out.first->is_rfactor_domain_ = true;
  split_out.second->is_rfactor_domain_ = true;
  return split_out;
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    Swizzle2DType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y,
    SwizzleMode swizzle_mode) {
  TORCH_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  TORCH_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs(in_x->fusion(), {in_x, in_y})) {
    TORCH_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  // TODO: gather and shift check on swizzle
  TORCH_INTERNAL_ASSERT(
      !in_x->isGather() && !in_y->isGather(),
      "Swizzled gather not yet supported");

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::create<Swizzle2D>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type, swizzle_mode);

  return std::make_pair(out_x, out_y);
}

// TODO: We should change parallelize interface to be on tensorview or at least
// vectorize should be done on tensorview. This would let us check that we don't
// vectorize to the left of the computeAt domain, and could allow us to do some
// simple validation of vectorize as it's inputs are right most and contiguous.
void IterDomain::parallelize(ParallelType t) {
  if (parallel_type_ == t) {
    // No op, don't do any more checks, it was already set to this value.
    return;
  }

  if (t == ParallelType::Unroll || isParallelTypeVectorize(t) ||
      t == ParallelType::Group) {
    TORCH_CHECK(
        start()->isZeroInt() && extent()->isConstScalar(),
        "Vectorization, unrolling, unswitching and grouping are only supported with start = 0 and extent as a const int, but got ",
        "a start of ",
        start(),
        " and extent ",
        extent(),
        " .");
  }

  if (t == ParallelType::Group) {
    TORCH_CHECK(
        getIterType() == IterType::Iteration,
        "Grouping IterDomain of non Iteration type is not allowed. ",
        getIterType());
  }

  if (isMmaSwizzled()) {
    // Mma swizzled axes represent data representation within a warp
    //  so only allow updates that keep the parallelization within
    //  a warp.
    // Note && TODO: this check is actually used to allow indexing path
    //  to make copies of the iterdomains. We might eventually just want
    //  to lock these parallel types and not allowing any changes once
    //  they are swizzled.
    TORCH_CHECK(
        t == ParallelType::Vectorize || t == ParallelType::TIDx ||
            t == ParallelType::Serial,
        "Parallel type other than serial, tidx, vectorize not allowed for mma swizzled ids");
  }

  parallel_type_ = t;
}

bool IterDomain::maybePartial() const {
  return !start()->isZeroInt() || !stopOffset()->isZeroInt();
}

Val* IterDomain::stopOffset() const {
  return stop_offset_;
}

Val* IterDomain::stop() const {
  if (stopOffset()->isZeroInt()) {
    return extent();
  }

  return sub(extent(), stopOffset());
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<bool> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == getMaybeRFactorDomain().size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  // Just due to clang-tidy, correct value set in resetDomains
  has_nontrivial_reduction_ = false;
  domain_ = root_domain_;
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> domain,
    std::vector<bool> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      domain_(std::move(domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == getMaybeRFactorDomain().size(),
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

  // Just due to clang-tidy, correct value set in resetDomains
  has_nontrivial_reduction_ = false;
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> rfactor_domain,
    std::vector<IterDomain*> domain,
    std::vector<bool> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      domain_(std::move(domain)),
      rfactor_domain_(std::move(rfactor_domain)),
      contiguity_(
          contiguity.empty() ? std::vector<bool>(rfactor_domain_.size(), false)
                             : std::move(contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == getMaybeRFactorDomain().size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      getMaybeRFactorDomain().size());

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

  // Just due to clang-tidy, correct value set in resetDomains
  has_nontrivial_reduction_ = false;
  resetDomains();
}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      domain_(ir_cloner->clone(src->domain_)),
      no_bcast_domain_(ir_cloner->clone(src->no_bcast_domain_)),
      no_reduction_domain_(ir_cloner->clone(src->no_reduction_domain_)),
      rfactor_domain_(ir_cloner->clone(src->rfactor_domain_)),
      contiguity_(src->contiguity()),
      has_nontrivial_reduction_(src->has_nontrivial_reduction_) {}

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isBroadcast() && id->isThreadDim();
  });
}

bool TensorDomain::hasGridBroadcast() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isBroadcast() && id->isBlockDim();
  });
}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Should not be necessary to
  // check no_bcast_domain_ and no_reduction_domain_ as they are just
  // derived from domain_.
  return root_domain_ == other.root_domain_ && domain_ == other.domain_ &&
      rfactor_domain_ == other.rfactor_domain_ &&
      contiguity_ == other.contiguity_;
}

bool TensorDomain::sameAs(const Statement* const other) const {
  if (this == other) {
    return true;
  }

  if (!other->isA<TensorDomain>()) {
    return false;
  }

  const TensorDomain* other_td = other->as<TensorDomain>();

  if (nDims() != other_td->nDims()) {
    return false;
  }
  if (getRootDomain().size() != other_td->getRootDomain().size()) {
    return false;
  }
  if (getRFactorDomain().size() != other_td->getRFactorDomain().size()) {
    return false;
  }

  for (const auto i : c10::irange(nDims())) {
    if (!(axis(i)->sameAs(other_td->axis(i)))) {
      return false;
    }
  }

  for (const auto i : c10::irange(getRootDomain().size())) {
    if (!(getRootDomain()[i]->sameAs(other_td->getRootDomain()[i]))) {
      return false;
    }
  }

  for (const auto i : c10::irange(getRFactorDomain().size())) {
    if (!(getRFactorDomain()[i]->sameAs(other_td->getRFactorDomain()[i]))) {
      return false;
    }
  }

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

void TensorDomain::setContiguity(const std::vector<bool>& contig) {
  TORCH_INTERNAL_ASSERT(
      getMaybeRFactorDomain().size() == contig.size(),
      "Invalid contiguity vector: ",
      contig);

  contiguity_ = contig;
}

bool TensorDomain::hasReduction() const {
  return has_nontrivial_reduction_;
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

bool TensorDomain::hasBroadcast() const {
  return no_bcast_domain_.size() != domain_.size();
}

bool TensorDomain::hasRFactor() const {
  return !rfactor_domain_.empty();
}

bool TensorDomain::hasViewLikeRFactor() const {
  if (!hasRFactor()) {
    // Can't have view like rfactor if there is no rfactor domain
    return false;
  }

  // If there's an rfactor domain and no rfactor product is a reduction, this is
  // a view like rfactor
  return std::none_of(
      getMaybeRFactorDomain().begin(),
      getMaybeRFactorDomain().end(),
      [](IterDomain* id) {
        return id->isReduction() && id->isRFactorProduct();
      });
}

bool TensorDomain::hasVectorize() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->getParallelType() == ParallelType::Vectorize ||
        id->getParallelType() == ParallelType::MisalignedVectorize;
  });
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

size_t TensorDomain::rootPosOf(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      root_domain_.size() > 0, "Tried to find an axis in a 0-dim root domain");
  auto it = std::find(root_domain_.begin(), root_domain_.end(), id);
  TORCH_INTERNAL_ASSERT(
      it != root_domain_.end(), "Provided id is not part of root domain.");
  return std::distance(root_domain_.begin(), it);
}

void TensorDomain::split(
    int axis_,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim domain");
  if (axis_ < 0)
    axis_ += nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0 && (unsigned int)axis_ < nDims(),
      "Tried to split on axis outside TensorDomain's range.");

  IterDomain* id = axis(axis_);

  // partial split is only allowed with root domains
  if (trim_out_of_bounds) {
    TORCH_INTERNAL_ASSERT(
        std::find(getRootDomain().begin(), getRootDomain().end(), id) !=
            getRootDomain().end(),
        "Partial split is only allowed with root domains");
  }

  TORCH_INTERNAL_ASSERT(
      !id->isMmaSwizzled(),
      "Further transformation on warp mapped id's not allowed.");

  auto split_ids =
      IterDomain::split(id, factor, inner_split, trim_out_of_bounds);
  domain_.erase(domain_.begin() + axis_);
  domain_.insert(domain_.begin() + axis_, split_ids.second);
  domain_.insert(domain_.begin() + axis_, split_ids.first);
  resetDomains();
}

// Merge "axis_o" and "axis_i" into 1 dimension
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

  TORCH_INTERNAL_ASSERT(
      !first->isMmaSwizzled() && !second->isMmaSwizzled(),
      "Further transformation on warp mapped id's not allowed.");

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

  auto new2old = ir_utils::normalizeOld2New(old2new_, dom.size());

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

void TensorDomain::swizzle(
    Swizzle2DType swizzle_type,
    int x,
    int y,
    SwizzleMode swizzle_mode) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim domain");

  TORCH_CHECK(
      x >= 0 && (unsigned int)x < nDims(),
      "Invalid swizzle detected, either one or both axes are outside of TensorView's range.");

  TORCH_CHECK(
      y >= 0 && (unsigned int)y < nDims(),
      "Invalid swizzle detected, either one or both axes are outside of TensorView's range.");

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y, swizzle_mode);

  domain_.erase(domain_.begin() + x);
  domain_.insert(domain_.begin() + x, axis_out_x);

  domain_.erase(domain_.begin() + y);
  domain_.insert(domain_.begin() + y, axis_out_y);

  resetDomains();
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  size_t size_out = 0;
  for (auto id : td) {
    if (!id->isReduction() && !id->isStride()) {
      size_out++;
    }
  }
  std::vector<IterDomain*> noReductionDomain(size_out);

  int it = 0;
  for (auto id : td) {
    if (!id->isReduction() && !id->isStride()) {
      noReductionDomain[it++] = id;
    }
  }

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

TensorDomain* TensorDomain::view(const AnalyzeViewResult& view_analysis) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to view transform a 0-dim domain");
  return transformView(this, view_analysis);
}

TensorDomain* TensorDomain::flatten(int64_t start_dim, int64_t end_dim) {
  auto inp_domain = noReductions(getMaybeRFactorDomain());

  if (start_dim < 0) {
    start_dim += inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += inp_domain.size();
  }
  TORCH_CHECK(
      start_dim >= 0 && start_dim < inp_domain.size(),
      "Invalid start_dim ",
      start_dim);
  TORCH_CHECK(
      end_dim >= 0 && end_dim < inp_domain.size(), "Invalid end_dim ", end_dim);
  TORCH_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  std::vector<IterDomain*> new_root_domain;
  new_root_domain.reserve(inp_domain.size());
  for (auto i : c10::irange(inp_domain.size())) {
    bool is_rfactor_dim = i >= start_dim && i <= end_dim;
    auto inp_id = inp_domain[i];
    auto out_id = IterDomainBuilder(inp_id)
                      .is_rfactor_domain(is_rfactor_dim)
                      .extent(
                          (is_rfactor_dim && inp_id->hasExpandedExtent())
                              ? inp_id->expandedExtent()
                              : inp_id->extent())
                      .iter_type(
                          (is_rfactor_dim && inp_id->isBroadcast())
                              ? IterType::Iteration
                              : inp_id->getIterType())
                      .build();
    new_root_domain.push_back(out_id);
  }

  std::vector<IterDomain*> rfactor_domain;
  rfactor_domain.reserve(new_root_domain.size() - (end_dim - start_dim));
  for (auto i : c10::irange(start_dim)) {
    rfactor_domain.push_back(new_root_domain[i]);
  }

  IterDomain* merged_id = new_root_domain[start_dim];
  for (auto i : c10::irange(start_dim + 1, end_dim + 1)) {
    IterDomain* new_merged_id =
        IterDomainBuilder(
            merged_id->container()->zeroVal(),
            mul(merged_id->extent(), new_root_domain[i]->extent()))
            .is_rfactor_domain(true)
            .build();
    IrBuilder::create<Merge>(new_merged_id, merged_id, new_root_domain[i]);
    merged_id = new_merged_id;
  }
  rfactor_domain.push_back(merged_id);

  for (auto i : c10::irange(end_dim + 1, inp_domain.size())) {
    rfactor_domain.push_back(new_root_domain[i]);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      rfactor_domain,
      rfactor_domain,
      std::vector<bool>(rfactor_domain.size(), true));
}

// TODO: Rfactor a Welford

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int>& axes_) {
  return TransformRFactor::runReplay(this, axes_);
}

Split::Split(
    IrBuilderPasskey passkey,
    IterDomain* outer,
    IterDomain* inner,
    IterDomain* in,
    Val* factor,
    bool inner_split,
    Val* start_offset,
    Val* stop_offset)
    : Expr(passkey, ExprType::Split),
      outer_{outer},
      inner_{inner},
      in_{in},
      factor_{factor},
      inner_split_{inner_split},
      start_offset_{
          start_offset != nullptr ? start_offset
                                  : passkey.ir_container_->zeroVal()},
      stop_offset_{
          stop_offset != nullptr ? stop_offset
                                 : passkey.ir_container_->zeroVal()} {
  TORCH_INTERNAL_ASSERT(
      factor_->isAnInt(),
      "Attempted to create a Split node with a non-integer factor.");
  addOutput(outer);
  addOutput(inner);
  addInput(in);
  // TODO add factor as an input, need to check Split::Split during validation
  // and need to check BestEffortReplay::findFirstMismatchedID addInput(factor);
}

Split::Split(const Split* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)),
      in_(ir_cloner->clone(src->in_)),
      factor_(ir_cloner->clone(src->factor_)),
      inner_split_(src->inner_split_),
      start_offset_(ir_cloner->clone(src->start_offset_)),
      stop_offset_(ir_cloner->clone(src->stop_offset_)) {}

Val* Split::extent(Val* in_extent, Val* start_offset, Val* stop_offset) {
  TORCH_INTERNAL_ASSERT(in_extent != nullptr);

  if (start_offset != nullptr && !start_offset->isZeroInt()) {
    in_extent = sub(in_extent, start_offset);
  }

  if (stop_offset != nullptr && !stop_offset->isZeroInt()) {
    in_extent = sub(in_extent, stop_offset);
  }

  return in_extent;
}

bool Split::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Split>()) {
    return false;
  }
  return Expr::sameAs(other) &&
      factor()->sameAs(other->as<Split>()->factor()) &&
      innerSplit() == other->as<Split>()->innerSplit() &&
      startOffset()->sameAs(other->as<Split>()->startOffset()) &&
      stopOffset()->sameAs(other->as<Split>()->stopOffset());
}

Merge::Merge(
    IrBuilderPasskey passkey,
    IterDomain* out,
    IterDomain* outer,
    IterDomain* inner)
    : Expr(passkey, ExprType::Merge), out_{out}, outer_{outer}, inner_{inner} {
  addOutput(out);
  addInput(outer);
  addInput(inner);
}

Merge::Merge(const Merge* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)) {}

bool Merge::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Merge>()) {
    return false;
  }
  return Expr::sameAs(other);
}

Swizzle2D::Swizzle2D(
    IrBuilderPasskey passkey,
    IterDomain* out_x,
    IterDomain* out_y,
    IterDomain* in_x,
    IterDomain* in_y,
    Swizzle2DType swizzle_type,
    SwizzleMode swizzle_mode)
    : Expr(passkey, ExprType::Swizzle2D),
      out_x_{out_x},
      out_y_{out_y},
      in_x_{in_x},
      in_y_{in_y},
      swizzle_type_(swizzle_type),
      swizzle_mode_(swizzle_mode) {
  addOutput(out_x);
  addOutput(out_y);
  addInput(in_x);
  addInput(in_y);
}

bool Swizzle2D::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Swizzle2D>()) {
    return false;
  }
  if (!(swizzle_type_ == other->as<Swizzle2D>()->swizzle_type_)) {
    return false;
  }
  return Expr::sameAs(other);
}

Swizzle2D::Swizzle2D(const Swizzle2D* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_x_(ir_cloner->clone(src->out_x_)),
      out_y_(ir_cloner->clone(src->out_y_)),
      in_x_(ir_cloner->clone(src->in_x_)),
      in_y_(ir_cloner->clone(src->in_y_)),
      swizzle_type_(src->swizzle_type_),
      swizzle_mode_(src->swizzle_mode_) {}

NamedScalar::NamedScalar(
    IrBuilderPasskey passkey,
    std::string name,
    DataType dtype)
    : Val(passkey, ValType::NamedScalar, dtype), name_(std::move(name)) {}

NamedScalar::NamedScalar(const NamedScalar* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), name_(src->name_) {}

bool NamedScalar::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<NamedScalar>()) {
    return false;
  }
  return other->as<NamedScalar>()->name().compare(name()) == 0;
}

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  TORCH_INTERNAL_ASSERT(
      isParallelTypeThread(p_type),
      "Cannot get parallel dim of non thread type, received: ",
      p_type);
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_dim = stringifyThreadSize(p_type);
  return IrBuilder::create<NamedScalar>(parallel_dim, DataType::Int);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_ind = stringifyThread(p_type);
  return IrBuilder::create<NamedScalar>(parallel_ind, DataType::Int);
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
