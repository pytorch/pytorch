#include <dispatch.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_cloner.h>
#include <ir_printer.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <mutator.h>

#include <torch/csrc/jit/ir/ir.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace nvfuser {

Statement::Statement(IrBuilderPasskey passkey) {
  ir_container_ = passkey.ir_container_;
}

Statement::Statement(const Statement* src, IrCloner* ir_cloner) {
  ir_container_ = ir_cloner->container();
}

NVFUSER_DEFINE_CLONE(Statement)

void Statement::setName(IrContainerPasskey, StmtNameType name) {
  name_ = name;
}

void Statement::setName(IrBuilderPasskey, StmtNameType name) {
  name_ = name;
}

Val* Statement::asVal() {
  TORCH_INTERNAL_ASSERT(isVal(), "Cannot cast to Val as this is not a Val.");
  return this->as<Val>();
}

Expr* Statement::asExpr() {
  TORCH_INTERNAL_ASSERT(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return this->as<Expr>();
}

bool Statement::lessThan(const Statement* stmt1, const Statement* stmt2) {
  TORCH_INTERNAL_ASSERT(stmt1 != nullptr);
  TORCH_INTERNAL_ASSERT(stmt2 != nullptr);
  return stmt1->name() < stmt2->name();
}

std::string Statement::toString(int indent_size) const {
  TORCH_INTERNAL_ASSERT(
      false, "toString for IR node ", typeid(*this).name(), " is not defined");
}

std::string Statement::toInlineString(int indent_size) const {
  TORCH_INTERNAL_ASSERT(
      false,
      "toInlineString for IR node ",
      typeid(*this).name(),
      " is not defined");
}

Fusion* Statement::fusion() const {
  TORCH_INTERNAL_ASSERT(
      ir_container_->isA<Fusion>(), "Statement does not belong to a fusion.");
  return ir_container_->as<Fusion>();
}

kir::Kernel* Statement::kernel() const {
  TORCH_INTERNAL_ASSERT(
      ir_container_->isA<kir::Kernel>(),
      "Statement does not belong to a kernel.");
  return ir_container_->as<kir::Kernel>();
}

// When we create a Val we immediately register them with the active fusion.
Val::Val(IrBuilderPasskey passkey, ValType _vtype, DataType _dtype)
    : Statement(passkey), vtype_(_vtype), dtype_(_dtype) {}

// NOTE: we don't clone the definition_ and uses_ here
//  since they may introduce cloning cycles. Instead, we copy
//  the original pointers and we'll fix them up later part of the
//  Fusion copy. Neither definition_ nor uses_ are copied through
//  this constructor now leaving them to be resolved by later stages
//
Val::Val(const Val* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner), vtype_(src->vtype_), dtype_(src->dtype_) {}

NVFUSER_DEFINE_CLONE(Val)

const std::vector<Expr*>& Val::uses() const {
  if (vtype_ == ValType::TensorView) {
    if (!fusion()->isTVUseInfoValid() && !fusion()->isUpdatingTVUseInfo()) {
      fusion()->resetTvUses();
    }
  }
  return uses_;
}

// Converts the data type of TensorView or Scalar representing index
// values. The data type of the original input should be
// DataType::Index, but DataType::Int is also allowed as it is used
// for index expressions.
void Val::resolveIndexDtype() {
  TORCH_INTERNAL_ASSERT(
      vtype_ == ValType::TensorView || vtype_ == ValType::Scalar ||
          vtype_ == ValType::NamedScalar,
      "Resolving index type is currently only supported on tensor view or scalar values. "
      "Value type: ",
      vtype_);
  TORCH_INTERNAL_ASSERT(
      isIntegralType(dtype_),
      "Can only resolve index type if a Val has an Index or Int DataType. ",
      "Data type: ",
      dtype_);
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(),
      "Index type can only be resolved at compile time.");
  dtype_ = container()->as<kir::Kernel>()->indexType();
}

namespace {

// Traverse definition of all values involved in constructing the provided val.
// Check if all values involved are constant values, meaning the provided
// val is also a constant value.
class ConstCheck : private OptOutConstDispatch {
 private:
  bool is_const_ = true;

  // Returns true if all Val's in the hisotry of provided Val is an Int. Since
  // our expression evaluator doesn't support any type besides int, it's
  // important to check it is one.
  bool is_int_ = true;

  void handle(const Bool* b) final {
    is_const_ = is_const_ && b->isConst();
  }

  void handle(const Double* d) final {
    is_const_ = is_const_ && d->isConst();
  }

  void handle(const Int* i) final {
    is_const_ = is_const_ && i->isConst();
  }

  void handle(const NamedScalar* ns) final {
    is_const_ = false;
  }

  void handle(const TensorView* ns) final {
    is_const_ = false;
  }

  void handle(const kir::TensorIndex* ns) final {
    is_const_ = false;
  }

  void handle(const Expr* expr) final {
    for (auto inp : expr->inputs()) {
      handle(inp);
    }
  }

  void handle(const Val* val) final {
    if (!val->isIntegralScalar()) {
      is_int_ = false;
    }

    if (val->definition() != nullptr) {
      handle(val->definition());
    } else {
      OptOutConstDispatch::handle(val);
    }
  }

 public:
  static bool isConst(const Val* val) {
    ConstCheck cc;
    cc.handle(val);
    return cc.is_const_;
  }

  static bool isConstInt(const Val* val) {
    ConstCheck cc;
    cc.handle(val);
    return cc.is_const_ && cc.is_int_;
  }
};

} // namespace

bool Val::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (auto other_val = dynamic_cast<const Val*>(other)) {
    if (definition_ == nullptr || other_val->definition_ == nullptr) {
      return false;
    }
    if (vtype_ != other_val->vtype_) {
      return false;
    }
    if (dtype_ != other_val->dtype_) {
      return false;
    }
    if (!definition_->sameAs(other_val->definition_)) {
      return false;
    }
    if (definition_->outputs().size() !=
        other_val->definition_->outputs().size()) {
      return false;
    }
    // For definition with multiple outputs, only outputs at the same position
    // could be the same
    for (auto i : c10::irange(definition_->outputs().size())) {
      if ((definition_->output(i) == this) !=
          (other_val->definition_->output(i) == other_val)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool Val::isConstScalar() const {
  if (!isScalar()) {
    return false;
  }
  return ConstCheck::isConst(this);
}

bool Val::isConstInt() const {
  return ConstCheck::isConst(this) && isIntegralScalar();
}

int64_t Val::evaluateInt() {
  TORCH_INTERNAL_ASSERT(
      ConstCheck::isConst(this),
      "Cannot get Int of not const values through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->as<Int>()->value().has_value()) {
    return this->as<Int>()->value().value();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.has_value(),
      "Detected a const integer but failed to infer its value.");
  return evaluated_val->as<int64_t>();
}

double Val::evaluateDouble() {
  TORCH_INTERNAL_ASSERT(
      ConstCheck::isConst(this),
      "Cannot get Double of not const doubles through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->as<Double>()->value().has_value()) {
    return this->as<Double>()->value().value();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.has_value(),
      "Detected a const integer but failed to infer its value.");
  return evaluated_val->as<double>();
}

bool Val::evaluateBool() {
  TORCH_INTERNAL_ASSERT(
      ConstCheck::isConst(this),
      "Cannot get Bool of not const bools through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->as<Bool>()->value().has_value()) {
    return this->as<Bool>()->value().value();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.has_value(),
      "Detected a const integer but failed to infer its value.");
  return evaluated_val->as<bool>();
}

c10::optional<int64_t> Val::getInt() const {
  if (isConstScalar() && isIntegralScalar() && isA<Int>()) {
    return this->as<Int>()->value();
  }
  return c10::nullopt;
}

c10::optional<double> Val::getDouble() const {
  if (isConstScalar() && isFloatingPointScalar() && isA<Double>()) {
    return this->as<Double>()->value();
  }
  return c10::nullopt;
}

c10::optional<bool> Val::getBool() const {
  if (isConstScalar() && isABool() && isA<Bool>()) {
    return this->as<Bool>()->value();
  }
  return c10::nullopt;
}

bool Val::isZero() const {
  return getInt() == 0 || getDouble() == 0.0;
}

bool Val::isZeroInt() const {
  auto int_val = getInt();
  return int_val.has_value() && int_val.value() == 0;
}

bool Val::isOne() const {
  return getInt() == 1 || getDouble() == 1.0;
}

bool Val::isOneInt() const {
  auto int_val = getInt();
  return int_val.has_value() && int_val.value() == 1;
}

c10::optional<DataType> Val::getDataType() const {
  TORCH_INTERNAL_ASSERT(
      dtype_ != DataType::Null, "Value does not have a data type.");
  return dtype_;
}

bool Val::isProducerOf(const Val* other) const {
  TORCH_INTERNAL_ASSERT(other != nullptr);
  TORCH_INTERNAL_ASSERT(container() == other->container());

  if (definition() == nullptr) {
    return false;
  }
  return std::any_of(
      definition()->inputs().begin(),
      definition()->inputs().end(),
      [other](const Val* input) { return input == other; });
}

bool Val::isConsumerOf(const Val* other) const {
  return other->isProducerOf(this);
}

// We don't register with the active fusion in Expr as this needs to be done
// after inputs and outputs are registered with the Expr
Expr::Expr(IrBuilderPasskey passkey) : Statement(passkey) {}

Expr::Expr(const Expr* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner),
      attributes_(ir_cloner->clone(src->attributes_)),
      inputs_(ir_cloner->clone(src->inputs_)),
      outputs_(ir_cloner->clone(src->outputs_)) {}

Expr::Expr(
    IrBuilderPasskey passkey,
    std::vector<Val*> inputs,
    std::vector<Val*> outputs,
    std::vector<Statement*> attributes)
    : Statement(passkey),
      attributes_(std::move(attributes)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)) {}

Expr* Expr::shallowCopy() const {
  auto result =
      newObjectFunc()(ir_container_, inputs(), outputs(), attributes());
  if (container()->isA<kir::Kernel>()) {
    result->predicate_ = predicate_;
    result->write_predicate_ = write_predicate_;
  }
  return result;
}

std::string Expr::getGraphvizLabel() const {
  if (attributes().empty()) {
    return getOpString();
  }
  std::stringstream ss;
  const char* separator = "";
  ss << "{" << getOpString() << "|{";
  for (auto attr : attributes()) {
    ss << separator << attr->toString();
    separator = "|";
  }
  ss << "}}";
  return ss.str();
}

bool Expr::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Expr>()) {
    return false;
  }
  const Expr* other_expr = other->as<Expr>();
  if (typeid(*this) != typeid(*other_expr)) {
    return false;
  }
  if (inputs().size() != other_expr->inputs().size() ||
      outputs().size() != other_expr->outputs().size() ||
      attributes().size() != other_expr->attributes().size()) {
    return false;
  }
  for (const auto i : c10::irange(inputs().size())) {
    if (!input(i)->sameAs(other_expr->input(i))) {
      return false;
    }
  }
  for (const auto i : c10::irange(attributes().size())) {
    if (!attribute(i)->sameAs(other_expr->attribute(i))) {
      return false;
    }
  }
  return true;
}

kir::Predicate* Expr::predicate() const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return predicate_;
}

void Expr::setPredicate(kir::Predicate* predicate) {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  predicate_ = predicate;
}

Expr* Expr::withPredicate(kir::Predicate* predicate) {
  auto result = shallowCopy();
  result->setPredicate(predicate);
  return result;
}

kir::Predicate* Expr::writePredicate() const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return write_predicate_;
}

void Expr::setWritePredicate(kir::Predicate* write_predicate) {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  write_predicate_ = write_predicate;
}

Expr* Expr::withWritePredicate(kir::Predicate* predicate) {
  auto result = shallowCopy();
  result->setWritePredicate(predicate);
  return result;
}

std::vector<EvaluatorValue> Expr::evaluate(
    const std::vector<EvaluatorValue>& inputs) const {
  TORCH_INTERNAL_ASSERT(
      false,
      "`evaluate` method for expression ",
      getOpString(),
      " is not defined. ",
      "Please override the evaluate method");
}

} // namespace nvfuser
