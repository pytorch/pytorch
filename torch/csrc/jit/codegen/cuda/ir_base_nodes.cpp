#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>

#include <torch/csrc/jit/ir/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

Statement::Statement(const Statement* src, IrCloner* ir_cloner) {
  name_ = src->name_;
  fusion_ = ir_cloner->fusion();
  ir_cloner->registerClone(src, this);
}

Val* Statement::asVal() {
  TORCH_INTERNAL_ASSERT(isVal(), "Cannot cast to Val as this is not a Val.");
  return this->as<Val>();
}

Expr* Statement::asExpr() {
  TORCH_INTERNAL_ASSERT(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return this->as<Expr>();
}

void Statement::print() const {
  IrPrinter ir_printer(std::cout);
  ir_printer.handle(this);
  std::cout << std::endl;
}

// When we create a Val we immediately register them with the active fusion.
Val::Val(ValType _vtype, DataType _dtype, bool register_val, bool lowered)
    : vtype_(_vtype), dtype_(_dtype) {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_CHECK(
      fusion != nullptr, "No active fusion group found when creating a Val.");
  fusion_ = fusion;
  if (register_val) {
    if (lowered) {
      name_ = fusion_->registerLoweredVal(this);
    } else {
      name_ = fusion_->registerVal(this);
    }
  }
}

namespace {

// TODO(kir): remove this
ValType lowerValType(ValType vtype) {
  switch (vtype) {
    case ValType::Scalar:
      return ValType::KirScalar;
    case ValType::NamedScalar:
      return ValType::KirNamedScalar;
    case ValType::TensorDomain:
      return ValType::KirTensorDomain;
    case ValType::IterDomain:
      return ValType::KirIterDomain;
    case ValType::TensorView:
      return ValType::KirTensorView;
    default:
      TORCH_CHECK(false, "Unexpected");
  }
}

} // namespace

// TODO(kir): remove this
Val::Val(const Val* fusion_ir_node)
    : vtype_(lowerValType(fusion_ir_node->vtype_)),
      dtype_(fusion_ir_node->dtype_) {
  // The lowered nodes preserve the names from the fusion IR counterparts
  name_ = fusion_ir_node->name_;
  fusion_ = fusion_ir_node->fusion_;
  fusion_->registerLoweredVal(this);
}

Val::Val(const Val* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner), vtype_(src->vtype_), dtype_(src->dtype_) {}

// Traverse origin of all values involved in constructing the provided val.
// Check if all values involved are constant values, meaning the provided
// val is also a constant value.
namespace {

class ConstCheck : OptOutConstDispatch {
 private:
  bool is_const_ = true;

  void handle(const Bool* b) override {
    is_const_ = is_const_ && b->isConst();
  }

  void handle(const Float* f) override {
    is_const_ = is_const_ && f->isConst();
  }

  void handle(const Half* h) override {
    is_const_ = is_const_ && h->isConst();
  }

  void handle(const Int* i) override {
    is_const_ = is_const_ && i->isConst();
  }

  void handle(const NamedScalar* ns) override {
    is_const_ = is_const_ && false;
  }

  void handle(const kir::Bool* b) override {
    is_const_ = is_const_ && b->isConst();
  }

  void handle(const kir::Float* f) override {
    is_const_ = is_const_ && f->isConst();
  }

  void handle(const kir::Half* h) override {
    is_const_ = is_const_ && h->isConst();
  }

  void handle(const kir::Int* i) override {
    is_const_ = is_const_ && i->isConst();
  }

  void handle(const kir::NamedScalar* ns) override {
    is_const_ = is_const_ && false;
  }

  void handle(const Expr* expr) override {
    for (auto inp : expr->inputs()) {
      handle(inp);
    }
  }

  void handle(const Val* val) override {
    const Expr* orig = FusionGuard::getCurFusion()->origin(val);
    if (orig != nullptr)
      handle(orig);
    else
      OptOutConstDispatch::handle(val);
  }

 public:
  static bool isConst(const Val* val) {
    ConstCheck cc;
    cc.handle(val);
    return cc.is_const_;
  }
};

} // namespace

bool Val::isConstScalar() const {
  if (!isScalar())
    return false;
  return ConstCheck::isConst(this);
}

c10::optional<int64_t> Val::getInt() const {
  if (isConstScalar() && isAnInt()) {
    if (this->getValType() == ValType::Scalar) {
      return this->as<Int>()->value();
    } else if (this->getValType() == ValType::KirScalar) {
      return this->as<kir::Int>()->value();
    }
  }
  return c10::optional<int64_t>();
}

bool Val::isZeroInt() const {
  auto int_val = getInt();
  return int_val.has_value() && int_val.value() == 0;
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

Expr* Val::getOrigin() {
  return fusion_->origin(this);
}

const Expr* Val::getOrigin() const {
  return fusion_->origin(this);
}

// We don't register with the active fusion in Expr as this needs to be done
// after inputs and outputs are registered with the Expr
Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    TORCH_CHECK(false, "No active fusion group found when creating an Expr.");
  fusion_ = fusion;
}

Expr::Expr(const Expr* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner),
      type_(src->type_),
      inputs_(ir_cloner->clone(src->inputs_)),
      outputs_(ir_cloner->clone(src->outputs_)) {}

bool Expr::sameAs(const Expr* const other) const {
  if (getExprType() != other->getExprType())
    return false;
  if (inputs().size() != other->inputs().size() ||
      outputs().size() != other->outputs().size())
    return false;
  for (size_t i = 0; i < inputs().size(); i++) {
    if (!input(i)->sameAs(other->input(i)))
      return false;
  }
  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
