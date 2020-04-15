#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/ir/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

Val* Statement::asVal() {
  TORCH_INTERNAL_ASSERT(isVal(), "Cannot cast to Val as this is not a Val.");
  return static_cast<Val*>(this);
}

Expr* Statement::asExpr() {
  TORCH_INTERNAL_ASSERT(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return static_cast<Expr*>(this);
}

// When we create a Val we immediately register them with the active fusion.
Val::Val(ValType _vtype, DataType _dtype) : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion != nullptr) {
    this->name_ = fusion->registerVal(this);
    this->fusion_ = fusion;
  } else {
    TORCH_CHECK(false, "No active fusion group found when creating a Val.");
  }
}

namespace {

struct ConstCheck : OptOutConstDispatch {
 private:
  bool is_const_ = false;

  void handle(const Float* const f) override {
    is_const_ = f->isConst();
  }

  void handle(const Int* const i) override {
    is_const_ = i->isConst();
  }

  void handle(const Expr* const expr) override {
    for (auto inp : expr->inputs()) {
      OptOutConstDispatch::handle(inp);
    }
  }

  void handle(const NamedScalar* const ns) override {
    is_const_ = false;
  }

  void handle(const Val* const val) override {
    const Expr* orig = FusionGuard::getCurFusion()->origin(val);
    if (orig != nullptr)
      handle(orig);
    else
      OptOutConstDispatch::handle(val);
  }

 public:
  static bool isConst(const Val* const val) {
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

c10::optional<DataType> Val::getDataType() const {
  TORCH_INTERNAL_ASSERT(
      dtype_ != DataType::Null, "Value does not have a data type.");
  return dtype_;
}

Expr* Val::getOrigin() {
  return (fusion_->origin(this));
}

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

bool Scope::sameAs(const Scope& other) const {
  if (other.exprs().size() != this->exprs().size())
    return false;
  for (decltype(exprs().size()) i{0}; i < exprs().size(); i++)
    if (other.exprs()[i] != exprs()[i])
      return false;
  return true;
}

bool IRInputOutput::hasInput(const Val* const input) const {
  for (auto val : inputs_)
    if (val == input)
      return true;
  return false;
}

bool IRInputOutput::hasOutput(const Val* const output) const {
  for (auto val : outputs_)
    if (val == output)
      return true;
  return false;
}

void IRInputOutput::replaceInput(Val* replace, Val* with) {
  bool changed = false;
  for (decltype(inputs_.size()) i{0}; i < inputs_.size(); i++) {
    if (inputs_[i] == replace) {
      inputs_[i] = with;
      changed = true;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(
      changed,
      "Error detected when trying to replace input ",
      replace,
      " with ",
      with,
      " .");
}

void IRInputOutput::replaceOutput(Val* replace, Val* with) {
  bool changed = false;
  for (decltype(outputs_.size()) i{0}; i < outputs_.size(); i++) {
    if (outputs_[i] == replace) {
      outputs_[i] = with;
      changed = true;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(
      changed,
      "Error detected when trying to replace output ",
      replace,
      " with ",
      with,
      " .");
}

void IRInputOutput::removeInput(Val* val) {
  auto it = inputs_.begin();
  for (; it != inputs_.end(); ++it) {
    if ((*it) == val)
      break;
  }
  TORCH_INTERNAL_ASSERT(it != inputs_.end());
  inputs_.erase(it);
}

void IRInputOutput::removeOutput(Val* val) {
  auto it = outputs_.begin();
  for (; it != outputs_.end(); ++it) {
    if ((*it) == val)
      break;
  }
  TORCH_INTERNAL_ASSERT(it != outputs_.end());
  outputs_.erase(it);
}

// We don't register with the active fusion in Expr as this needs to be done
// after inputs and outputs are registered with the Expr
Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    TORCH_CHECK(false, "No active fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

} // namespace fuser
} // namespace jit
} // namespace torch
