#include <instrumentation.h>
#include <ir_builder.h>
#include <ir_cloner.h>
#include <ir_container.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void swap(IrContainer& a, IrContainer& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  using std::swap;

  // Swap the content
  swap(a.vals_up_, b.vals_up_);
  swap(a.vals_, b.vals_);

  swap(a.exprs_up_, b.exprs_up_);
  swap(a.exprs_, b.exprs_);

  swap(a.raw_ptrs_, b.raw_ptrs_);

  swap(a.val_type_name_map_, b.val_type_name_map_);
  swap(a.expr_name_counter_, b.expr_name_counter_);

  // Fixup the Statement::fusion_ links for a
  for (auto val : a.vals_) {
    val->ir_container_ = &a;
  }
  for (auto expr : a.exprs_) {
    expr->ir_container_ = &a;
  }

  // Fixup the Statement::fusion_ links for b
  for (auto val : b.vals_) {
    val->ir_container_ = &a;
  }
  for (auto expr : b.exprs_) {
    expr->ir_container_ = &a;
  }
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  to->clear();
  IrCloner ir_cloner(to);

  for (auto val : from->vals_) {
    to->vals_.insert(ir_cloner.clone(val));
  }

  for (auto expr : from->exprs_) {
    to->exprs_.insert(ir_cloner.clone(expr));
  }

  to->val_type_name_map_ = from->val_type_name_map_;
  to->expr_name_counter_ = from->expr_name_counter_;

  return ir_cloner;
}

IrContainer::IrContainer() = default;

IrContainer::IrContainer(const IrContainer& other) {
  FUSER_PERF_SCOPE("IrContainer copy");
  IrContainer::copy(&other, this);
}

IrContainer::IrContainer(IrContainer&& other) noexcept {
  FUSER_PERF_SCOPE("IrContainer move");
  swap(*this, other);
}

IrContainer& IrContainer::operator=(const IrContainer& other) {
  FUSER_PERF_SCOPE("IrContainer copy assign");
  IrContainer copy(other);
  clear();
  swap(*this, copy);
  return *this;
}

IrContainer& IrContainer::operator=(IrContainer&& other) noexcept {
  FUSER_PERF_SCOPE("IrContainer move assign");
  clear();
  swap(*this, other);
  return *this;
}

IrContainer::~IrContainer() {
  clear();
}

//! Register the Statement with this container
void IrContainer::registerStmt(IrBuilderPasskey, Statement* stmt) {
  if (stmt->isVal()) {
    registerVal(stmt->asVal());
  } else {
    registerExpr(stmt->asExpr());
  }
}

//! Register the Val with this container
void IrContainer::registerVal(IrBuilderPasskey, Val* val) {
  registerVal(val);
}

//! Register expr with this container.
void IrContainer::registerExpr(IrBuilderPasskey, Expr* expr) {
  registerExpr(expr);
}

void IrContainer::registerExpr(ExprPasskey, Expr* expr) {
  registerExpr(expr);
}

void IrContainer::removeExpr(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      exprs_.find(expr) != exprs_.end(),
      "Wanted to remove an expression but it doesn't exist in this container.");
  auto expr_in_deque = std::find_if(
      exprs_up_.begin(),
      exprs_up_.end(),
      [expr](std::unique_ptr<Expr>& expr_up) { return expr_up.get() == expr; });

  TORCH_INTERNAL_ASSERT(
      expr_in_deque != exprs_up_.end(),
      "Wanted to remove an expression but its unique ptr is missing.");

  exprs_.erase(expr);
  exprs_up_.erase(expr_in_deque);
  raw_ptrs_.erase((void*)expr);
}

//! Completely remove val from the fusion, break all dependencies associated
//! with it
void IrContainer::removeVal(Val* val) {
  // Don't remove shortcuts
  if (val == true_val_.get() || val == false_val_.get() ||
      val == one_val_.get() || val == zero_val_.get() ||
      val == magic_zero_val_.get()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      vals_.find(val) != vals_.end(),
      "Wanted to remove a value but it doesn't exist in this container.");
  auto val_in_deque = std::find_if(
      vals_up_.begin(), vals_up_.end(), [val](std::unique_ptr<Val>& val_up) {
        return val_up.get() == val;
      });

  TORCH_INTERNAL_ASSERT(
      val_in_deque != vals_up_.end(),
      "Wanted to remove a value but its unique ptr is missing.");

  vals_.erase(val);
  vals_up_.erase(val_in_deque);
  raw_ptrs_.erase((void*)val);
}

//! Register the Val with this container
void IrContainer::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }

  vals_up_.emplace_back(std::unique_ptr<Val>(val));
  vals_.emplace(vals_up_.back().get());
  val->setName(IrContainerPasskey(), getValName(vals_up_.back()->vtype()));
  raw_ptrs_.emplace((void*)vals_up_.back().get());
}

//! Register expr with this container.
void IrContainer::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }
  exprs_up_.emplace_back(std::unique_ptr<Expr>(expr));
  exprs_.emplace(exprs_up_.back().get());
  expr->setName(IrContainerPasskey(), getExprName());
  raw_ptrs_.emplace((void*)exprs_up_.back().get());
}

void IrContainer::clear() noexcept {
  FUSER_PERF_SCOPE("IrContainer clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  raw_ptrs_.clear();

  val_type_name_map_.clear();
  expr_name_counter_ = 0;
}

bool IrContainer::inContainer(const Statement* stmt) const {
  const void* const_void = (const void*)(stmt);
  void* nonconst_void = const_cast<void*>(const_void); // NOLINT
  if (raw_ptrs_.find(nonconst_void) == raw_ptrs_.end()) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      stmt->container() == this,
      "Container claims to own stmt, but stmt disagrees.");

  Statement* nonconst_stmt = const_cast<Statement*>(stmt); // NOLINT
  if (stmt->isExpr()) {
    TORCH_INTERNAL_ASSERT(
        exprs_.find(nonconst_stmt->as<Expr>()) != exprs_.end(),
        "Somehow container claims to and not to own an Expr.");
  }
  if (stmt->isVal()) {
    TORCH_INTERNAL_ASSERT(
        vals_.find(nonconst_stmt->as<Val>()) != vals_.end(),
        "Somehow container claims to and not to own an Val.");
  }

  return true;
}

// Shortcuts for frequently used vals
Int* IrContainer::zeroVal() {
  if (!zero_val_) {
    auto zero_val = IrBuilder::create<Int>(this, 0);
    TORCH_INTERNAL_ASSERT(vals_up_.back().get() == zero_val);
    zero_val_ = std::unique_ptr<Int>(vals_up_.back().release()->as<Int>());
    vals_up_.pop_back();
  }
  return zero_val_.get();
}

Int* IrContainer::oneVal() {
  if (!one_val_) {
    auto one_val = IrBuilder::create<Int>(this, 1);
    TORCH_INTERNAL_ASSERT(vals_up_.back().get() == one_val);
    one_val_ = std::unique_ptr<Int>(vals_up_.back().release()->as<Int>());
    vals_up_.pop_back();
  }
  return one_val_.get();
}

Bool* IrContainer::falseVal() {
  if (!false_val_) {
    auto false_val = IrBuilder::create<Bool>(this, false);
    TORCH_INTERNAL_ASSERT(vals_up_.back().get() == false_val);
    false_val_ = std::unique_ptr<Bool>(vals_up_.back().release()->as<Bool>());
    vals_up_.pop_back();
  }
  return false_val_.get();
}

Bool* IrContainer::trueVal() {
  if (!true_val_) {
    auto true_val = IrBuilder::create<Bool>(this, true);
    TORCH_INTERNAL_ASSERT(vals_up_.back().get() == true_val);
    true_val_ = std::unique_ptr<Bool>(vals_up_.back().release()->as<Bool>());
    vals_up_.pop_back();
  }
  return true_val_.get();
}

NamedScalar* IrContainer::magicZeroVal() {
  if (!magic_zero_val_) {
    auto magic_zero =
        IrBuilder::create<NamedScalar>(kMagicZeroName, DataType::Int);
    TORCH_INTERNAL_ASSERT(vals_up_.back().get() == magic_zero);
    magic_zero_val_ = std::unique_ptr<NamedScalar>(
        vals_up_.back().release()->as<NamedScalar>());
    vals_up_.pop_back();
  }
  return magic_zero_val_.get();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
