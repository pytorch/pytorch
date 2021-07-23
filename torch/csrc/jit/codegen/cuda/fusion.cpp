#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

// TODO(kir): only needed until we can fix Fusion::origin()
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

static thread_local Fusion* ACTIVE_FUSION = nullptr;

FusionGuard::FusionGuard(Fusion* fusion) {
  prev_fusion = ACTIVE_FUSION;
  ACTIVE_FUSION = fusion;
}

FusionGuard::~FusionGuard() {
  ACTIVE_FUSION = prev_fusion;
}

Fusion* FusionGuard::getCurFusion() {
  return ACTIVE_FUSION;
}

void swap(Fusion& a, Fusion& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  using std::swap;

  // Swap the content
  swap(a.val_set_, b.val_set_);
  swap(a.expr_set_, b.expr_set_);
  swap(a.val_deque_, b.val_deque_);

  swap(a.val_type_name_map_, b.val_type_name_map_);
  swap(a.expr_name_counter_, b.expr_name_counter_);

  swap(a.origin_, b.origin_);
  swap(a.uses_, b.uses_);

  swap(a.inputs_, b.inputs_);
  swap(a.outputs_, b.outputs_);

  // Fixup the Statement::fusion_ links for a
  for (auto val : a.val_set_) {
    val->fusion_ = &a;
  }
  for (auto expr : a.expr_set_) {
    expr->fusion_ = &a;
  }

  // Fixup the Statement::fusion_ links for b
  for (auto val : b.val_set_) {
    val->fusion_ = &b;
  }
  for (auto expr : b.expr_set_) {
    expr->fusion_ = &b;
  }

  // Lowered IR nodes
  swap(a.lowered_val_set_, b.lowered_val_set_);
  swap(a.lowered_expr_set_, b.lowered_expr_set_);
  swap(a.lowered_origin_, b.lowered_origin_);

  for (auto val : a.lowered_val_set_) {
    val->fusion_ = &a;
  }
  for (auto expr : a.lowered_expr_set_) {
    expr->fusion_ = &a;
  }
  for (auto val : b.lowered_val_set_) {
    val->fusion_ = &b;
  }
  for (auto expr : b.lowered_expr_set_) {
    expr->fusion_ = &b;
  }
}

Fusion::Fusion(const Fusion& other) {
  FUSER_PERF_SCOPE("Fusion copy");

  IrCloner ir_cloner(this);

  for (auto val : other.val_set_) {
    val_set_.insert(ir_cloner.clone(val));
  }

  for (auto expr : other.expr_set_) {
    expr_set_.insert(ir_cloner.clone(expr));
  }

  for (auto val : other.val_deque_) {
    val_deque_.push_back(ir_cloner.clone(val));
  }

  val_type_name_map_ = other.val_type_name_map_;
  expr_name_counter_ = other.expr_name_counter_;

  for (const auto& kv : other.origin_) {
    auto val = ir_cloner.clone(kv.first);
    auto expr = ir_cloner.clone(kv.second);
    origin_.insert({val, expr});
  }

  for (const auto& kv : other.uses_) {
    auto val = ir_cloner.clone(kv.first);
    std::unordered_set<Expr*> val_uses;
    for (auto expr : kv.second) {
      val_uses.insert(ir_cloner.clone(expr));
    }
    uses_.insert({val, std::move(val_uses)});
  }

  inputs_ = ir_cloner.clone(other.inputs_);
  outputs_ = ir_cloner.clone(other.outputs_);
}

Fusion::Fusion(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move");
  swap(*this, other);
}

Fusion& Fusion::operator=(const Fusion& other) {
  FUSER_PERF_SCOPE("Fusion copy assign");
  Fusion copy(other);
  clear();
  swap(*this, copy);
  return *this;
}

Fusion& Fusion::operator=(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move assign");
  clear();
  swap(*this, other);
  return *this;
}

Fusion::~Fusion() {
  clear();
}

void Fusion::clear() noexcept {
  FUSER_PERF_SCOPE("Fusion clear");

  // Free the owned values
  for (auto ptr : val_set_) {
    delete ptr;
  }

  // Free the owned expressions
  for (auto ptr : expr_set_) {
    delete ptr;
  }

  val_set_.clear();
  val_deque_.clear();
  expr_set_.clear();

  for (auto& kv : val_type_name_map_) {
    kv.second = 0;
  }

  expr_name_counter_ = 0;

  origin_.clear();
  uses_.clear();

  inputs_.clear();
  outputs_.clear();

  // Lowered IR nodes
  for (auto ptr : lowered_val_set_) {
    delete ptr;
  }
  for (auto ptr : lowered_expr_set_) {
    delete ptr;
  }
  lowered_val_set_.clear();
  lowered_expr_set_.clear();
  lowered_origin_.clear();
}

void Fusion::removeExpr(Expr* expr) {
  assertInFusion(expr, "Cannot remove expr ");
  // If we hit this error too frequently, we could lighten the restrictions so
  // that removing something that doesn't exist simply does nothing. For now,
  // we're going with the strictest model which errors.

  for (auto out : expr->outputs())
    if (origin_.find(out) != origin_.end())
      if (origin_.find(out)->second == expr)
        origin_.erase(out);

  for (auto inp : expr->inputs()) {
    if (uses_.find(inp) != uses_.end()) {
      if (uses_.find(inp)->second.find(expr) != uses_.find(inp)->second.end()) {
        uses_.find(inp)->second.erase(expr);
      }
    }
  }

  expr_set_.erase(expr);

  delete expr;
}

void Fusion::removeVal(Val* val) {
  assertInFusion(val, "Cannot remove val ");

  for (Val* inp : inputs())
    if (val->sameAs(inp))
      TORCH_CHECK(false, "Cannot remove val as it is an input of the fusion.");

  for (Val* out : outputs())
    if (val->sameAs(out))
      TORCH_CHECK(false, "Cannot remove val as it is an output of the fusion.");

  Expr* orig = origin(val);
  if (orig != nullptr)
    removeExpr(origin(val));

  for (Expr* use : unordered_uses(val))
    removeExpr(use);

  val_set_.erase(val);

  for (auto it = val_deque_.begin(); it != val_deque_.end(); it++)
    if (*it == val) {
      val_deque_.erase(it);
      break;
    }

  delete val;
}

void Fusion::addInput(Val* input) {
  assertInFusion(input, "Cannot register input ");

  if (input->getValType().value() == ValType::TensorView) {
    auto tv = input->as<TensorView>();
    if (tv->hasReduction()) {
      TORCH_WARN_ONCE(
          "Registered input ",
          input,
          " has a reduction axis, but this does nothing in the fusion.");
    }
    tv->setMemoryType(MemoryType::Global);
  }

  TORCH_INTERNAL_ASSERT(
      input->getOrigin() == nullptr,
      input,
      " cannot be registered as an input as it is used as an output of an expression (",
      input->getOrigin(),
      ").");
  inputs_.push_back(input);
}

void Fusion::addOutput(Val* output) {
  assertInFusion(output, "Cannot register output ");
  if (output->getValType().value() == ValType::TensorView) {
    auto tv = output->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  }
  outputs_.push_back(output);
}

bool Fusion::inFusion(const Statement* stmt) const {
  bool in_fusion = stmt->fusion() == this;
  Statement* nonconst_stmt = const_cast<Statement*>(stmt); // NOLINT

  if (stmt->isExpr()) {
    in_fusion &= expr_set_.find(nonconst_stmt->as<Expr>()) != expr_set_.end();
  }
  if (stmt->isVal()) {
    in_fusion &= val_set_.find(nonconst_stmt->as<Val>()) != val_set_.end();
  }

  return in_fusion;
}

bool Fusion::inKernelIr(const Statement* stmt) const {
  bool in_fusion = stmt->fusion() == this;
  Statement* nonconst_stmt = const_cast<Statement*>(stmt); // NOLINT

  if (stmt->isExpr()) {
    in_fusion &= lowered_expr_set_.find(nonconst_stmt->as<Expr>()) !=
        lowered_expr_set_.end();
  }
  if (stmt->isVal()) {
    in_fusion &= lowered_val_set_.find(nonconst_stmt->as<Val>()) !=
        lowered_val_set_.end();
  }

  return in_fusion;
}

void Fusion::assertInFusion(const Statement* stmt, const std::string& msg)
    const {
  if (inFusion(stmt)) {
    return;
  }
  if (inKernelIr(stmt)) {
    return;
  }
  TORCH_CHECK(false, msg, " it was not found in the active fusion.");
}

std::vector<Expr*> Fusion::exprs(bool from_outputs_only) {
  return ExprSort::getExprs(this, from_outputs_only);
}

std::unordered_set<Val*> Fusion::inputsOf(Val* val) {
  return InputsOf::output(this, val);
}

void Fusion::validateInputs() {
  std::unordered_set<Val*> all_inputs;
  for (Val* out : outputs()) {
    for (Val* input : inputsOf(out)) {
      all_inputs.insert(input);
    }
  }
  for (Val* input : all_inputs) {
    if (!input->isConstScalar())
      TORCH_CHECK(
          hasInput(input),
          "Could not figure out how ",
          input,
          " is generated, however it was not specified as an input.");
  }
}

void Fusion::print() {
  FUSER_PERF_SCOPE("Fusion::print");

  FusionGuard fg(this);
  std::cout << "%kernel {\n";
  IrMathPrinter op_exprs(std::cout);
  op_exprs.handle(this);
  IrTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
  std::cout << "}\n";
}

void Fusion::printKernel() {
  FUSER_PERF_SCOPE("Fusion::printKernel");
  std::cout << codegen::generateCudaKernel(GpuLower(this).kernel());
}

void Fusion::printMath() {
  FUSER_PERF_SCOPE("Fusion::printMath");

  FusionGuard fg(this);
  for (auto expr : exprs(true))
    std::cout << expr;
}

void Fusion::printTransforms() {
  FUSER_PERF_SCOPE("Fusion::printTransforms");

  FusionGuard fg(this);
  IrTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
}

StmtNameType Fusion::registerVal(Val* val) {
  TORCH_CHECK(!inKernelIr(val));

  if (val->fusion()) {
    if (val->fusion() != this) {
      TORCH_CHECK(false, val, " was not found in the active fusion.");
    }
    if (inFusion(val)) {
      return val->name();
    }
  }

  val_set_.emplace(val);
  val_deque_.push_back(val);
  return getValName(*(val->getValType()));
}

StmtNameType Fusion::registerExpr(Expr* expr) {
  TORCH_CHECK(!inKernelIr(expr));

  if (expr->fusion()) {
    if (expr->fusion() != this) {
      TORCH_CHECK(false, expr, " was not found in the active fusion.");
    }
    if (inFusion(expr)) {
      return expr->name();
    }
  }

  for (Val* input : expr->inputs()) {
    assertInFusion(input, "Input to expr is invalid, ");
    TORCH_CHECK(!inKernelIr(input));
    if (uses_.find(input) == uses_.end()) {
      uses_[input] = {expr};
    } else {
      uses_.find(input)->second.emplace(expr);
    }
  }

  for (Val* output : expr->outputs()) {
    assertInFusion(output, "Output to expr is invalid, ");
    TORCH_CHECK(!inKernelIr(output));
    auto it = origin_.find(output);
    if (it != origin_.end()) {
      removeExpr(it->second); // will also remove origin entry
    }

    origin_[output] = expr;
  }

  expr_set_.emplace(expr);
  return getExprName();
}

StmtNameType Fusion::registerStatement(Statement* stmt) {
  if (inFusion(stmt))
    return stmt->name();

  if (stmt->isVal()) {
    return registerVal(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    return registerExpr(stmt->as<Expr>());
  }

  TORCH_INTERNAL_ASSERT(
      false,
      "Could not register statement as Fusion could not recognize its type.");
  return UNINITIALIZED_STMTNAMETYPE;
}

StmtNameType Fusion::registerLoweredVal(Val* val) {
  TORCH_INTERNAL_ASSERT(val->fusion() == this);
  TORCH_INTERNAL_ASSERT(!inFusion(val));
  TORCH_INTERNAL_ASSERT(!inKernelIr(val));
  lowered_val_set_.insert(val);
  return getValName(*val->getValType());
}

StmtNameType Fusion::registerLoweredExpr(Expr* expr) {
  TORCH_INTERNAL_ASSERT(expr->fusion() == this);
  TORCH_INTERNAL_ASSERT(!inFusion(expr));
  TORCH_INTERNAL_ASSERT(!inKernelIr(expr));

  for (Val* input : expr->inputs()) {
    TORCH_CHECK(inKernelIr(input));
  }

  for (Val* output : expr->outputs()) {
    TORCH_CHECK(inKernelIr(output));
    TORCH_CHECK(lowered_origin_.insert({output, expr}).second);
  }

  lowered_expr_set_.insert(expr);
  return getExprName();
}

bool Fusion::used(Val* val) const {
  assertInFusion(val, "Cannot detect if val was used, ");
  return (uses_.find(val) != uses_.end()) &&
      (uses_.find(val)->second.size() > 0);
}

const std::unordered_set<Val*>& Fusion::vals() const noexcept {
  return val_set_;
}

const std::deque<Val*>& Fusion::deterministic_vals() const noexcept {
  return val_deque_;
}

const std::unordered_set<Expr*>& Fusion::unordered_exprs() const noexcept {
  return expr_set_;
}

std::unordered_set<Expr*> Fusion::unordered_uses(Val* val) const {
  assertInFusion(val, "Cannot detect where val was used, ");
  if (uses_.find(val) != uses_.end()) {
    auto ret = uses_.find(val)->second;
    return ret;
  }
  return std::unordered_set<Expr*>();
}

Expr* Fusion::origin(const Val* val) const {
  // TODO(kir): remove the lowered branch
  if (kir::isLoweredVal(val)) {
    TORCH_INTERNAL_ASSERT(inKernelIr(val));
    auto it = lowered_origin_.find(val);
    return it != lowered_origin_.end() ? it->second : nullptr;
  } else {
    assertInFusion(val, "Cannot detect the origin of val, ");
    auto it = origin_.find(val);
    return it != origin_.end() ? it->second : nullptr;
  }
}

bool Fusion::hasInput(const Val* val) const {
  return std::find(inputs_.begin(), inputs_.end(), val) != inputs_.end();
}

bool Fusion::hasOutput(const Val* val) const {
  return std::find(outputs_.begin(), outputs_.end(), val) != outputs_.end();
}

void Fusion::replaceInput(Val* replace, Val* with) {
  std::replace(inputs_.begin(), inputs_.end(), replace, with);
}

void Fusion::replaceOutput(Val* replace, Val* with) {
  std::replace(outputs_.begin(), outputs_.end(), replace, with);
}

StmtNameType Fusion::getValName(ValType vtype) {
  return val_type_name_map_[vtype]++;
}

StmtNameType Fusion::getExprName() {
  return expr_name_counter_++;
}

// Indicate to kernel to set itself up to generate random numbers
bool Fusion::isStochastic() {
  for (auto expr : exprs(true))
    if (expr->getExprType() == ExprType::UnaryOp)
      if (expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::RandLike)
        return true;
  return false;
}

bool Fusion::hasReduction() {
  FUSER_PERF_SCOPE("Fusion::hasReduction");

  for (auto expr : exprs(true))
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (out->as<TensorView>()->hasReduction())
          return true;

  return false;
}

bool Fusion::hasBlockReduction() {
  FUSER_PERF_SCOPE("Fusion::hasBlockReduction");

  for (auto expr : exprs(true))
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (out->as<TensorView>()->hasBlockReduction())
          return true;

  return false;
}

bool Fusion::hasGridReduction() {
  FUSER_PERF_SCOPE("Fusion::hasGridReduction");

  for (auto expr : exprs(true))
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (out->as<TensorView>()->hasGridReduction())
          return true;

  return false;
}

bool Fusion::hasBlockBroadcast() {
  for (auto expr : exprs(true)) {
    for (auto out : expr->outputs()) {
      if (out->getValType() == ValType::TensorView) {
        if (out->as<TensorView>()->hasBlockBroadcast()) {
          return true;
        }
      }
    }
  }
  return false;
}

bool Fusion::hasBroadcast() {
  for (auto expr : exprs(true))
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (out->as<TensorView>()->hasBroadcast())
          return true;

  return false;
}

std::vector<Val*> Fusion::getTerminatingOutputs() {
  FUSER_PERF_SCOPE("getTerminatingOutputs");

  FusionGuard fg(this);

  std::unordered_set<Val*> used_vals;

  const auto exprs = ExprSort::getExprs(
      this, std::vector<Val*>(outputs().begin(), outputs().end()));

  for (auto expr : exprs) {
    for (auto inp : expr->inputs())
      used_vals.emplace(inp);
  }

  std::vector<Val*> terminating_outputs;
  for (auto out : outputs()) {
    if (used_vals.find(out) != used_vals.end())
      continue;
    terminating_outputs.push_back(out);
  }
  return terminating_outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
