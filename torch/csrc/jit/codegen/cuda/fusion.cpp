#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>

namespace torch {
namespace jit {
namespace fuser {

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

void ExprSort::handle(Expr* expr) {
  exprs.push_back(expr);
}

std::vector<Expr*> ExprSort::getExprs(
    Fusion* fusion,
    bool from_outputs_only,
    bool breadth_first) {
  ExprSort es;
  es.traverse(fusion, from_outputs_only, breadth_first);
  return es.exprs;
}

void InputsOf::handle(Val* v) {
  if (FusionGuard::getCurFusion()->origin(v) == nullptr)
    inputs.emplace(v);
}

std::set<Val*> InputsOf::output(Fusion* fusion, Val* output_) {
  TORCH_CHECK(
      fusion->hasOutput(output_),
      "Asked for the inputs of ",
      output_,
      " however, it is not an output of the provided fusion.");
  InputsOf io;
  io.traverseFrom(FusionGuard::getCurFusion(), {output_}, false);
  return io.inputs;
}

Fusion::~Fusion() {
  {
    auto it = val_set_.begin();
    while (it != val_set_.end()) {
      auto del = it;
      it = ++it;
      delete (*del);
    }
  }
  auto it = expr_set_.begin();
  while (it != expr_set_.end()) {
    auto del = it;
    it = ++it;
    delete (*del);
  }
};

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

  for (Expr* use : uses(val))
    removeExpr(use);

  val_set_.erase(val);

  for (auto it = val_deque_.begin(); it != val_deque_.end(); it++)
    if (*it == val) {
      val_deque_.erase(it);
      break;
    }

  delete val;
}

void Fusion::addInput(Val* const input) {
  assertInFusion(input, "Cannot register input ");
  IRInputOutput::addInput(input);
}

void Fusion::addOutput(Val* const output) {
  assertInFusion(output, "Cannot register output ");
  IRInputOutput::addOutput(output);
}

bool Fusion::inFusion(const Statement* stmt) const {
  bool infusion = stmt->fusion() == this;
  Statement* nonconst_stmt = const_cast<Statement*>(stmt);

  if (stmt->isExpr())
    infusion &=
        expr_set_.find(static_cast<Expr*>(nonconst_stmt)) != expr_set_.end();
  if (stmt->isVal())
    infusion &=
        val_set_.find(static_cast<Val*>(nonconst_stmt)) != val_set_.end();

  return infusion;
}

void Fusion::assertInFusion(const Statement* stmt, const std::string& msg)
    const {
  if (inFusion(stmt))
    return;
  TORCH_CHECK(false, msg, " it was not found in the active fusion.");
}

std::vector<Expr*> Fusion::exprs(bool from_outputs_only, bool breadth_first) {
  if (breadth_first)
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
  return ExprSort::getExprs(this, from_outputs_only, breadth_first);
}

std::set<Val*> Fusion::inputsOf(Val* val) {
  return InputsOf::output(this, val);
}

void Fusion::validateInputs() {
  std::set<Val*> all_inputs;
  for (Val* out : outputs()) {
    auto outs_inputs = inputsOf(out);
    std::set_union(
        all_inputs.begin(),
        all_inputs.end(),
        outs_inputs.begin(),
        outs_inputs.end(),
        std::inserter(all_inputs, all_inputs.begin()));
  }
  for (Val* inp : all_inputs) {
    if (!inp->isConstScalar())
      TORCH_CHECK(
          hasInput(inp),
          "Could not figure out how ",
          inp,
          " is generated, however it was not specified as an input.");
  }
}

void Fusion::print() {
  FusionGuard fg(this);
  std::cout << "%kernel {\n";
  IRMathPrinter op_exprs(std::cout);
  op_exprs.handle(this);
  IRTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
  std::cout << "}\n";
}

void Fusion::printMath() {
  FusionGuard fg(this);
  IRMathPrinter op_exprs(std::cout);
  op_exprs.handle(this);
}

void Fusion::printTransforms() {
  FusionGuard fg(this);
  IRTransformPrinter t_exprs(std::cout);
  t_exprs.handle(this);
}

StmtNameType Fusion::registerVal(Val* val) {
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
  if (expr->fusion()) {
    if (expr->fusion() != this) {
      TORCH_CHECK(false, expr, " was not found in the active fusion.");
    }
    if (inFusion(expr)) {
      return expr->name();
    }
  }

  for (Val* input : expr->inputs()) {
    registerVal(input);
    if (uses_.find(input) == uses_.end()) {
      uses_[input] = {expr};
    } else {
      uses_.find(input)->second.emplace(expr);
    }
  }

  for (Val* output : expr->outputs()) {
    registerVal(output);
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
    return registerVal(static_cast<Val*>(stmt));
  } else if (stmt->isExpr()) {
    return registerExpr(static_cast<Expr*>(stmt));
  }

  TORCH_INTERNAL_ASSERT(
      false,
      "Could not register statement as Fusion could not recognize its type.");
  return UNINITIALIZED_STMTNAMETYPE;
}

bool Fusion::used(Val* val) const {
  assertInFusion(val, "Cannot detect if val was used, ");
  return (uses_.find(val) != uses_.end()) &&
      (uses_.find(val)->second.size() > 0);
}

const std::set<Val*>& Fusion::vals() const noexcept {
  return val_set_;
}

const std::deque<Val*>& Fusion::deterministic_vals() const noexcept {
  return val_deque_;
}

const std::set<Expr*>& Fusion::unordered_exprs() const noexcept {
  return expr_set_;
}

std::set<Expr*> Fusion::uses(Val* val) const {
  assertInFusion(val, "Cannot detect where val was used, ");
  if (uses_.find(val) != uses_.end()) {
    auto ret = uses_.find(val)->second;
    return ret;
  }
  return std::set<Expr*>();
}

Expr* Fusion::origin(Val* val) const {
  assertInFusion(val, "Cannot dettect the origin of val, ");
  auto it = origin_.find(val);

  if (it == origin_.end())
    return nullptr;

  return it->second;
}

const Expr* Fusion::origin(const Val* val) const {
  assertInFusion(val, "Cannot dettect the origin of val, ");
  auto it = origin_.find(const_cast<Val*>(val));
  if (it == origin_.end())
    return nullptr;
  return it->second;
}

StmtNameType Fusion::getValName(ValType vtype) {
  if (val_type_name_map.find(vtype) != val_type_name_map.end())
    return val_type_name_map[vtype]++;
  return val_name_counter_++;
}
StmtNameType Fusion::getExprName() {
  return expr_name_counter_++;
}

// Indicate to kernel to set itself up to generate random numbers
bool Fusion::hasRNG() {
  for (auto expr : exprs(true))
    if (expr->getExprType() == ExprType::UnaryOp)
      if (static_cast<UnaryOp*>(expr)->getUnaryOpType() ==
          UnaryOpType::RandLike)
        return true;
  return false;
}

// Indicate to kernel to set itself up to generate random numbers
bool Fusion::hasReduction() {
  for (auto expr : exprs(true))
    for (auto out : expr->outputs())
      if (out->getValType() == ValType::TensorView)
        if (static_cast<TensorView*>(out)->hasReduction())
          return true;

  return false;
}

} // namespace fuser
} // namespace jit
} // namespace torch
