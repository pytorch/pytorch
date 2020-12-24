#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

template <typename T>
T* ptr(T& obj) {
  return &obj;
}

template <typename T>
T* ptr(T* obj) {
  return obj;
}

/*
 * Generic dispatch for any handler that does not modify the IR directly.
 * For example we may want to walk the graph to construct a topologically sorted
 * set of exprs. This doesn't modify the IR directly. We also use this to print
 * the IR itself.
 * This dispatch is paired with a class that implements the functions:
 * template <typenname node_type>
 * int handler(node_type* node)
 *
 * handler should call:
 * dispatch(this, node_to_dispatch)
 *
 * It could also implement:
 * int handler(Statement* stmt){
 *   dispatch(this, stmt);
 * }
 *
 * And therefore dispatch should never call:
 * ptr(mutator)->handle(this->as<Statement>());
 */

template <typename T>
void Val::dispatch(T handler, Val* val) {
  switch (*(val->getValType())) {
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Float:
          ptr(handler)->handle(val->as<Float>());
          return;
        case DataType::Half:
          ptr(handler)->handle(val->as<Half>());
          return;
        case DataType::Int:
          ptr(handler)->handle(val->as<Int>());
          return;
        default:
          break;
      }
      break;
    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;

    // TODO: remove once the Kernel IR has its own visitor
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::KirScalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<kir::Bool>());
          return;
        case DataType::Float:
          ptr(handler)->handle(val->as<kir::Float>());
          return;
        case DataType::Half:
          ptr(handler)->handle(val->as<kir::Half>());
          return;
        case DataType::Int:
          ptr(handler)->handle(val->as<kir::Int>());
          return;
        default:
          break;
      }
      break;
    case ValType::KirNamedScalar:
      ptr(handler)->handle(val->as<kir::NamedScalar>());
      return;
    case ValType::KirIterDomain:
      ptr(handler)->handle(val->as<kir::IterDomain>());
      return;
    case ValType::KirTensorDomain:
      ptr(handler)->handle(val->as<kir::TensorDomain>());
      return;
    case ValType::KirTensorView:
      ptr(handler)->handle(val->as<kir::TensorView>());
      return;

    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::Split:
      ptr(handler)->handle(expr->as<Split>());
      return;
    case ExprType::Merge:
      ptr(handler)->handle(expr->as<Merge>());
      return;
    case ExprType::UnaryOp:
      ptr(handler)->handle(expr->as<UnaryOp>());
      return;
    case ExprType::BinaryOp:
      ptr(handler)->handle(expr->as<BinaryOp>());
      return;
    case ExprType::TernaryOp:
      ptr(handler)->handle(expr->as<TernaryOp>());
      return;
    case ExprType::ReductionOp:
      ptr(handler)->handle(expr->as<ReductionOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;

    case ExprType::KirUnaryOp:
      ptr(handler)->handle(expr->as<kir::UnaryOp>());
      return;
    case ExprType::KirBinaryOp:
      ptr(handler)->handle(expr->as<kir::BinaryOp>());
      return;
    case ExprType::KirTernaryOp:
      ptr(handler)->handle(expr->as<kir::TernaryOp>());
      return;
    case ExprType::KirReductionOp:
      ptr(handler)->handle(expr->as<kir::ReductionOp>());
      return;
    case ExprType::KirBroadcastOp:
      ptr(handler)->handle(expr->as<kir::BroadcastOp>());
      return;

    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<kir::GridReduction>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<kir::ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<kir::IfThenElse>());
      return;
    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<kir::Allocate>());
      return;
    case ExprType::Sync:
      ptr(handler)->handle(expr->as<kir::Sync>());
      return;

    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Statement::dispatch(T handler, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->handle(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->handle(stmt->as<Expr>());
  } else
    TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

template <typename T>
void Val::constDispatch(T handler, const Val* val) {
  switch (*(val->getValType())) {
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Float:
          ptr(handler)->handle(val->as<Float>());
          return;
        case DataType::Half:
          ptr(handler)->handle(val->as<Half>());
          return;
        case DataType::Int:
          ptr(handler)->handle(val->as<Int>());
          return;
        default:
          break;
      }
      break;
    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;

    // TODO: remove once the Kernel IR has its own visitor
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::KirScalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<kir::Bool>());
          return;
        case DataType::Float:
          ptr(handler)->handle(val->as<kir::Float>());
          return;
        case DataType::Half:
          ptr(handler)->handle(val->as<kir::Half>());
          return;
        case DataType::Int:
          ptr(handler)->handle(val->as<kir::Int>());
          return;
        default:
          break;
      }
      break;
    case ValType::KirNamedScalar:
      ptr(handler)->handle(val->as<kir::NamedScalar>());
      return;
    case ValType::KirIterDomain:
      ptr(handler)->handle(val->as<kir::IterDomain>());
      return;
    case ValType::KirTensorDomain:
      ptr(handler)->handle(val->as<kir::TensorDomain>());
      return;
    case ValType::KirTensorView:
      ptr(handler)->handle(val->as<kir::TensorView>());
      return;

    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::Split:
      ptr(handler)->handle(expr->as<Split>());
      return;
    case ExprType::Merge:
      ptr(handler)->handle(expr->as<Merge>());
      return;
    case ExprType::UnaryOp:
      ptr(handler)->handle(expr->as<UnaryOp>());
      return;
    case ExprType::BinaryOp:
      ptr(handler)->handle(expr->as<BinaryOp>());
      return;
    case ExprType::TernaryOp:
      ptr(handler)->handle(expr->as<TernaryOp>());
      return;
    case ExprType::ReductionOp:
      ptr(handler)->handle(expr->as<ReductionOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;

    case ExprType::KirUnaryOp:
      ptr(handler)->handle(expr->as<kir::UnaryOp>());
      return;
    case ExprType::KirBinaryOp:
      ptr(handler)->handle(expr->as<kir::BinaryOp>());
      return;
    case ExprType::KirTernaryOp:
      ptr(handler)->handle(expr->as<kir::TernaryOp>());
      return;
    case ExprType::KirReductionOp:
      ptr(handler)->handle(expr->as<kir::ReductionOp>());
      return;
    case ExprType::KirBroadcastOp:
      ptr(handler)->handle(expr->as<kir::BroadcastOp>());
      return;

    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<kir::GridReduction>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<kir::ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<kir::IfThenElse>());
      return;
    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<kir::Allocate>());
      return;
    case ExprType::Sync:
      ptr(handler)->handle(expr->as<kir::Sync>());
      return;

    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Statement::constDispatch(T handler, const Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->handle(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->handle(stmt->as<Expr>());
  } else
    TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

/*
 * Generic mutatorDispatch for any handler that modifies the IR. This could be
 * a transformation on loop structures, or parallelizing a loop. This
 * mutatorDispatch is paired with a class that implements the functions
 * template <typenname node_type> Statement* mutate(node_type* node) mutate
 * should call (statement* node_to_dispatch)->mutatorDispatch() It could also
 * implement Statement* mutate(Statement* stmt){ stmt->mutatorDispatch(this);
 * }
 * And therefore dispatch should never call:
 *   ptr(mutator)->mutate(this->as<Statement>());
 */
template <typename T>
Statement* Val::mutatorDispatch(T mutator, Val* val) {
  switch (*(val->getValType())) {
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          return ptr(mutator)->mutate(val->as<Bool>());
        case DataType::Float:
          return ptr(mutator)->mutate(val->as<Float>());
        case DataType::Half:
          return ptr(mutator)->mutate(val->as<Half>());
        case DataType::Int:
          return ptr(mutator)->mutate(val->as<Int>());
        default:
          break;
      }
      break;
    case ValType::IterDomain:
      return ptr(mutator)->mutate(val->as<IterDomain>());
    case ValType::TensorDomain:
      return ptr(mutator)->mutate(val->as<TensorDomain>());
    case ValType::TensorView:
      return ptr(mutator)->mutate(val->as<TensorView>());
    case ValType::TensorIndex:
      return ptr(mutator)->mutate(val->as<kir::TensorIndex>());
    case ValType::NamedScalar:
      return ptr(mutator)->mutate(val->as<NamedScalar>());
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
Statement* Expr::mutatorDispatch(T mutator, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::Split:
      return ptr(mutator)->mutate(expr->as<Split>());
    case ExprType::Merge:
      return ptr(mutator)->mutate(expr->as<Merge>());
    case ExprType::UnaryOp:
      return ptr(mutator)->mutate(expr->as<UnaryOp>());
    case ExprType::BinaryOp:
      return ptr(mutator)->mutate(expr->as<BinaryOp>());
    case ExprType::TernaryOp:
      return ptr(mutator)->mutate(expr->as<TernaryOp>());
    case ExprType::ReductionOp:
      return ptr(mutator)->mutate(expr->as<ReductionOp>());
    case ExprType::GridReduction:
      return ptr(mutator)->mutate(expr->as<kir::GridReduction>());
    case ExprType::BroadcastOp:
      return ptr(mutator)->mutate(expr->as<BroadcastOp>());
    case ExprType::ForLoop:
      return ptr(mutator)->mutate(expr->as<kir::ForLoop>());
    case ExprType::IfThenElse:
      return ptr(mutator)->mutate(expr->as<kir::IfThenElse>());
    case ExprType::Allocate:
      return ptr(mutator)->mutate(expr->as<kir::Allocate>());
    case ExprType::Sync:
      return ptr(mutator)->mutate(expr->as<kir::Sync>());
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
Statement* Statement::mutatorDispatch(T mutator, Statement* stmt) {
  if (stmt->isVal()) {
    return ptr(mutator)->mutate(stmt->as<Val>());
  }
  if (stmt->isExpr()) {
    return ptr(mutator)->mutate(stmt->as<Expr>());
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

/*
 * Handler template instantiations. These should only have to be done on base
 * classes. Actual visitors/mutators should inhereit from these classes and call
 * ->dispatch(this) to avoid needing an explicit instantiation.
 */
template void Statement::dispatch(OptOutDispatch, Statement*);
template void Statement::dispatch(OptOutDispatch*, Statement*);
template void Val::dispatch(OptOutDispatch, Val*);
template void Val::dispatch(OptOutDispatch*, Val*);
template void Expr::dispatch(OptOutDispatch, Expr*);
template void Expr::dispatch(OptOutDispatch*, Expr*);

template void Statement::dispatch(OptInDispatch, Statement*);
template void Statement::dispatch(OptInDispatch*, Statement*);
template void Val::dispatch(OptInDispatch, Val*);
template void Val::dispatch(OptInDispatch*, Val*);
template void Expr::dispatch(OptInDispatch, Expr*);
template void Expr::dispatch(OptInDispatch*, Expr*);

template void Statement::constDispatch(OptOutConstDispatch, const Statement*);
template void Statement::constDispatch(OptOutConstDispatch*, const Statement*);
template void Val::constDispatch(OptOutConstDispatch, const Val*);
template void Val::constDispatch(OptOutConstDispatch*, const Val*);
template void Expr::constDispatch(OptOutConstDispatch, const Expr*);
template void Expr::constDispatch(OptOutConstDispatch*, const Expr*);

template void Statement::constDispatch(OptInConstDispatch, const Statement*);
template void Statement::constDispatch(OptInConstDispatch*, const Statement*);
template void Val::constDispatch(OptInConstDispatch, const Val*);
template void Val::constDispatch(OptInConstDispatch*, const Val*);
template void Expr::constDispatch(OptInConstDispatch, const Expr*);
template void Expr::constDispatch(OptInConstDispatch*, const Expr*);

template Statement* Statement::mutatorDispatch(OptOutMutator, Statement*);
template Statement* Statement::mutatorDispatch(OptOutMutator*, Statement*);
template Statement* Val::mutatorDispatch(OptOutMutator, Val*);
template Statement* Val::mutatorDispatch(OptOutMutator*, Val*);
template Statement* Expr::mutatorDispatch(OptOutMutator, Expr*);
template Statement* Expr::mutatorDispatch(OptOutMutator*, Expr*);

template Statement* Statement::mutatorDispatch(OptInMutator, Statement*);
template Statement* Statement::mutatorDispatch(OptInMutator*, Statement*);
template Statement* Val::mutatorDispatch(OptInMutator, Val*);
template Statement* Val::mutatorDispatch(OptInMutator*, Val*);
template Statement* Expr::mutatorDispatch(OptInMutator, Expr*);
template Statement* Expr::mutatorDispatch(OptInMutator*, Expr*);

void OptOutDispatch::handle(Statement* s) {
  Statement::dispatch(this, s);
}

void OptOutDispatch::handle(Expr* e) {
  Expr::dispatch(this, e);
}

void OptOutDispatch::handle(Val* v) {
  Val::dispatch(this, v);
}

void OptInDispatch::handle(Statement* s) {
  Statement::dispatch(this, s);
}

void OptInDispatch::handle(Expr* e) {
  Expr::dispatch(this, e);
}

void OptInDispatch::handle(Val* v) {
  Val::dispatch(this, v);
}

void OptOutConstDispatch::handle(const Statement* s) {
  Statement::constDispatch(this, s);
}

void OptOutConstDispatch::handle(const Expr* e) {
  Expr::constDispatch(this, e);
}

void OptOutConstDispatch::handle(const Val* v) {
  Val::constDispatch(this, v);
}

void OptInConstDispatch::handle(const Statement* s) {
  Statement::constDispatch(this, s);
}

void OptInConstDispatch::handle(const Expr* e) {
  Expr::constDispatch(this, e);
}

void OptInConstDispatch::handle(const Val* v) {
  Val::constDispatch(this, v);
}

Statement* OptInMutator::mutate(Statement* s) {
  return Statement::mutatorDispatch(this, s);
}

Statement* OptInMutator::mutate(Expr* e) {
  return Expr::mutatorDispatch(this, e);
}

Statement* OptInMutator::mutate(Val* v) {
  // If value is already mutated, return the mutation
  if (mutations.find(v) != mutations.end())
    return mutations[v];
  return Val::mutatorDispatch(this, v);
}

Statement* OptOutMutator::mutate(Statement* s) {
  return Statement::mutatorDispatch(this, s);
}

Statement* OptOutMutator::mutate(Expr* e) {
  return Expr::mutatorDispatch(this, e);
}

Statement* OptOutMutator::mutate(Val* v) {
  // If value is already mutated, return the mutation
  if (mutations.find(v) != mutations.end())
    return mutations[v];
  return Val::mutatorDispatch(this, v);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
