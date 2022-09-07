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
 * ptr(mutator)->mutate(this->as<Statement>());
 */

template <typename T>
void Val::dispatch(T handler, Val* val) {
  switch (*(val->getValType())) {
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
          return;
        case DataType::Int:
        case DataType::Int32:
          // Dispatch to Int even with Int32 as we don't have Int32 IR
          // node.
          ptr(handler)->handle(val->as<Int>());
          return;
        case DataType::ComplexDouble:
          ptr(handler)->handle(val->as<ComplexDouble>());
          return;
        default:
          break;
      }
      break;
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;

    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(handler)->handle(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::IntPair:
      ptr(handler)->handle(val->as<kir::IntPair>());
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::ARangeOp:
      ptr(handler)->handle(expr->as<ARangeOp>());
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
    case ExprType::RNGOp:
      ptr(handler)->handle(expr->as<RNGOp>());
      return;
    case ExprType::ReductionOp:
      ptr(handler)->handle(expr->as<ReductionOp>());
      return;
    case ExprType::GroupedReductionOp:
      ptr(handler)->handle(expr->as<GroupedReductionOp>());
      return;
    case ExprType::WelfordOp:
      ptr(handler)->handle(expr->as<WelfordOp>());
      return;
    case ExprType::GroupedWelfordOp:
      ptr(handler)->handle(expr->as<GroupedWelfordOp>());
      return;
    case ExprType::LoadStoreOp:
      ptr(handler)->handle(expr->as<LoadStoreOp>());
      return;
    case ExprType::MmaOp:
      ptr(handler)->handle(expr->as<MmaOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;

    case ExprType::Split:
      ptr(handler)->handle(expr->as<Split>());
      return;
    case ExprType::Merge:
      ptr(handler)->handle(expr->as<Merge>());
      return;
    case ExprType::Swizzle2D:
      ptr(handler)->handle(expr->as<Swizzle2D>());
      return;
    case ExprType::TransposeOp:
      ptr(handler)->handle(expr->as<TransposeOp>());
      return;
    case ExprType::ExpandOp:
      ptr(handler)->handle(expr->as<ExpandOp>());
      return;
    case ExprType::ShiftOp:
      ptr(handler)->handle(expr->as<ShiftOp>());
      return;
    case ExprType::GatherOp:
      ptr(handler)->handle(expr->as<GatherOp>());
      return;
    case ExprType::ViewAsScalar:
      ptr(handler)->handle(expr->as<ViewAsScalar>());
      return;
    case ExprType::ViewOp:
      ptr(handler)->handle(expr->as<ViewOp>());
      return;

    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<kir::Allocate>());
      return;
    case ExprType::BlockSync:
      ptr(handler)->handle(expr->as<kir::BlockSync>());
      return;
    case ExprType::GridSync:
      ptr(handler)->handle(expr->as<kir::GridSync>());
      return;
    case ExprType::CpAsyncWait:
      ptr(handler)->handle(expr->as<kir::CpAsyncWait>());
      return;
    case ExprType::CpAsyncCommit:
      ptr(handler)->handle(expr->as<kir::CpAsyncCommit>());
      return;
    case ExprType::InitMagicZero:
      ptr(handler)->handle(expr->as<kir::InitMagicZero>());
      return;
    case ExprType::UpdateMagicZero:
      ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<kir::ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<kir::IfThenElse>());
      return;
    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<kir::GridReduction>());
      return;
    case ExprType::GroupedGridReduction:
      ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
      return;
    case ExprType::GridBroadcast:
      ptr(handler)->handle(expr->as<kir::GridBroadcast>());
      return;
    case ExprType::GridWelford:
      ptr(handler)->handle(expr->as<kir::GridWelford>());
      return;
    case ExprType::GroupedGridWelford:
      ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
      return;
    case ExprType::AllocateFusedReduction:
      ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
      return;
    case ExprType::Swizzle2DInt:
      ptr(handler)->handle(expr->as<kir::Swizzle2DInt>());
      return;
    case ExprType::PairSelect:
      ptr(handler)->handle(expr->as<kir::PairSelect>());
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
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
          return;
        case DataType::Int:
        case DataType::Int32:
          // Dispatch to Int even with Int32 as we don't have Int32 IR
          // node.
          ptr(handler)->handle(val->as<Int>());
          return;
        case DataType::ComplexDouble:
          ptr(handler)->handle(val->as<ComplexDouble>());
          return;
        default:
          break;
      }
      break;
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;

    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(handler)->handle(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::IntPair:
      ptr(handler)->handle(val->as<kir::IntPair>());
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::ARangeOp:
      ptr(handler)->handle(expr->as<ARangeOp>());
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
    case ExprType::RNGOp:
      ptr(handler)->handle(expr->as<RNGOp>());
      return;
    case ExprType::ReductionOp:
      ptr(handler)->handle(expr->as<ReductionOp>());
      return;
    case ExprType::GroupedReductionOp:
      ptr(handler)->handle(expr->as<GroupedReductionOp>());
      return;
    case ExprType::WelfordOp:
      ptr(handler)->handle(expr->as<WelfordOp>());
      return;
    case ExprType::GroupedWelfordOp:
      ptr(handler)->handle(expr->as<GroupedWelfordOp>());
      return;
    case ExprType::LoadStoreOp:
      ptr(handler)->handle(expr->as<LoadStoreOp>());
      return;
    case ExprType::MmaOp:
      ptr(handler)->handle(expr->as<MmaOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;

    case ExprType::Split:
      ptr(handler)->handle(expr->as<Split>());
      return;
    case ExprType::Merge:
      ptr(handler)->handle(expr->as<Merge>());
      return;
    case ExprType::Swizzle2D:
      ptr(handler)->handle(expr->as<Swizzle2D>());
      return;
    case ExprType::TransposeOp:
      ptr(handler)->handle(expr->as<TransposeOp>());
      return;
    case ExprType::ExpandOp:
      ptr(handler)->handle(expr->as<ExpandOp>());
      return;
    case ExprType::ShiftOp:
      ptr(handler)->handle(expr->as<ShiftOp>());
      return;
    case ExprType::GatherOp:
      ptr(handler)->handle(expr->as<GatherOp>());
      return;
    case ExprType::ViewAsScalar:
      ptr(handler)->handle(expr->as<ViewAsScalar>());
      return;
    case ExprType::ViewOp:
      ptr(handler)->handle(expr->as<ViewOp>());
      return;

    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<kir::Allocate>());
      return;
    case ExprType::BlockSync:
      ptr(handler)->handle(expr->as<kir::BlockSync>());
      return;
    case ExprType::GridSync:
      ptr(handler)->handle(expr->as<kir::GridSync>());
      return;
    case ExprType::CpAsyncWait:
      ptr(handler)->handle(expr->as<kir::CpAsyncWait>());
      return;
    case ExprType::CpAsyncCommit:
      ptr(handler)->handle(expr->as<kir::CpAsyncCommit>());
      return;
    case ExprType::InitMagicZero:
      ptr(handler)->handle(expr->as<kir::InitMagicZero>());
      return;
    case ExprType::UpdateMagicZero:
      ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<kir::ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<kir::IfThenElse>());
      return;
    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<kir::GridReduction>());
      return;
    case ExprType::GroupedGridReduction:
      ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
      return;
    case ExprType::GridBroadcast:
      ptr(handler)->handle(expr->as<kir::GridBroadcast>());
      return;
    case ExprType::GridWelford:
      ptr(handler)->handle(expr->as<kir::GridWelford>());
      return;
    case ExprType::GroupedGridWelford:
      ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
      return;
    case ExprType::AllocateFusedReduction:
      ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
      return;
    case ExprType::Swizzle2DInt:
      ptr(handler)->handle(expr->as<kir::Swizzle2DInt>());
      return;
    case ExprType::PairSelect:
      ptr(handler)->handle(expr->as<kir::PairSelect>());
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
void Val::mutatorDispatch(T mutator, Val* val) {
  switch (*(val->getValType())) {
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Bool:
          ptr(mutator)->mutate(val->as<Bool>());
          return;
        case DataType::Double:
          ptr(mutator)->mutate(val->as<Double>());
          return;
        case DataType::Int:
          ptr(mutator)->mutate(val->as<Int>());
          return;
        case DataType::ComplexDouble:
          ptr(mutator)->mutate(val->as<ComplexDouble>());
          return;
        default:
          break;
      }
      break;
    case ValType::NamedScalar:
      ptr(mutator)->mutate(val->as<NamedScalar>());
      return;

    case ValType::IterDomain:
      ptr(mutator)->mutate(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(mutator)->mutate(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(mutator)->mutate(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(mutator)->mutate(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(mutator)->mutate(val->as<kir::TensorIndex>());
      return;
    case ValType::IntPair:
      ptr(mutator)->mutate(val->as<kir::IntPair>());
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::mutatorDispatch(T mutator, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::ARangeOp:
      ptr(mutator)->mutate(expr->as<ARangeOp>());
      return;
    case ExprType::UnaryOp:
      ptr(mutator)->mutate(expr->as<UnaryOp>());
      return;
    case ExprType::BinaryOp:
      ptr(mutator)->mutate(expr->as<BinaryOp>());
      return;
    case ExprType::TernaryOp:
      ptr(mutator)->mutate(expr->as<TernaryOp>());
      return;
    case ExprType::RNGOp:
      ptr(mutator)->mutate(expr->as<RNGOp>());
      return;
    case ExprType::ReductionOp:
      ptr(mutator)->mutate(expr->as<ReductionOp>());
      return;
    case ExprType::GroupedReductionOp:
      ptr(mutator)->mutate(expr->as<GroupedReductionOp>());
      return;
    case ExprType::WelfordOp:
      ptr(mutator)->mutate(expr->as<WelfordOp>());
      return;
    case ExprType::GroupedWelfordOp:
      ptr(mutator)->mutate(expr->as<GroupedWelfordOp>());
      return;
    case ExprType::LoadStoreOp:
      ptr(mutator)->mutate(expr->as<LoadStoreOp>());
      return;
    case ExprType::MmaOp:
      ptr(mutator)->mutate(expr->as<MmaOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(mutator)->mutate(expr->as<BroadcastOp>());
      return;

    case ExprType::Split:
      ptr(mutator)->mutate(expr->as<Split>());
      return;
    case ExprType::Merge:
      ptr(mutator)->mutate(expr->as<Merge>());
      return;
    case ExprType::Swizzle2D:
      ptr(mutator)->mutate(expr->as<Swizzle2D>());
      return;
    case ExprType::TransposeOp:
      ptr(mutator)->mutate(expr->as<TransposeOp>());
      return;
    case ExprType::ExpandOp:
      ptr(mutator)->mutate(expr->as<ExpandOp>());
      return;
    case ExprType::ShiftOp:
      ptr(mutator)->mutate(expr->as<ShiftOp>());
      return;
    case ExprType::GatherOp:
      ptr(mutator)->mutate(expr->as<GatherOp>());
      return;
    case ExprType::ViewAsScalar:
      ptr(mutator)->mutate(expr->as<ViewAsScalar>());
      return;
    case ExprType::ViewOp:
      ptr(mutator)->mutate(expr->as<ViewOp>());
      return;

    case ExprType::Allocate:
      ptr(mutator)->mutate(expr->as<kir::Allocate>());
      return;
    case ExprType::BlockSync:
      ptr(mutator)->mutate(expr->as<kir::BlockSync>());
      return;
    case ExprType::GridSync:
      ptr(mutator)->mutate(expr->as<kir::GridSync>());
      return;
    case ExprType::CpAsyncWait:
      ptr(mutator)->mutate(expr->as<kir::CpAsyncWait>());
      return;
    case ExprType::CpAsyncCommit:
      ptr(mutator)->mutate(expr->as<kir::CpAsyncCommit>());
      return;
    case ExprType::InitMagicZero:
      ptr(mutator)->mutate(expr->as<kir::InitMagicZero>());
      return;
    case ExprType::UpdateMagicZero:
      ptr(mutator)->mutate(expr->as<kir::UpdateMagicZero>());
      return;
    case ExprType::ForLoop:
      ptr(mutator)->mutate(expr->as<kir::ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(mutator)->mutate(expr->as<kir::IfThenElse>());
      return;
    case ExprType::GridReduction:
      ptr(mutator)->mutate(expr->as<kir::GridReduction>());
      return;
    case ExprType::GroupedGridReduction:
      ptr(mutator)->mutate(expr->as<kir::GroupedGridReduction>());
      return;
    case ExprType::GridBroadcast:
      ptr(mutator)->mutate(expr->as<kir::GridBroadcast>());
      return;
    case ExprType::GridWelford:
      ptr(mutator)->mutate(expr->as<kir::GridWelford>());
      return;
    case ExprType::GroupedGridWelford:
      ptr(mutator)->mutate(expr->as<kir::GroupedGridWelford>());
      return;
    case ExprType::AllocateFusedReduction:
      ptr(mutator)->mutate(expr->as<kir::AllocateFusedReduction>());
      return;
    case ExprType::Swizzle2DInt:
      ptr(mutator)->mutate(expr->as<kir::Swizzle2DInt>());
      return;
    case ExprType::PairSelect:
      ptr(mutator)->mutate(expr->as<kir::PairSelect>());
      return;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Statement::mutatorDispatch(T mutator, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(mutator)->mutate(stmt->as<Val>());
    return;
  }
  if (stmt->isExpr()) {
    ptr(mutator)->mutate(stmt->as<Expr>());
    return;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

/*
 * Handler template instantiations. These should only have to be done on base
 * classes. Actual visitors/mutators should inhereit from these classes and call
 * ->dispatch(this) to avoid needing an explicit instantiation.
 */
template void Statement::dispatch(OptOutDispatch&, Statement*);
template void Statement::dispatch(OptOutDispatch*, Statement*);
template void Val::dispatch(OptOutDispatch&, Val*);
template void Val::dispatch(OptOutDispatch*, Val*);
template void Expr::dispatch(OptOutDispatch&, Expr*);
template void Expr::dispatch(OptOutDispatch*, Expr*);

template void Statement::dispatch(OptInDispatch, Statement*);
template void Statement::dispatch(OptInDispatch*, Statement*);
template void Val::dispatch(OptInDispatch, Val*);
template void Val::dispatch(OptInDispatch*, Val*);
template void Expr::dispatch(OptInDispatch, Expr*);
template void Expr::dispatch(OptInDispatch*, Expr*);

template void Statement::constDispatch(OptOutConstDispatch&, const Statement*);
template void Statement::constDispatch(OptOutConstDispatch*, const Statement*);
template void Val::constDispatch(OptOutConstDispatch&, const Val*);
template void Val::constDispatch(OptOutConstDispatch*, const Val*);
template void Expr::constDispatch(OptOutConstDispatch&, const Expr*);
template void Expr::constDispatch(OptOutConstDispatch*, const Expr*);

template void Statement::constDispatch(OptInConstDispatch&, const Statement*);
template void Statement::constDispatch(OptInConstDispatch*, const Statement*);
template void Val::constDispatch(OptInConstDispatch&, const Val*);
template void Val::constDispatch(OptInConstDispatch*, const Val*);
template void Expr::constDispatch(OptInConstDispatch&, const Expr*);
template void Expr::constDispatch(OptInConstDispatch*, const Expr*);

template void Statement::mutatorDispatch(OptOutMutator&, Statement*);
template void Statement::mutatorDispatch(OptOutMutator*, Statement*);
template void Val::mutatorDispatch(OptOutMutator&, Val*);
template void Val::mutatorDispatch(OptOutMutator*, Val*);
template void Expr::mutatorDispatch(OptOutMutator&, Expr*);
template void Expr::mutatorDispatch(OptOutMutator*, Expr*);

void OptOutDispatch::handle(Statement* s) {
  Statement::dispatch(this, s);
}

void OptOutDispatch::handle(Expr* e) {
  Expr::dispatch(this, e);
}

void OptOutDispatch::handle(Val* v) {
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

void OptInConstDispatch::unhandled(const Statement* stmt) {
  if (stmt->isExpr()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getExprType().value(), ".");
  } else if (stmt->isVal()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unrecognized statement type.");
  }
}

void OptInDispatch::unhandled(Statement* stmt) {
  if (stmt->isExpr()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getExprType().value(), ".");
  } else if (stmt->isVal()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unrecognized statement type.");
  }
}

// Vals
void OptOutConstDispatch::handle(const Bool* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Double* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Int* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ComplexDouble* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorView* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const kir::Predicate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::TensorIndex* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::IntPair* stmt) {
  unhandled(stmt);
}

// Exprs
void OptOutConstDispatch::handle(const ARangeOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const UnaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const BinaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TernaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const RNGOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GroupedReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GroupedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const LoadStoreOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const MmaOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const BroadcastOp* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const Split* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Merge* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Swizzle2D* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TransposeOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ExpandOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ShiftOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GatherOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ViewAsScalar* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ViewOp* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const kir::Allocate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::BlockSync* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridSync* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::CpAsyncWait* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::CpAsyncCommit* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GroupedGridReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridWelford* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GroupedGridWelford* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::Swizzle2DInt* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::PairSelect* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::unhandled(Statement*) {}

// Vals
void OptOutDispatch::handle(Bool* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Double* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Int* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ComplexDouble* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorView* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(kir::Predicate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::TensorIndex* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::IntPair* stmt) {
  unhandled(stmt);
}

// Exprs
void OptOutDispatch::handle(ARangeOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(UnaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(BinaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TernaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(RNGOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GroupedReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GroupedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(LoadStoreOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(MmaOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(BroadcastOp* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(Split* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Merge* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Swizzle2D* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TransposeOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ExpandOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ShiftOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GatherOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ViewAsScalar* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ViewOp* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(kir::Allocate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::BlockSync* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridSync* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::CpAsyncWait* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::CpAsyncCommit* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GroupedGridReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridWelford* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GroupedGridWelford* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::Swizzle2DInt* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::PairSelect* stmt) {
  unhandled(stmt);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
