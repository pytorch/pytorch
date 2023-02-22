#include <fusion.h>
#include <ir_all_nodes.h>
#include <type.h>

#include <dispatch.h>

namespace nvfuser {

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
      switch (std::get<PrimDataType>(val->getDataType()->type)) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Float:
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
          return;
        case DataType::Int:
        case DataType::Int32:
        case DataType::Index:
        case DataType::SMemAddress:
          // Dispatch to Int even with Int32 as we don't have Int32 IR
          // node.
          ptr(handler)->handle(val->as<Int>());
          return;
        case DataType::ComplexFloat:
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
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  if (expr->isStrictlyA<FullOp>()) {
    ptr(handler)->handle(expr->as<FullOp>());
    return;
  }
  if (expr->isStrictlyA<IotaOp>()) {
    ptr(handler)->handle(expr->as<IotaOp>());
    return;
  }
  if (expr->isStrictlyA<EyeOp>()) {
    ptr(handler)->handle(expr->as<EyeOp>());
    return;
  }
  if (expr->isStrictlyA<UnaryOp>()) {
    ptr(handler)->handle(expr->as<UnaryOp>());
    return;
  }
  if (expr->isStrictlyA<BinaryOp>()) {
    ptr(handler)->handle(expr->as<BinaryOp>());
    return;
  }
  if (expr->isStrictlyA<TernaryOp>()) {
    ptr(handler)->handle(expr->as<TernaryOp>());
    return;
  }
  if (expr->isStrictlyA<SelectOp>()) {
    ptr(handler)->handle(expr->as<SelectOp>());
    return;
  }
  if (expr->isStrictlyA<IndexSelectOp>()) {
    ptr(handler)->handle(expr->as<IndexSelectOp>());
    return;
  }
  if (expr->isStrictlyA<TorchGatherOp>()) {
    ptr(handler)->handle(expr->as<TorchGatherOp>());
    return;
  }
  if (expr->isStrictlyA<ScatterOp>()) {
    ptr(handler)->handle(expr->as<ScatterOp>());
    return;
  }
  if (expr->isStrictlyA<RNGOp>()) {
    ptr(handler)->handle(expr->as<RNGOp>());
    return;
  }
  if (expr->isStrictlyA<ReductionOp>()) {
    ptr(handler)->handle(expr->as<ReductionOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedReductionOp>()) {
    ptr(handler)->handle(expr->as<GroupedReductionOp>());
    return;
  }
  if (expr->isStrictlyA<WelfordOp>()) {
    ptr(handler)->handle(expr->as<WelfordOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedWelfordOp>()) {
    ptr(handler)->handle(expr->as<GroupedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<LoadStoreOp>()) {
    ptr(handler)->handle(expr->as<LoadStoreOp>());
    return;
  }
  if (expr->isStrictlyA<MmaOp>()) {
    ptr(handler)->handle(expr->as<MmaOp>());
    return;
  }
  if (expr->isStrictlyA<BroadcastOp>()) {
    ptr(handler)->handle(expr->as<BroadcastOp>());
    return;
  }
  if (expr->isStrictlyA<SqueezeOp>()) {
    ptr(handler)->handle(expr->as<SqueezeOp>());
    return;
  }
  if (expr->isStrictlyA<Split>()) {
    ptr(handler)->handle(expr->as<Split>());
    return;
  }
  if (expr->isStrictlyA<Merge>()) {
    ptr(handler)->handle(expr->as<Merge>());
    return;
  }
  if (expr->isStrictlyA<Swizzle2D>()) {
    ptr(handler)->handle(expr->as<Swizzle2D>());
    return;
  }
  if (expr->isStrictlyA<TransposeOp>()) {
    ptr(handler)->handle(expr->as<TransposeOp>());
    return;
  }
  if (expr->isStrictlyA<ExpandOp>()) {
    ptr(handler)->handle(expr->as<ExpandOp>());
    return;
  }
  if (expr->isStrictlyA<ShiftOp>()) {
    ptr(handler)->handle(expr->as<ShiftOp>());
    return;
  }
  if (expr->isStrictlyA<GatherOp>()) {
    ptr(handler)->handle(expr->as<GatherOp>());
    return;
  }
  if (expr->isStrictlyA<ViewAsScalar>()) {
    ptr(handler)->handle(expr->as<ViewAsScalar>());
    return;
  }
  if (expr->isStrictlyA<ViewOp>()) {
    ptr(handler)->handle(expr->as<ViewOp>());
    return;
  }
  if (expr->isStrictlyA<kir::Allocate>()) {
    ptr(handler)->handle(expr->as<kir::Allocate>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSync>()) {
    ptr(handler)->handle(expr->as<kir::BlockSync>());
    return;
  }
  if (expr->isStrictlyA<kir::GridSync>()) {
    ptr(handler)->handle(expr->as<kir::GridSync>());
    return;
  }
  if (expr->isStrictlyA<kir::CpAsyncWait>()) {
    ptr(handler)->handle(expr->as<kir::CpAsyncWait>());
    return;
  }
  if (expr->isStrictlyA<kir::CpAsyncCommit>()) {
    ptr(handler)->handle(expr->as<kir::CpAsyncCommit>());
    return;
  }
  if (expr->isStrictlyA<kir::InitMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::InitMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::UpdateMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::ForLoop>()) {
    ptr(handler)->handle(expr->as<kir::ForLoop>());
    return;
  }
  if (expr->isStrictlyA<kir::IfThenElse>()) {
    ptr(handler)->handle(expr->as<kir::IfThenElse>());
    return;
  }
  if (expr->isStrictlyA<kir::GridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GridBroadcast>()) {
    ptr(handler)->handle(expr->as<kir::GridBroadcast>());
    return;
  }
  if (expr->isStrictlyA<kir::GridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::VectorizedWelfordOp>()) {
    ptr(handler)->handle(expr->as<kir::VectorizedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<kir::AllocateFusedReduction>()) {
    ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::SMemAddress>()) {
    ptr(handler)->handle(expr->as<kir::SMemAddress>());
    return;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
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
      switch (std::get<PrimDataType>(val->getDataType()->type)) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Float:
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
          return;
        case DataType::Int:
        case DataType::Index:
        case DataType::Int32:
        case DataType::SMemAddress:
          // Dispatch to Int even with Int32 as we don't have Int32 IR
          // node.
          ptr(handler)->handle(val->as<Int>());
          return;
        case DataType::ComplexFloat:
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
    case ValType::Attribute:
      // Attribute Val is just a wrapper for non-IR data, so there is nothing to
      // handle
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
  if (expr->isStrictlyA<FullOp>()) {
    ptr(handler)->handle(expr->as<FullOp>());
    return;
  }
  if (expr->isStrictlyA<IotaOp>()) {
    ptr(handler)->handle(expr->as<IotaOp>());
    return;
  }
  if (expr->isStrictlyA<EyeOp>()) {
    ptr(handler)->handle(expr->as<EyeOp>());
    return;
  }
  if (expr->isStrictlyA<UnaryOp>()) {
    ptr(handler)->handle(expr->as<UnaryOp>());
    return;
  }
  if (expr->isStrictlyA<BinaryOp>()) {
    ptr(handler)->handle(expr->as<BinaryOp>());
    return;
  }
  if (expr->isStrictlyA<TernaryOp>()) {
    ptr(handler)->handle(expr->as<TernaryOp>());
    return;
  }
  if (expr->isStrictlyA<SelectOp>()) {
    ptr(handler)->handle(expr->as<SelectOp>());
    return;
  }
  if (expr->isStrictlyA<IndexSelectOp>()) {
    ptr(handler)->handle(expr->as<IndexSelectOp>());
    return;
  }
  if (expr->isStrictlyA<TorchGatherOp>()) {
    ptr(handler)->handle(expr->as<TorchGatherOp>());
    return;
  }
  if (expr->isStrictlyA<ScatterOp>()) {
    ptr(handler)->handle(expr->as<ScatterOp>());
    return;
  }
  if (expr->isStrictlyA<RNGOp>()) {
    ptr(handler)->handle(expr->as<RNGOp>());
    return;
  }
  if (expr->isStrictlyA<ReductionOp>()) {
    ptr(handler)->handle(expr->as<ReductionOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedReductionOp>()) {
    ptr(handler)->handle(expr->as<GroupedReductionOp>());
    return;
  }
  if (expr->isStrictlyA<WelfordOp>()) {
    ptr(handler)->handle(expr->as<WelfordOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedWelfordOp>()) {
    ptr(handler)->handle(expr->as<GroupedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<LoadStoreOp>()) {
    ptr(handler)->handle(expr->as<LoadStoreOp>());
    return;
  }
  if (expr->isStrictlyA<MmaOp>()) {
    ptr(handler)->handle(expr->as<MmaOp>());
    return;
  }
  if (expr->isStrictlyA<BroadcastOp>()) {
    ptr(handler)->handle(expr->as<BroadcastOp>());
    return;
  }
  if (expr->isStrictlyA<SqueezeOp>()) {
    ptr(handler)->handle(expr->as<SqueezeOp>());
    return;
  }
  if (expr->isStrictlyA<Split>()) {
    ptr(handler)->handle(expr->as<Split>());
    return;
  }
  if (expr->isStrictlyA<Merge>()) {
    ptr(handler)->handle(expr->as<Merge>());
    return;
  }
  if (expr->isStrictlyA<Swizzle2D>()) {
    ptr(handler)->handle(expr->as<Swizzle2D>());
    return;
  }
  if (expr->isStrictlyA<TransposeOp>()) {
    ptr(handler)->handle(expr->as<TransposeOp>());
    return;
  }
  if (expr->isStrictlyA<ExpandOp>()) {
    ptr(handler)->handle(expr->as<ExpandOp>());
    return;
  }
  if (expr->isStrictlyA<ShiftOp>()) {
    ptr(handler)->handle(expr->as<ShiftOp>());
    return;
  }
  if (expr->isStrictlyA<GatherOp>()) {
    ptr(handler)->handle(expr->as<GatherOp>());
    return;
  }
  if (expr->isStrictlyA<ViewAsScalar>()) {
    ptr(handler)->handle(expr->as<ViewAsScalar>());
    return;
  }
  if (expr->isStrictlyA<ViewOp>()) {
    ptr(handler)->handle(expr->as<ViewOp>());
    return;
  }
  if (expr->isStrictlyA<kir::Allocate>()) {
    ptr(handler)->handle(expr->as<kir::Allocate>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSync>()) {
    ptr(handler)->handle(expr->as<kir::BlockSync>());
    return;
  }
  if (expr->isStrictlyA<kir::GridSync>()) {
    ptr(handler)->handle(expr->as<kir::GridSync>());
    return;
  }
  if (expr->isStrictlyA<kir::CpAsyncWait>()) {
    ptr(handler)->handle(expr->as<kir::CpAsyncWait>());
    return;
  }
  if (expr->isStrictlyA<kir::CpAsyncCommit>()) {
    ptr(handler)->handle(expr->as<kir::CpAsyncCommit>());
    return;
  }
  if (expr->isStrictlyA<kir::InitMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::InitMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::UpdateMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::ForLoop>()) {
    ptr(handler)->handle(expr->as<kir::ForLoop>());
    return;
  }
  if (expr->isStrictlyA<kir::IfThenElse>()) {
    ptr(handler)->handle(expr->as<kir::IfThenElse>());
    return;
  }
  if (expr->isStrictlyA<kir::GridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GridBroadcast>()) {
    ptr(handler)->handle(expr->as<kir::GridBroadcast>());
    return;
  }
  if (expr->isStrictlyA<kir::GridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::VectorizedWelfordOp>()) {
    ptr(handler)->handle(expr->as<kir::VectorizedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<kir::AllocateFusedReduction>()) {
    ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::SMemAddress>()) {
    ptr(handler)->handle(expr->as<kir::SMemAddress>());
    return;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
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
      switch (std::get<PrimDataType>(val->getDataType()->type)) {
        case DataType::Bool:
          ptr(mutator)->mutate(val->as<Bool>());
          return;
        case DataType::Half:
        case DataType::BFloat16:
        case DataType::Float:
        case DataType::Double:
          ptr(mutator)->mutate(val->as<Double>());
          return;
        case DataType::Int:
        case DataType::Int32:
        case DataType::Index:
        case DataType::SMemAddress:
          ptr(mutator)->mutate(val->as<Int>());
          return;
        case DataType::ComplexFloat:
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
    case ValType::Attribute:
      TORCH_INTERNAL_ASSERT(
          false,
          "ValType::Attribute can not be dispatched. Template type is needed.");
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
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
        false,
        "Handle not overriden for ",
        stmt->as<Expr>()->getOpString(),
        ".");
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
        false,
        "Handle not overriden for ",
        stmt->as<Expr>()->getOpString(),
        ".");
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

// Exprs
void OptOutConstDispatch::handle(const FullOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IotaOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const EyeOp* stmt) {
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
void OptOutConstDispatch::handle(const SelectOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IndexSelectOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TorchGatherOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ScatterOp* stmt) {
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
void OptOutConstDispatch::handle(const SqueezeOp* stmt) {
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
void OptOutConstDispatch::handle(const kir::VectorizedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::SMemAddress* stmt) {
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

// Exprs
void OptOutDispatch::handle(FullOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IotaOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(EyeOp* stmt) {
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
void OptOutDispatch::handle(SelectOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IndexSelectOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TorchGatherOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ScatterOp* stmt) {
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
void OptOutDispatch::handle(SqueezeOp* stmt) {
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
void OptOutDispatch::handle(kir::VectorizedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::SMemAddress* stmt) {
  unhandled(stmt);
}

} // namespace nvfuser
