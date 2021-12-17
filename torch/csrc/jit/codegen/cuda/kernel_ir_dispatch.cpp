#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

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
  switch (val->vtype()) {
    case ValType::Scalar:
      switch (val->dtype()) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
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
    case ValType::Predicate:
      ptr(handler)->handle(val->as<Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<TensorIndex>());
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  switch (expr->etype()) {
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
    case ExprType::WelfordOp:
      ptr(handler)->handle(expr->as<WelfordOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;
    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<Allocate>());
      return;
    case ExprType::Sync:
      ptr(handler)->handle(expr->as<Sync>());
      return;
    case ExprType::InitMagicZero:
      ptr(handler)->handle(expr->as<InitMagicZero>());
      return;
    case ExprType::UpdateMagicZero:
      ptr(handler)->handle(expr->as<UpdateMagicZero>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<IfThenElse>());
      return;
    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<GridReduction>());
      return;
    case ExprType::GridBroadcast:
      ptr(handler)->handle(expr->as<GridBroadcast>());
      return;
    case ExprType::GridWelford:
      ptr(handler)->handle(expr->as<GridWelford>());
      return;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Node::dispatch(T handler, Node* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->handle(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->handle(stmt->as<Expr>());
  } else
    TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

template <typename T>
void Val::constDispatch(T handler, const Val* val) {
  switch (val->vtype()) {
    case ValType::Scalar:
      switch (val->dtype()) {
        case DataType::Bool:
          ptr(handler)->handle(val->as<Bool>());
          return;
        case DataType::Double:
          ptr(handler)->handle(val->as<Double>());
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
    case ValType::Predicate:
      ptr(handler)->handle(val->as<Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<TensorIndex>());
      return;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown valtype in dispatch!");
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
  switch (expr->etype()) {
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
    case ExprType::WelfordOp:
      ptr(handler)->handle(expr->as<WelfordOp>());
      return;
    case ExprType::BroadcastOp:
      ptr(handler)->handle(expr->as<BroadcastOp>());
      return;
    case ExprType::Allocate:
      ptr(handler)->handle(expr->as<Allocate>());
      return;
    case ExprType::Sync:
      ptr(handler)->handle(expr->as<Sync>());
      return;
    case ExprType::InitMagicZero:
      ptr(handler)->handle(expr->as<InitMagicZero>());
      return;
    case ExprType::UpdateMagicZero:
      ptr(handler)->handle(expr->as<UpdateMagicZero>());
      return;
    case ExprType::ForLoop:
      ptr(handler)->handle(expr->as<ForLoop>());
      return;
    case ExprType::IfThenElse:
      ptr(handler)->handle(expr->as<IfThenElse>());
      return;
    case ExprType::GridReduction:
      ptr(handler)->handle(expr->as<GridReduction>());
      return;
    case ExprType::GridBroadcast:
      ptr(handler)->handle(expr->as<GridBroadcast>());
      return;
    case ExprType::GridWelford:
      ptr(handler)->handle(expr->as<GridWelford>());
      return;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Node::constDispatch(T handler, const Node* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->handle(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->handle(stmt->as<Expr>());
  } else
    TORCH_INTERNAL_ASSERT(false, "Unknown stmttype in dispatch!");
}

/*
 * Handler template instantiations. These should only have to be done on base
 * classes. Actual visitors/mutators should inhereit from these classes and call
 * ->dispatch(this) to avoid needing an explicit instantiation.
 */
template void Node::dispatch(OptOutDispatch, Node*);
template void Node::dispatch(OptOutDispatch*, Node*);
template void Val::dispatch(OptOutDispatch, Val*);
template void Val::dispatch(OptOutDispatch*, Val*);
template void Expr::dispatch(OptOutDispatch, Expr*);
template void Expr::dispatch(OptOutDispatch*, Expr*);

template void Node::dispatch(OptInDispatch, Node*);
template void Node::dispatch(OptInDispatch*, Node*);
template void Val::dispatch(OptInDispatch, Val*);
template void Val::dispatch(OptInDispatch*, Val*);
template void Expr::dispatch(OptInDispatch, Expr*);
template void Expr::dispatch(OptInDispatch*, Expr*);

template void Node::constDispatch(OptOutConstDispatch, const Node*);
template void Node::constDispatch(OptOutConstDispatch*, const Node*);
template void Val::constDispatch(OptOutConstDispatch, const Val*);
template void Val::constDispatch(OptOutConstDispatch*, const Val*);
template void Expr::constDispatch(OptOutConstDispatch, const Expr*);
template void Expr::constDispatch(OptOutConstDispatch*, const Expr*);

template void Node::constDispatch(OptInConstDispatch, const Node*);
template void Node::constDispatch(OptInConstDispatch*, const Node*);
template void Val::constDispatch(OptInConstDispatch, const Val*);
template void Val::constDispatch(OptInConstDispatch*, const Val*);
template void Expr::constDispatch(OptInConstDispatch, const Expr*);
template void Expr::constDispatch(OptInConstDispatch*, const Expr*);

void OptOutDispatch::handle(Node* s) {
  Node::dispatch(this, s);
}

void OptOutDispatch::handle(Expr* e) {
  Expr::dispatch(this, e);
}

void OptOutDispatch::handle(Val* v) {
  Val::dispatch(this, v);
}

void OptOutConstDispatch::handle(const Node* s) {
  Node::constDispatch(this, s);
}

void OptOutConstDispatch::handle(const Expr* e) {
  Expr::constDispatch(this, e);
}

void OptOutConstDispatch::handle(const Val* v) {
  Val::constDispatch(this, v);
}

void OptInConstDispatch::unhandled(const Node* stmt) {
  if (stmt->isExpr()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getExprType().value(), ".");
  } else if (stmt->isVal()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    TORCH_INTERNAL_ASSERT("Unrecognized Node type.");
  }
}

void OptInDispatch::unhandled(Node* stmt) {
  if (stmt->isExpr()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getExprType().value(), ".");
  } else if (stmt->isVal()) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    TORCH_INTERNAL_ASSERT("Unrecognized Node type.");
  }
}

// Vals
void OptOutConstDispatch::handle(const IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorView* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Bool* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Double* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Int* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Predicate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorIndex* stmt) {
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
void OptOutConstDispatch::handle(const ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const BroadcastOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Allocate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Sync* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GridWelford* stmt) {
  unhandled(stmt);
}

// Vals
void OptOutDispatch::handle(IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorView* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Bool* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Double* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Int* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Predicate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorIndex* stmt) {
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
void OptOutDispatch::handle(ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(BroadcastOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Allocate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Sync* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GridWelford* stmt) {
  unhandled(stmt);
}

std::vector<Expr*> KirVisitor::handle(const std::vector<Expr*>& exprs) {
  exprs_ = std::vector<Expr*>(exprs);
  for (auto expr : exprs) {
    handle(expr);
  }
  return exprs_;
}

void KirVisitor::handle(ForLoop* fl) {
  for_loops_.push_back(fl);
  scope_.push_back(&fl->body());
  auto body_exprs = std::vector<Expr*>(fl->body().exprs());
  for (auto expr : body_exprs) {
    handle(expr);
  }
  scope_.pop_back();
  for_loops_.pop_back();
}

void KirVisitor::handle(IfThenElse* ite) {
  scope_.push_back(&ite->thenBody());
  auto then_exprs = std::vector<Expr*>(ite->thenBody().exprs());
  for (auto expr : then_exprs) {
    handle(expr);
  }
  scope_.pop_back();

  scope_.push_back(&ite->elseBody());
  auto else_exprs = std::vector<Expr*>(ite->elseBody().exprs());
  for (auto expr : else_exprs) {
    handle(expr);
  }
  scope_.pop_back();
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
