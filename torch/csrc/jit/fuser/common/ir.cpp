#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Statement member definitions & related functions
 */

// When we create a Val or EXPR we immediately register them with the active
// fusion.
Val::Val(ValType _vtype, DataType _dtype)
    : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion != nullptr) {
    this->name_ = fusion->registerVal(this);
    this->fusion_ = fusion;
  } else {
    throw std::runtime_error("No fusion group found when creating a Val.");
  }
}

Expr* Val::getOrigin() {
  FusionGuard fg(fusion_);
  return (fusion_->origin(this));
}

Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    throw std::runtime_error("No fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

UnaryOp::UnaryOp(UnaryOpType _type, Val* _out, Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(
    BinaryOpType _type,
    Val* _out,
    Val* _lhs,
    Val* _rhs)
    : Expr(ExprType::BinaryOp),
      binary_op_type_{_type},
      out_{_out},
      lhs_{_lhs},
      rhs_{_rhs} {
  addOutput(_out);
  addInput(_lhs);
  addInput(_rhs);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Statement::~Statement() {}

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
 * ptr(mutator)->handle(static_cast<Statement*>(this));
 */

template <typename T>
void Val::dispatch(T handler, Val* val) {
  switch (*(val->getValType())) {
    case ValType::TensorDomain:
      ptr(handler)->handle(static_cast<TensorDomain*>(val));
      return;
    case ValType::TensorView:
      ptr(handler)->handle(static_cast<TensorView*>(val));
      return;
    case ValType::IterDomain:
      ptr(handler)->handle(static_cast<IterDomain*>(val));
      return;
    case ValType::Tensor:
      ptr(handler)->handle(static_cast<Tensor*>(val));
      return;
    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Float:
          ptr(handler)->handle(static_cast<Float*>(val));
          return;
        case DataType::Int:
          ptr(handler)->handle(static_cast<Int*>(val));
          return;
        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Unknown valtype in dispatch!");
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::UnaryOp:
      ptr(handler)->handle(static_cast<UnaryOp*>(expr));
      return;
    case ExprType::BinaryOp:
      ptr(handler)->handle(static_cast<BinaryOp*>(expr));
      return;
    case ExprType::Split:
      ptr(handler)->handle(static_cast<Split*>(expr));
      return;
    case ExprType::Merge:
      ptr(handler)->handle(static_cast<Merge*>(expr));
      return;
    case ExprType::Reorder:
      ptr(handler)->handle(static_cast<Reorder*>(expr));
      return;
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Statement::dispatch(T handler, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->handle(static_cast<Val*>(stmt));
  } else if (stmt->isExpr()) {
    ptr(handler)->handle(static_cast<Expr*>(stmt));
  } else
    throw std::runtime_error("Unknown stmttype in dispatch!");
}

/*
 * Generic dispatch for any handler that modifies the IR. This could be a
 * transformation on loop structures, or parallelizing a loop. This dispatch is
 * paired with a class that implements the functions template <typenname
 * node_type> Statement* mutate(node_type* node) mutate should call
 * (statement* node_to_dispatch)->dispatch_mutator() It could also implement
 * Statement* mutate(Statement* stmt){
 *   stmt->dispatch_mutator(this);
 * }
 * And therefore dispatch_mutator should never call:
 *   ptr(mutator)->mutate(static_cast<Statement*>(this));
 */
template <typename T>
Statement* Val::dispatch_mutator(T mutator, Val* val) {
  switch (*(val->getValType())) {
    case ValType::Tensor:
      return ptr(mutator)->mutate(static_cast<Tensor*>(val));

    case ValType::TensorDomain:
      return ptr(mutator)->mutate(static_cast<TensorDomain*>(val));

    case ValType::TensorView:
      return ptr(mutator)->mutate(static_cast<TensorView*>(val));

    case ValType::IterDomain:
      return ptr(mutator)->mutate(static_cast<IterDomain*>(val));

    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Float:
          return ptr(mutator)->mutate(static_cast<Float*>(val));

        case DataType::Int:
          return ptr(mutator)->mutate(static_cast<Int*>(val));

        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Unknown valtype in dispatch_mutator!");
}

template <typename T>
Statement* Expr::dispatch_mutator(T mutator, Expr* expr) {
  switch (*(expr->getExprType())) {
    case ExprType::UnaryOp:
      return ptr(mutator)->mutate(static_cast<UnaryOp*>(expr));
    case ExprType::BinaryOp:
      return ptr(mutator)->mutate(static_cast<BinaryOp*>(expr));
    case ExprType::Split:
      return ptr(mutator)->mutate(static_cast<Split*>(expr));
    case ExprType::Merge:
      return ptr(mutator)->mutate(static_cast<Merge*>(expr));
    case ExprType::Reorder:
      return ptr(mutator)->mutate(static_cast<Reorder*>(expr));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch_mutator!");
  }
}

template <typename T>
Statement* Statement::dispatch_mutator(T mutator, Statement* stmt) {
  if (stmt->isVal()) {
    return ptr(mutator)->mutate(static_cast<Val*>(stmt));
  }
  if (stmt->isExpr()) {
    return ptr(mutator)->mutate(static_cast<Expr*>(stmt));
  }
  throw std::runtime_error("Unknown stmttype in dispatch_mutator!");
}

/*
 * Handler template instantiations. These should only have to be done on base classes.
 * Actual visitors/mutators should inhereit from these classes and call ->dispatch(this)
 * to avoid needing an explicit instantiation.
 */
template void Statement::dispatch(IterVisitor, Statement*);
template void Statement::dispatch(IterVisitor*, Statement*);
template void Val::dispatch(IterVisitor, Val*);
template void Val::dispatch(IterVisitor*, Val*);
template void Expr::dispatch(IterVisitor, Expr*);
template void Expr::dispatch(IterVisitor*, Expr*);

template Statement* Statement::dispatch_mutator(BaseMutator, Statement*);
template Statement* Statement::dispatch_mutator(BaseMutator*, Statement*);
template Statement* Val::dispatch_mutator(BaseMutator, Val*);
template Statement* Val::dispatch_mutator(BaseMutator*, Val*);
template Statement* Expr::dispatch_mutator(BaseMutator, Expr*);
template Statement* Expr::dispatch_mutator(BaseMutator*, Expr*);

/*
 * Val member definitions
 */

Val::~Val() {}

/*
 * IRInputOutput member definitions
 */

IRInputOutput::~IRInputOutput() {}

/*
 * Expr member definitions
 */

Expr::~Expr() {}

} // namespace fuser
} // namespace jit
} // namespace torch
